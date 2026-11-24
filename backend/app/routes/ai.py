"""AI assistance endpoints: SAM click/box-to-polygon and YOLO pre-labels."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import config
from ..db import get_session
from ..inference import registry
from ..inference.sam import load_rgb
from ..schemas import (AiStatus, AutoLabelRequest, AutoLabelResponse, DetectRequest, DetectResponse,
                       EmbedResponse, SegmentRequest, SegmentResponse)
from ..tables import Annotation, Event, Image, Label, utcnow
from .common import annotation_counts, image_or_404, project_or_404

# Same palette as the UI (frontend/src/lib/geometry.ts LABEL_COLORS).
LABEL_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7",
                "#f97316", "#06b6d4", "#ec4899", "#84cc16", "#6366f1"]

router = APIRouter(prefix="/api", tags=["ai"])


def _log_inference(db: Session, project_id: int, image_id: int, model: str, stage: str, ms: float) -> None:
    db.add(Event(project_id=project_id, image_id=image_id, kind="inference",
                 payload={"model": model, "stage": stage, "ms": round(ms, 2)}))
    db.commit()


def _model(getter):
    try:
        return getter()
    except registry.ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/ai/status", response_model=AiStatus)
def ai_status():
    return registry.status()


@router.post("/images/{image_id}/embed", response_model=EmbedResponse)
def embed(image_id: int, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    sam = _model(registry.get_sam)
    _, _, ms, cached = sam.embed(image_id, config.IMAGES_DIR / image.stored_name)
    if not cached:
        _log_inference(db, image.project_id, image_id, "sam", "encoder", ms)
    return EmbedResponse(latency_ms=ms, cached=cached)


@router.post("/images/{image_id}/segment", response_model=SegmentResponse)
def segment(image_id: int, body: SegmentRequest, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    sam = _model(registry.get_sam)
    box = None
    if body.box is not None:
        b = body.box
        box = (b.x, b.y, b.x + b.w, b.y + b.h)
    points = [(p.x, p.y, p.label) for p in body.points]
    polygon, score, decode_ms, embed_ms = sam.segment(image_id, config.IMAGES_DIR / image.stored_name, points, box)
    if embed_ms:
        _log_inference(db, image.project_id, image_id, "sam", "encoder", embed_ms)
    _log_inference(db, image.project_id, image_id, "sam", "decoder", decode_ms)
    return SegmentResponse(polygon=polygon, score=score, latency_ms=decode_ms, embed_ms=embed_ms)


@router.post("/images/{image_id}/detect", response_model=DetectResponse)
def detect(image_id: int, body: DetectRequest, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    yolo = _model(registry.get_yolo)
    detections, ms = yolo.detect(load_rgb(config.IMAGES_DIR / image.stored_name), body.conf, body.iou)
    _log_inference(db, image.project_id, image_id, "yolo", "detect", ms)
    return DetectResponse(detections=detections, latency_ms=ms)


@router.post("/projects/{project_id}/auto-label", response_model=AutoLabelResponse)
def auto_label_project(project_id: int, body: AutoLabelRequest, db: Session = Depends(get_session)):
    """Run YOLO on every image of the project and save the detections as annotations.

    Detections become boxes with source "ai" and their score. Class names map to
    project labels by name (case-insensitive); missing labels are created.
    Images that already have annotations are skipped unless skip_labeled is False.
    """
    project_or_404(db, project_id)
    yolo = _model(registry.get_yolo)
    images = db.exec(select(Image).where(Image.project_id == project_id).order_by(Image.id)).all()  # type: ignore[arg-type]
    counts = annotation_counts(db, [i.id for i in images])  # type: ignore[misc]
    labels = {l.name.lower(): l for l in db.exec(select(Label).where(Label.project_id == project_id)).all()}
    used_colors = [l.color for l in labels.values()]
    created_labels: list[str] = []
    labeled = skipped = created = 0
    total_ms = 0.0

    for image in images:
        if body.skip_labeled and counts.get(image.id, 0):  # type: ignore[arg-type]
            skipped += 1
            continue
        detections, ms = yolo.detect(load_rgb(config.IMAGES_DIR / image.stored_name), body.conf, body.iou)
        total_ms += ms
        db.add(Event(project_id=project_id, image_id=image.id, kind="inference",
                     payload={"model": "yolo", "stage": "auto_label", "ms": round(ms, 2)}))
        now = utcnow()
        for d in detections:
            box = {k: float(d["box"][k]) for k in ("x", "y", "w", "h")}
            if box["w"] <= 0 or box["h"] <= 0:  # the UI's PUT validation would reject it later
                continue
            label = labels.get(d["label"].lower())
            if label is None:
                color = next((c for c in LABEL_COLORS if c not in used_colors),
                             LABEL_COLORS[len(used_colors) % len(LABEL_COLORS)])
                label = Label(project_id=project_id, name=d["label"], color=color)
                db.add(label)
                db.flush()
                labels[d["label"].lower()] = label
                used_colors.append(color)
                created_labels.append(d["label"])
            db.add(Annotation(id=uuid.uuid4().hex, image_id=image.id, label_id=label.id,  # type: ignore[arg-type]
                              kind="box", geometry=box,
                              source="ai", score=float(d["score"]), created_at=now, updated_at=now))
            created += 1
        if detections and image.status == "new":
            image.status = "in_progress"
            db.add(image)
        labeled += 1
        db.commit()

    return AutoLabelResponse(images_labeled=labeled, images_skipped=skipped, annotations_created=created,
                             labels_created=created_labels, latency_ms=round(total_ms, 1))
