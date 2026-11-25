"""AI assistance endpoints: SAM click/box-to-polygon and YOLO pre-labels."""

import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import config
from ..db import get_session
from ..inference import owl as owl_lib
from ..inference import registry
from ..inference.sam import load_rgb
from ..schemas import (AiStatus, AutoLabelRequest, AutoLabelResponse, DetectRequest, DetectResponse,
                       EmbedResponse, SegmentRequest, SegmentResponse, SimilarRegion, SimilarRequest,
                       SimilarResponse, UndoAiRequest, UndoAiResponse)
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


@router.post("/images/{image_id}/similar", response_model=SimilarResponse)
def find_similar(image_id: int, body: SimilarRequest, db: Session = Depends(get_session)):
    """Find every object of the same kind as the example the user marked (one-shot, no training).

    E.g. mark one cricket ball, get the other balls, also ones of another colour or size. Uses
    OWLv2 image-guided detection. Objects already annotated with the same label (and the example
    itself) are skipped, so running it again does not add duplicates. Results are boxes; when the
    example is an outline and SAM is available, each box is turned into an outline by SAM.
    With save=True they are stored as AI annotations of the example's kind.
    """
    target = image_or_404(db, image_id)
    source = image_or_404(db, body.source_image_id)
    if source.project_id != target.project_id:
        raise HTTPException(400, "source and target images belong to different projects")
    label = db.get(Label, body.label_id)
    if label is None or label.project_id != target.project_id:
        raise HTTPException(404, "label not found in this project")
    owl = _model(registry.get_owl)

    if body.box is not None:
        b = body.box
        example = np.array([b.x, b.y, b.x + b.w, b.y + b.h], dtype=np.float32)
        kind = "box"
    else:
        pts = np.asarray(body.polygon, dtype=np.float32).reshape(-1, 2)
        example = np.array([*pts.min(axis=0), *pts.max(axis=0)], dtype=np.float32)
        kind = "polygon"
    if example[2] <= example[0] or example[3] <= example[1]:
        raise HTTPException(400, "marked object has no area")

    src_patches, _, ms_src = owl.embed(source.id, config.IMAGES_DIR / source.stored_name)
    try:
        query = owl_lib.select_query(src_patches, example)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    threshold = owl_lib.auto_threshold(src_patches, query, example, body.strictness)
    patches, _, ms_tgt = owl.embed(target.id, config.IMAGES_DIR / target.stored_name)
    found = owl_lib.detect_like(patches, query, threshold)

    # Skip the example and objects that already carry this label.
    taken = [example] if target.id == source.id else []
    same_label = db.exec(select(Annotation).where(Annotation.image_id == target.id,
                                                  Annotation.label_id == label.id)).all()
    for a in same_label:
        g = a.geometry
        if a.kind == "box":
            taken.append(np.array([g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]], dtype=np.float32))
        else:
            p = np.asarray(g["points"], dtype=np.float32).reshape(-1, 2)
            taken.append(np.array([*p.min(axis=0), *p.max(axis=0)], dtype=np.float32))
    taken_arr = np.stack(taken) if taken else np.zeros((0, 4), dtype=np.float32)

    sam = None
    if kind == "polygon":
        try:
            sam = registry.get_sam()
        except registry.ModelUnavailable:
            sam = None  # outlines fall back to the box rectangle

    regions = []
    now = utcnow()
    for xyxy, score in found:
        x1, y1 = max(float(xyxy[0]), 0.0), max(float(xyxy[1]), 0.0)
        x2, y2 = min(float(xyxy[2]), float(target.width)), min(float(xyxy[3]), float(target.height))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
        if len(taken_arr) and float(owl_lib.box_iou(taken_arr, box_xyxy).max()) >= 0.5:
            continue
        box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
        if sam is not None:
            outline, _, _, _ = sam.segment(target.id, config.IMAGES_DIR / target.stored_name, [], (x1, y1, x2, y2))
            if len(outline) >= 6:
                polygon = outline
        region_out = SimilarRegion(polygon=polygon, box=box, score=score)
        if body.save:
            region_out.id = uuid.uuid4().hex
            geometry = box if kind == "box" else {"points": polygon}
            db.add(Annotation(id=region_out.id, image_id=target.id, label_id=label.id,  # type: ignore[arg-type]
                              kind=kind, geometry=geometry,
                              source="ai", score=score, created_at=now, updated_at=now))
        regions.append(region_out)
        taken_arr = np.concatenate([taken_arr, box_xyxy[None]])

    ms = ms_src + ms_tgt
    db.add(Event(project_id=target.project_id, image_id=target.id, kind="inference",
                 payload={"model": "owlv2", "stage": "similar", "ms": round(ms, 2), "regions": len(regions)}))
    if body.save and regions and target.status == "new":
        target.status = "in_progress"
        db.add(target)
    db.commit()
    return SimilarResponse(regions=regions, latency_ms=round(ms, 1))


@router.post("/similar/undo", response_model=UndoAiResponse)
def undo_similar(body: UndoAiRequest, db: Session = Depends(get_session)):
    """Delete AI annotations created by a similar-region search (only untouched AI ones)."""
    if not body.ids:
        return UndoAiResponse(deleted=0)
    rows = db.exec(select(Annotation).where(Annotation.id.in_(body.ids),  # type: ignore[union-attr]
                                            Annotation.source == "ai")).all()
    for a in rows:
        db.delete(a)
    db.commit()
    return UndoAiResponse(deleted=len(rows))
