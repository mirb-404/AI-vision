"""AI assistance endpoints: SAM click/box-to-polygon, YOLO pre-labels, OWLv2 find alike / find by name."""

import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import config
from ..db import get_session
from ..inference import owl as owl_lib
from ..inference import registry
from ..inference.sam import load_rgb
from ..inference.yolo import merge_detections
from ..schemas import (AiStatus, AutoLabelRequest, AutoLabelResponse, DetectRequest, DetectResponse,
                       EmbedResponse, NextImage, NextImagesResponse, SegmentRequest, SegmentResponse, SimilarRegion, SimilarRequest,
                       SimilarResponse, TextSearchRequest, UndoAiRequest, UndoAiResponse)
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


MAX_OPEN_VOCAB_NAMES = 20  # OWLv2 text queries per image (each costs a cheap scoring pass)


def open_vocab_detections(image: Image, names: list[str], known: set[str]) -> tuple[list[dict], float]:
    """Boxes for label names the detector does not know, from OWLv2 text queries.

    YOLO only knows its training classes (80 for COCO, 601 for Open Images V7). A project label
    outside them ("rhino", "scratch") would never get a suggestion; OWLv2 is open-vocabulary, so it
    looks for the name itself. Returns ([] , 0) when OWLv2 is not installed: suggestions then come
    from YOLO alone, as before. Detections use class_id -1 (no detector class).
    """
    wanted = []
    for n in names:
        n = n.strip()
        if n and n.lower() not in known and n.lower() not in {w.lower() for w in wanted}:
            wanted.append(n)
    if not wanted:
        return [], 0.0
    try:
        owl, text = registry.get_owl(), registry.get_owl_text()
    except registry.ModelUnavailable:
        return [], 0.0
    patches, _, ms = owl.embed(image.id, config.IMAGES_DIR / image.stored_name)
    out = []
    for name in wanted[:MAX_OPEN_VOCAB_NAMES]:
        query, ms_text = text.encode(name)
        ms += ms_text
        for xyxy, score in owl_lib.detect_like(patches, query, owl_lib.text_threshold(0.5)):
            x1, y1 = max(float(xyxy[0]), 0.0), max(float(xyxy[1]), 0.0)
            x2, y2 = min(float(xyxy[2]), float(image.width)), min(float(xyxy[3]), float(image.height))
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                out.append({"label": name, "class_id": -1, "score": score, "model": "owlv2",
                            "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}})
    return out, ms


@router.post("/images/{image_id}/detect", response_model=DetectResponse)
def detect(image_id: int, body: DetectRequest, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    yolo = _model(registry.get_yolo)
    detections, ms = yolo.detect(load_rgb(config.IMAGES_DIR / image.stored_name), body.conf, body.iou)
    _log_inference(db, image.project_id, image_id, "yolo", "detect", ms)
    extra, ms_owl = open_vocab_detections(image, body.names, {n.lower() for n in yolo.names.values()})
    if ms_owl:
        _log_inference(db, image.project_id, image_id, "owlv2", "detect_text", ms_owl)
    if body.vocab:
        vocab, ms_vocab = vocab_detections(image)
        if ms_vocab or vocab:
            _log_inference(db, image.project_id, image_id, "owlv2", "detect_vocab", ms_vocab)
        # LVIS boxes merged with the detectors' (same object twice -> the more confident one);
        # scenery boxes over the whole image are dropped from the LVIS list.
        detections = merge_detections([detections, vocab], float(image.width * image.height), drop_giant={1})
        ms_owl += ms_vocab
    return DetectResponse(detections=detections + extra, latency_ms=ms + ms_owl)


def vocab_detections(image: Image) -> tuple[list[dict], float]:
    """Boxes of any of the 1203 LVIS classes (OWLv2, one image pass + one matrix product)."""
    try:
        owl, (names, embeds) = registry.get_owl(), registry.get_owl_vocab()
    except registry.ModelUnavailable:
        return [], 0.0
    patches, _, ms = owl.embed(image.id, config.IMAGES_DIR / image.stored_name)
    out = []
    for name, xyxy, score in owl_lib.detect_vocab(patches, embeds, names):  # type: ignore[arg-type]
        x1, y1 = max(float(xyxy[0]), 0.0), max(float(xyxy[1]), 0.0)
        x2, y2 = min(float(xyxy[2]), float(image.width)), min(float(xyxy[3]), float(image.height))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            out.append({"label": name, "class_id": -1, "score": score, "model": "owlv2_lvis",
                        "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}})
    return out, ms


@router.post("/projects/{project_id}/auto-label", response_model=AutoLabelResponse)
def auto_label_project(project_id: int, body: AutoLabelRequest, db: Session = Depends(get_session)):
    """Run YOLO on every image of the project and save the detections as annotations.

    Project labels YOLO does not know are searched with OWLv2 text queries (when installed).
    When the project has labels, only those classes are saved; with no labels, every class is.

    Detections become boxes with source "ai" and their score. Class names map to
    project labels by name (case-insensitive); missing labels are created.
    Images that already have annotations are skipped unless skip_labeled is False.
    """
    project_or_404(db, project_id)
    yolo = _model(registry.get_yolo)
    images = db.exec(select(Image).where(Image.project_id == project_id).order_by(Image.id)).all()  # type: ignore[arg-type]
    counts = annotation_counts(db, [i.id for i in images])  # type: ignore[misc]
    labels = {l.name.lower(): l for l in db.exec(select(Label).where(Label.project_id == project_id)).all()}
    project_names = [l.name for l in labels.values()]  # labels YOLO may not know: OWLv2 looks for them
    wanted = {n.lower() for n in project_names}
    known = {n.lower() for n in yolo.names.values()}
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
        extra, ms_owl = open_vocab_detections(image, project_names, known)
        if ms_owl:
            total_ms += ms_owl
            db.add(Event(project_id=project_id, image_id=image.id, kind="inference",
                         payload={"model": "owlv2", "stage": "auto_label_text", "ms": round(ms_owl, 2)}))
        detections = detections + [d for d in extra if d["score"] >= body.conf]
        if project_names:  # the project says what to label: skip other classes (same rule as the editor)
            detections = [d for d in detections if d["label"].lower() in wanted]
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


def uncertainty(scores: list[float]) -> float:
    """Image uncertainty for "label these next" (uncertainty sampling, active learning).

    Mean of 1 - |2s - 1| over the detector's boxes: 0 when every box is sure (s near 0 or 1), 1 when
    every box is a coin flip (s = 0.5). An image with no box at all scores 1: the model sees nothing
    it knows, so a human look teaches the most (new classes, or the image really is empty).
    """
    if not scores:
        return 1.0
    return float(np.mean([1 - abs(2 * s - 1) for s in scores]))


@router.post("/projects/{project_id}/next-images", response_model=NextImagesResponse)
def next_images(project_id: int, limit: int = 5, db: Session = Depends(get_session)):
    """Unlabeled images ranked by how unsure the detector is about them, most unsure first.

    Labeling these first gives the most new information per image when the labels are later used
    to fine-tune the model (see training/data_engine.py). Only images with status "new" are scored.
    """
    project_or_404(db, project_id)
    yolo = _model(registry.get_yolo)
    images = db.exec(select(Image).where(Image.project_id == project_id, Image.status == "new")).all()
    ranked, total_ms = [], 0.0
    for image in images:
        detections, ms = yolo.detect(load_rgb(config.IMAGES_DIR / image.stored_name), 0.1, 0.45)
        total_ms += ms
        scores = [float(d["score"]) for d in detections]
        ranked.append(NextImage(image_id=image.id, uncertainty=round(uncertainty(scores), 4),  # type: ignore[arg-type]
                                detections=len(scores), unsure=sum(0.2 <= s <= 0.6 for s in scores)))
    ranked.sort(key=lambda r: -r.uncertainty)
    db.add(Event(project_id=project_id, kind="inference",
                 payload={"model": "yolo", "stage": "next_images", "ms": round(total_ms, 2), "images": len(ranked)}))
    db.commit()
    return NextImagesResponse(images=ranked[:max(limit, 1)], ranked=len(ranked), latency_ms=round(total_ms, 1))


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

    regions = _store_found(db, target, label, kind, found, [example] if target.id == source.id else [], body.save)

    ms = ms_src + ms_tgt
    _finish_search(db, target, "similar", ms, regions, body.save)
    return SimilarResponse(regions=regions, latency_ms=round(ms, 1))


@router.post("/images/{image_id}/text-search", response_model=SimilarResponse)
def text_search(image_id: int, body: TextSearchRequest, db: Session = Depends(get_session)):
    """Find every object matching a typed name ("car", "helmet") in this image, as boxes.

    OWLv2 open-vocabulary detection with a text query: works for names outside the YOLO classes.
    Objects already annotated with the label are skipped, so running it again adds no duplicates.
    """
    target = image_or_404(db, image_id)
    label = db.get(Label, body.label_id)
    if label is None or label.project_id != target.project_id:
        raise HTTPException(404, "label not found in this project")
    owl = _model(registry.get_owl)
    text = _model(registry.get_owl_text)
    query, ms_text = text.encode(body.text)
    patches, _, ms_img = owl.embed(target.id, config.IMAGES_DIR / target.stored_name)
    found = owl_lib.detect_like(patches, query, owl_lib.text_threshold(body.strictness))
    regions = _store_found(db, target, label, "box", found, [], body.save)
    ms = ms_text + ms_img
    _finish_search(db, target, "text", ms, regions, body.save)
    return SimilarResponse(regions=regions, latency_ms=round(ms, 1))


def _store_found(db: Session, target: Image, label: Label, kind: str, found: list[tuple[np.ndarray, float]],
                 exclude: list[np.ndarray], save: bool) -> list[SimilarRegion]:
    """Turn OWLv2 boxes into regions (and AI annotations when save=True).

    Skips boxes overlapping `exclude` (e.g. the example) or an object that already has this label
    (IoU >= 0.5). For kind "polygon", SAM outlines each box when it is available.
    """
    taken = list(exclude)
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
        if save:
            region_out.id = uuid.uuid4().hex
            geometry = box if kind == "box" else {"points": polygon}
            db.add(Annotation(id=region_out.id, image_id=target.id, label_id=label.id,  # type: ignore[arg-type]
                              kind=kind, geometry=geometry,
                              source="ai", score=score, created_at=now, updated_at=now))
        regions.append(region_out)
        taken_arr = np.concatenate([taken_arr, box_xyxy[None]])
    return regions


def _finish_search(db: Session, target: Image, stage: str, ms: float, regions: list[SimilarRegion], save: bool) -> None:
    """Log the OWLv2 run and mark the image as started when results were saved."""
    db.add(Event(project_id=target.project_id, image_id=target.id, kind="inference",
                 payload={"model": "owlv2", "stage": stage, "ms": round(ms, 2), "regions": len(regions)}))
    if save and regions and target.status == "new":
        target.status = "in_progress"
        db.add(target)
    db.commit()


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
