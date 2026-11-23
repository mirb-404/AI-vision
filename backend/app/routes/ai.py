"""AI assistance endpoints: SAM click/box-to-polygon and YOLO pre-labels."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from .. import config
from ..db import get_session
from ..inference import registry
from ..inference.sam import load_rgb
from ..schemas import (AiStatus, DetectRequest, DetectResponse, EmbedResponse, SegmentRequest,
                       SegmentResponse)
from ..tables import Event
from .common import image_or_404

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
