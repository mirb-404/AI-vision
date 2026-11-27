"""Request/response bodies. Validation of client data happens here."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["manual", "ai"]
Source = Literal["manual", "ai", "ai_edited"]
ImageStatus = Literal["new", "in_progress", "done"]
EventKind = Literal[
    "image_opened",
    "image_done",
    "annotation_committed",
    "annotation_edited",
    "annotation_deleted",
    "proposal_accepted",
    "proposal_rejected",
]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    image_count: int
    done_count: int
    annotation_count: int


class LabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: str = Field(default="#3b82f6", pattern=r"^#[0-9a-fA-F]{6}$")


class ImageOut(BaseModel):
    id: int
    project_id: int
    filename: str
    width: int
    height: int
    status: ImageStatus
    annotation_count: int


class ImagePatch(BaseModel):
    status: ImageStatus


class UploadError(BaseModel):
    filename: str
    error: str


class UploadResult(BaseModel):
    uploaded: list[ImageOut]
    errors: list[UploadError]


class AnnotationIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label_id: int
    kind: Literal["box", "polygon"]
    geometry: dict[str, Any]
    source: Source = "manual"
    score: float | None = None

    @model_validator(mode="after")
    def check_geometry(self) -> "AnnotationIn":
        g = self.geometry
        if self.kind == "box":
            try:
                x, y, w, h = (float(g[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("box geometry needs numeric x, y, w, h") from exc
            if w <= 0 or h <= 0:
                raise ValueError("box w and h must be > 0")
            self.geometry = {"x": x, "y": y, "w": w, "h": h}
        else:
            pts = g.get("points")
            if not isinstance(pts, list) or len(pts) < 6 or len(pts) % 2:
                raise ValueError("polygon geometry needs 'points' with >= 3 (x, y) pairs")
            try:
                self.geometry = {"points": [float(v) for v in pts]}
            except (TypeError, ValueError) as exc:
                raise ValueError("polygon points must be numbers") from exc
        return self


class AnnotationOut(AnnotationIn):
    image_id: int


class AnnotationsPut(BaseModel):
    annotations: list[AnnotationIn] = Field(max_length=5000)


class Box(BaseModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class PromptPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1]  # 1 = foreground, 0 = background


class SegmentRequest(BaseModel):
    points: list[PromptPoint] = Field(default_factory=list, max_length=50)
    box: Box | None = None

    @model_validator(mode="after")
    def need_prompt(self) -> "SegmentRequest":
        if not self.points and self.box is None:
            raise ValueError("give at least one point or a box")
        return self


class SegmentResponse(BaseModel):
    polygon: list[float]
    score: float
    latency_ms: float
    embed_ms: float


class DetectRequest(BaseModel):
    conf: float = Field(default=0.1, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    # Project label names. Names the detector does not know are searched with OWLv2 text queries
    # (open vocabulary), so suggestions also work for custom labels such as "rhino" or "scratch".
    names: list[str] = Field(default_factory=list, max_length=50)


class Detection(BaseModel):
    label: str
    class_id: int
    score: float
    box: Box
    model: str = "yolo"  # which detector found it ("yolo", "yolo_coco", "owlv2")


class DetectResponse(BaseModel):
    detections: list[Detection]
    latency_ms: float


class AutoLabelRequest(BaseModel):
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    # Images that already have annotations are left alone unless this is False
    # (then detections are added next to the existing annotations).
    skip_labeled: bool = True


class AutoLabelResponse(BaseModel):
    images_labeled: int
    images_skipped: int
    annotations_created: int
    labels_created: list[str]
    latency_ms: float


class EmbedResponse(BaseModel):
    latency_ms: float
    cached: bool


class ModelStatus(BaseModel):
    available: bool  # weight files present
    loaded: bool
    detail: str


class AiStatus(BaseModel):
    sam: ModelStatus
    yolo: ModelStatus
    owl: ModelStatus  # OWLv2, find alike objects
    owl_text: ModelStatus  # OWLv2 text tower, find by name


class SessionCreate(BaseModel):
    mode: Mode
    participant: str = Field(default="", max_length=100)


class SessionOut(BaseModel):
    id: int
    project_id: int
    mode: Mode
    participant: str
    started_at: datetime


class EventIn(BaseModel):
    session_id: int | None = None
    image_id: int | None = None
    annotation_id: str | None = Field(default=None, max_length=64)
    kind: EventKind
    ts: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventsIn(BaseModel):
    events: list[EventIn] = Field(max_length=1000)


class SimilarRequest(BaseModel):
    """Find objects alike one marked example (box or polygon on the source image) in this image.

    Found objects come back (and are saved) as the same kind as the example: box or polygon.
    """
    source_image_id: int
    box: Box | None = None
    polygon: list[float] | None = Field(default=None, max_length=20000)
    label_id: int
    # 0 = loose (more, rougher matches) ... 1 = strict (only close matches)
    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    # False: only return the regions (the UI adds them to the open image itself).
    save: bool = True

    @model_validator(mode="after")
    def need_region(self) -> "SimilarRequest":
        if self.box is None and (self.polygon is None or len(self.polygon) < 6 or len(self.polygon) % 2):
            raise ValueError("give a box or a polygon with at least 3 points")
        return self


class NextImage(BaseModel):
    image_id: int
    # 0 = the AI is sure about everything it sees ... 1 = it is unsure (or sees nothing it knows)
    uncertainty: float
    detections: int  # boxes at score >= 0.1
    unsure: int  # boxes with a borderline score (0.2-0.6)


class NextImagesResponse(BaseModel):
    images: list[NextImage]
    ranked: int  # unlabeled images that were scored
    latency_ms: float


class TextSearchRequest(BaseModel):
    """Find every object matching a typed name ("car", "helmet") in this image (OWLv2 text query).

    Found objects come back (and are saved) as boxes with label_id.
    """
    text: str = Field(min_length=1, max_length=60)
    label_id: int
    # 0 = loose (more, rougher matches) ... 1 = strict (only confident matches)
    strictness: float = Field(default=0.5, ge=0.0, le=1.0)
    save: bool = True


class SimilarRegion(BaseModel):
    polygon: list[float]
    box: Box
    score: float
    id: str | None = None  # annotation id when saved


class SimilarResponse(BaseModel):
    regions: list[SimilarRegion]
    latency_ms: float


class UndoAiRequest(BaseModel):
    ids: list[str] = Field(max_length=10000)


class UndoAiResponse(BaseModel):
    deleted: int
