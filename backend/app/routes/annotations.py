from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..schemas import AnnotationOut, AnnotationsPut
from ..tables import Annotation, Label, utcnow
from .common import image_or_404

router = APIRouter(prefix="/api", tags=["annotations"])


def _out(a: Annotation) -> AnnotationOut:
    return AnnotationOut(
        id=a.id,
        image_id=a.image_id,
        label_id=a.label_id,
        kind=a.kind,  # type: ignore[arg-type]
        geometry=a.geometry,
        source=a.source,  # type: ignore[arg-type]
        score=a.score,
    )


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationOut])
def get_annotations(image_id: int, db: Session = Depends(get_session)):
    image_or_404(db, image_id)
    rows = db.exec(select(Annotation).where(Annotation.image_id == image_id).order_by(Annotation.created_at)).all()  # type: ignore[arg-type]
    return [_out(a) for a in rows]


@router.put("/images/{image_id}/annotations", response_model=list[AnnotationOut])
def put_annotations(image_id: int, body: AnnotationsPut, db: Session = Depends(get_session)):
    """Replace all annotations of one image (the client autosaves the full list)."""
    image = image_or_404(db, image_id)
    label_ids = set(db.exec(select(Label.id).where(Label.project_id == image.project_id)).all())
    ids = [a.id for a in body.annotations]
    if len(ids) != len(set(ids)):
        raise HTTPException(422, "duplicate annotation ids")
    for a in body.annotations:
        if a.label_id not in label_ids:
            raise HTTPException(422, f"label {a.label_id} does not belong to this project")

    existing = {a.id: a for a in db.exec(select(Annotation).where(Annotation.image_id == image_id)).all()}
    clash = db.exec(select(Annotation.id).where(Annotation.id.in_(ids), Annotation.image_id != image_id)).first()  # type: ignore[union-attr]
    if clash is not None:
        raise HTTPException(409, f"annotation id {clash} belongs to another image")

    now = utcnow()
    keep = set(ids)
    for ann_id, row in existing.items():
        if ann_id not in keep:
            db.delete(row)
    for a in body.annotations:
        row = existing.get(a.id)
        if row is None:
            db.add(Annotation(id=a.id, image_id=image_id, label_id=a.label_id, kind=a.kind,
                              geometry=a.geometry, source=a.source, score=a.score, created_at=now, updated_at=now))
        else:
            changed = (row.label_id, row.kind, row.geometry, row.source, row.score) != (
                a.label_id, a.kind, a.geometry, a.source, a.score)
            if changed:
                row.label_id, row.kind, row.geometry = a.label_id, a.kind, a.geometry
                row.source, row.score, row.updated_at = a.source, a.score, now
                db.add(row)

    if body.annotations and image.status == "new":
        image.status = "in_progress"
        db.add(image)
    elif not body.annotations and image.status == "in_progress":
        image.status = "new"
        db.add(image)
    db.commit()
    return get_annotations(image_id, db)
