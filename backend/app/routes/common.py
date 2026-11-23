from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from ..tables import Annotation, Image, Project


def project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return project


def image_or_404(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


def annotation_counts(db: Session, image_ids: list[int]) -> dict[int, int]:
    if not image_ids:
        return {}
    rows = db.exec(
        select(Annotation.image_id, func.count())
        .where(Annotation.image_id.in_(image_ids))  # type: ignore[union-attr]
        .group_by(Annotation.image_id)
    ).all()
    return {image_id: count for image_id, count in rows}
