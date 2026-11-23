import os
import re
import tempfile
import zipfile
from collections import defaultdict
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from starlette.background import BackgroundTask

from .. import config
from ..db import get_session
from ..export.formats import write_coco_zip, write_yolo_zip
from ..tables import Annotation, Image, Label
from .common import project_or_404

router = APIRouter(prefix="/api", tags=["export"])


@router.get("/projects/{project_id}/export")
def export_project(project_id: int, format: Literal["coco", "yolo"] = "coco", only_done: bool = False,
                   db: Session = Depends(get_session)):
    project = project_or_404(db, project_id)
    query = select(Image).where(Image.project_id == project_id)
    if only_done:
        query = query.where(Image.status == "done")
    images = db.exec(query.order_by(Image.id)).all()  # type: ignore[arg-type]
    labels = db.exec(select(Label).where(Label.project_id == project_id).order_by(Label.id)).all()  # type: ignore[arg-type]
    annotations: dict[int, list[Annotation]] = defaultdict(list)
    if images:
        for a in db.exec(select(Annotation).where(Annotation.image_id.in_([i.id for i in images]))  # type: ignore[union-attr]
                         .order_by(Annotation.created_at)).all():  # type: ignore[arg-type]
            annotations[a.image_id].append(a)

    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if format == "coco":
            write_coco_zip(zf, project.name, list(images), list(labels), annotations, config.IMAGES_DIR)
        else:
            write_yolo_zip(zf, list(images), list(labels), annotations, config.IMAGES_DIR)

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", project.name).strip("_") or f"project_{project_id}"
    return FileResponse(tmp_path, media_type="application/zip", filename=f"{slug}_{format}.zip",
                        background=BackgroundTask(os.unlink, tmp_path))
