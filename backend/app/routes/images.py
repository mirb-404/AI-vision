import io
import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError
from sqlmodel import Session, select

from .. import config
from ..db import get_session
from ..schemas import ImageOut, ImagePatch, UploadError, UploadResult
from ..tables import Annotation, Image
from .common import annotation_counts, image_or_404, project_or_404

router = APIRouter(prefix="/api", tags=["images"])

ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}


def image_out(image: Image, count: int) -> ImageOut:
    return ImageOut(
        id=image.id,  # type: ignore[arg-type]
        project_id=image.project_id,
        filename=image.filename,
        width=image.width,
        height=image.height,
        status=image.status,  # type: ignore[arg-type]
        annotation_count=count,
    )


def _store_upload(data: bytes) -> tuple[str, int, int]:
    """Validate and save one image. Returns (stored_name, width, height).

    EXIF orientation is baked into the pixels so browser display, model input
    and stored coordinates all agree.
    """
    with PILImage.open(io.BytesIO(data)) as probe:
        fmt = probe.format
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"unsupported format {fmt}; use JPEG, PNG, WEBP or BMP")
        probe.load()
        oriented = ImageOps.exif_transpose(probe)
        stored_name = f"{uuid.uuid4().hex}{ALLOWED_FORMATS[fmt]}"
        path = config.IMAGES_DIR / stored_name
        if oriented is probe:
            path.write_bytes(data)
        else:
            save_kwargs = {"quality": 95} if fmt in ("JPEG", "WEBP") else {}
            oriented.save(path, format=fmt, **save_kwargs)
        return stored_name, oriented.width, oriented.height


@router.post("/projects/{project_id}/images", response_model=UploadResult, status_code=201)
def upload_images(project_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    uploaded: list[ImageOut] = []
    errors: list[UploadError] = []
    for f in files:
        name = f.filename or "upload"
        data = f.file.read(limit + 1)
        if len(data) > limit:
            errors.append(UploadError(filename=name, error=f"file larger than {config.MAX_UPLOAD_MB} MB"))
            continue
        try:
            stored_name, width, height = _store_upload(data)
        except (UnidentifiedImageError, PILImage.DecompressionBombError, ValueError, OSError) as exc:
            errors.append(UploadError(filename=name, error=str(exc) or "not a valid image"))
            continue
        image = Image(project_id=project_id, filename=name, stored_name=stored_name, width=width, height=height)
        db.add(image)
        db.commit()
        db.refresh(image)
        uploaded.append(image_out(image, 0))
    return UploadResult(uploaded=uploaded, errors=errors)


@router.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(project_id: int, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    images = db.exec(select(Image).where(Image.project_id == project_id).order_by(Image.id)).all()  # type: ignore[arg-type]
    counts = annotation_counts(db, [i.id for i in images if i.id is not None])
    return [image_out(i, counts.get(i.id, 0)) for i in images]  # type: ignore[arg-type]


@router.get("/images/{image_id}", response_model=ImageOut)
def get_image(image_id: int, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    return image_out(image, annotation_counts(db, [image_id]).get(image_id, 0))


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    return FileResponse(config.IMAGES_DIR / image.stored_name)


@router.patch("/images/{image_id}", response_model=ImageOut)
def patch_image(image_id: int, body: ImagePatch, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    image.status = body.status
    db.add(image)
    db.commit()
    db.refresh(image)
    return image_out(image, annotation_counts(db, [image_id]).get(image_id, 0))


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_session)):
    image = image_or_404(db, image_id)
    for ann in db.exec(select(Annotation).where(Annotation.image_id == image_id)).all():
        db.delete(ann)
    db.delete(image)
    db.commit()
    (config.IMAGES_DIR / image.stored_name).unlink(missing_ok=True)
