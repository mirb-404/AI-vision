from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..schemas import LabelCreate, ProjectCreate, ProjectOut, ResetRequest, ResetResponse
from ..tables import Annotation, Event, Image, Label, LabelingSession, Project
from .common import annotation_counts, project_or_404

router = APIRouter(prefix="/api", tags=["projects"])


def _project_out(db: Session, project: Project) -> ProjectOut:
    images = db.exec(select(Image).where(Image.project_id == project.id)).all()
    counts = annotation_counts(db, [i.id for i in images if i.id is not None])
    return ProjectOut(
        id=project.id,  # type: ignore[arg-type]
        name=project.name,
        created_at=project.created_at,
        image_count=len(images),
        done_count=sum(1 for i in images if i.status == "done"),
        annotation_count=sum(counts.values()),
    )


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_session)):
    projects = db.exec(select(Project).order_by(Project.created_at.desc())).all()  # type: ignore[attr-defined]
    return [_project_out(db, p) for p in projects]


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_session)):
    project = Project(name=body.name.strip())
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(db, project)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_session)):
    return _project_out(db, project_or_404(db, project_id))


@router.get("/projects/{project_id}/labels", response_model=list[Label])
def list_labels(project_id: int, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    return db.exec(select(Label).where(Label.project_id == project_id).order_by(Label.id)).all()  # type: ignore[arg-type]


@router.post("/projects/{project_id}/labels", response_model=Label, status_code=201)
def create_label(project_id: int, body: LabelCreate, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    name = body.name.strip()
    existing = db.exec(select(Label).where(Label.project_id == project_id)).all()
    if any(l.name.lower() == name.lower() for l in existing):
        raise HTTPException(409, f"label '{name}' already exists")
    label = Label(project_id=project_id, name=name, color=body.color)
    db.add(label)
    db.commit()
    db.refresh(label)
    return label


@router.delete("/labels/{label_id}", status_code=204)
def delete_label(label_id: int, db: Session = Depends(get_session)):
    label = db.get(Label, label_id)
    if label is None:
        raise HTTPException(404, "label not found")
    in_use = db.exec(select(Annotation).where(Annotation.label_id == label_id).limit(1)).first()
    if in_use is not None:
        raise HTTPException(409, "label is used by annotations; delete those first")
    db.delete(label)
    db.commit()


@router.post("/projects/{project_id}/reset", response_model=ResetResponse)
def reset_project(project_id: int, body: ResetRequest, db: Session = Depends(get_session)):
    """Delete every annotation of the project and mark all images "new" again (images are kept).

    Optional: also delete the effort log + labeling sessions (clear_stats) and the labels
    (remove_labels). Cannot be undone; the UI asks for confirmation first.
    """
    project_or_404(db, project_id)
    images = db.exec(select(Image).where(Image.project_id == project_id)).all()
    ids = [i.id for i in images]
    anns = db.exec(select(Annotation).where(Annotation.image_id.in_(ids))).all() if ids else []  # type: ignore[union-attr]
    for a in anns:
        db.delete(a)
    reset = 0
    for image in images:
        if image.status != "new":
            image.status = "new"
            db.add(image)
            reset += 1
    events = 0
    if body.clear_stats:
        rows = db.exec(select(Event).where(Event.project_id == project_id)).all()
        events = len(rows)
        for e in rows:
            db.delete(e)
        db.flush()  # events reference sessions: remove them first
        for s in db.exec(select(LabelingSession).where(LabelingSession.project_id == project_id)).all():
            db.delete(s)
    labels = 0
    if body.remove_labels:
        db.flush()  # annotations reference labels: remove them first
        rows = db.exec(select(Label).where(Label.project_id == project_id)).all()
        labels = len(rows)
        for l in rows:
            db.delete(l)
    db.commit()
    return ResetResponse(annotations_deleted=len(anns), images_reset=reset, events_deleted=events,
                         labels_deleted=labels)
