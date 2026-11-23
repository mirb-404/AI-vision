from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import get_session
from ..schemas import LabelCreate, ProjectCreate, ProjectOut
from ..tables import Annotation, Image, Label, Project
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
