"""Labeling sessions and the effort log used to measure manual vs AI-assisted labeling."""

import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from ..db import get_session
from ..schemas import EventsIn, SessionCreate, SessionOut
from ..tables import Event, LabelingSession, utcnow
from .common import project_or_404

router = APIRouter(prefix="/api", tags=["events"])


@router.post("/projects/{project_id}/sessions", response_model=SessionOut, status_code=201)
def create_session(project_id: int, body: SessionCreate, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    session = LabelingSession(project_id=project_id, mode=body.mode, participant=body.participant.strip())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/projects/{project_id}/events", status_code=204)
def post_events(project_id: int, body: EventsIn, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    session_ids = {e.session_id for e in body.events if e.session_id is not None}
    sessions = {s.id: s for s in db.exec(select(LabelingSession).where(LabelingSession.id.in_(session_ids))).all()}  # type: ignore[union-attr]
    for sid in session_ids:
        if sid not in sessions or sessions[sid].project_id != project_id:
            raise HTTPException(422, f"session {sid} does not belong to this project")
    for e in body.events:
        mode = sessions[e.session_id].mode if e.session_id is not None else None
        db.add(Event(project_id=project_id, session_id=e.session_id, image_id=e.image_id,
                     annotation_id=e.annotation_id, mode=mode, kind=e.kind, ts=e.ts or utcnow(), payload=e.payload))
    db.commit()


@router.get("/projects/{project_id}/events.csv")
def events_csv(project_id: int, db: Session = Depends(get_session)):
    """Raw effort log for offline analysis (eval scripts read this file)."""
    project_or_404(db, project_id)
    rows = db.exec(select(Event).where(Event.project_id == project_id).order_by(Event.id)).all()  # type: ignore[arg-type]
    participants = {s.id: s.participant for s in db.exec(
        select(LabelingSession).where(LabelingSession.project_id == project_id)).all()}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "ts", "session_id", "participant", "mode", "kind", "image_id", "annotation_id", "payload"])
    for e in rows:
        writer.writerow([e.id, e.ts.isoformat(), e.session_id, participants.get(e.session_id, ""), e.mode, e.kind,
                         e.image_id, e.annotation_id, json.dumps(e.payload)])
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="project_{project_id}_events.csv"'},
    )
