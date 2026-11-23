"""Effort and model-latency summary computed from the event log."""

import statistics
from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..db import get_session
from ..tables import Event, LabelingSession
from .common import project_or_404

router = APIRouter(prefix="/api", tags=["stats"])


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    if len(values) == 1:
        q1 = q3 = values[0]
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"n": len(values), "median": statistics.median(values), "q1": q1, "q3": q3,
            "mean": statistics.fmean(values)}


@router.get("/projects/{project_id}/stats")
def project_stats(project_id: int, db: Session = Depends(get_session)):
    project_or_404(db, project_id)
    events = db.exec(select(Event).where(Event.project_id == project_id)).all()
    sessions = db.exec(select(LabelingSession).where(LabelingSession.project_id == project_id)).all()

    seconds: dict[str, list[float]] = defaultdict(list)
    clicks: dict[str, list[float]] = defaultdict(list)
    by_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    proposals = {"accepted": 0, "rejected": 0, "edited_after_accept": 0}
    ai_deleted_after_accept = 0
    latency: dict[str, list[float]] = defaultdict(list)

    for e in events:
        p = e.payload or {}
        if e.kind == "annotation_committed" and e.mode:
            seconds[e.mode].append(float(p.get("duration_ms", 0)) / 1000)
            clicks[e.mode].append(float(p.get("clicks", 0)))
            by_tool[e.mode][str(p.get("tool", "unknown"))] += 1
        elif e.kind == "proposal_accepted":
            proposals["accepted"] += 1
        elif e.kind == "proposal_rejected":
            proposals["rejected"] += 1
        elif e.kind == "annotation_edited" and p.get("source_before") == "ai":
            proposals["edited_after_accept"] += 1
        elif e.kind == "annotation_deleted" and p.get("source") in ("ai", "ai_edited"):
            ai_deleted_after_accept += 1
        elif e.kind == "inference":
            latency[f"{p.get('model')}/{p.get('stage')}"].append(float(p.get("ms", 0)))

    per_mode = {
        mode: {
            "objects": len(seconds[mode]),
            "seconds_per_object": _summary(seconds[mode]),
            "clicks_per_object": _summary(clicks[mode]),
            "by_tool": dict(by_tool[mode]),
            "sessions": sum(1 for s in sessions if s.mode == mode),
        }
        for mode in ("manual", "ai")
    }
    manual_med = per_mode["manual"]["seconds_per_object"]["median"]
    ai_med = per_mode["ai"]["seconds_per_object"]["median"]
    speed_up = manual_med / ai_med if manual_med and ai_med else None

    reviewed = proposals["accepted"] + proposals["rejected"]
    return {
        "per_mode": per_mode,
        "speed_up_median": speed_up,
        "proposals": {
            **proposals,
            "deleted_after_accept": ai_deleted_after_accept,
            "reviewed": reviewed,
            "accept_rate": proposals["accepted"] / reviewed if reviewed else None,
            "edit_rate": proposals["edited_after_accept"] / proposals["accepted"] if proposals["accepted"] else None,
        },
        "inference_ms": {k: _summary(v) for k, v in sorted(latency.items())},
    }
