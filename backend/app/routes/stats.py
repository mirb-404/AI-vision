"""Effort and model-latency summary computed from the event log."""

import statistics
from collections import defaultdict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from ..db import get_session
from ..tables import Annotation, Event, Image, Label, LabelingSession
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


AI_SOURCES = ("ai", "ai_edited")


def _rates(c: dict[str, int]) -> dict[str, float | int | None]:
    hit = c["ai_kept"] + c["ai_fixed"]
    final = hit + c["hand"]
    return {
        **c,
        # AI labels the user kept (as-is or fixed) / all AI labels the user saw
        "precision": hit / (hit + c["ai_removed"]) if hit + c["ai_removed"] else None,
        # objects the AI found / all objects in the final labels
        "recall": hit / final if final else None,
        # final labels the AI got right with no change at all
        "exact_share": c["ai_kept"] / final if final else None,
    }


@router.get("/projects/{project_id}/scorecard")
def scorecard(project_id: int, download: bool = False, db: Session = Depends(get_session)):
    """How good were the AI's labels, judged against the user's final labels (no extra ground truth).

    Per label and in total, on images where the AI labeled something (only images marked done,
    when there are any):
      ai_kept    AI labels left unchanged          (right)
      ai_fixed   AI labels the user edited         (found, but imprecise)
      ai_removed AI labels the user deleted        (false positives)
      hand       labels the user drew themselves   (objects the AI missed)
    plus precision, recall, exact share, and the effort summary (time per object, speed-up).
    """
    project = project_or_404(db, project_id)
    images = db.exec(select(Image).where(Image.project_id == project_id)).all()
    ids = [i.id for i in images]
    anns = db.exec(select(Annotation).where(Annotation.image_id.in_(ids))).all() if ids else []  # type: ignore[union-attr]
    events = db.exec(select(Event).where(Event.project_id == project_id)).all()
    labels = {l.id: l.name for l in db.exec(select(Label).where(Label.project_id == project_id)).all()}

    present = {a.id for a in anns}
    committed_label = {e.annotation_id: (e.payload or {}).get("label_id") for e in events
                       if e.kind == "annotation_committed" and e.annotation_id}
    # Which model produced each AI label: the accepted suggestion's detector, or OWLv2 for
    # find alike / find by name. Older events without it count as "unknown".
    made_by: dict[str, str] = {}
    for e in events:
        p = e.payload or {}
        if not e.annotation_id:
            continue
        if e.kind == "proposal_accepted":
            made_by[e.annotation_id] = str(p.get("model") or "unknown")
        elif e.kind == "annotation_committed" and p.get("tool") in ("similar", "text"):
            made_by[e.annotation_id] = f"owlv2_{p['tool']}"
    removed = [e for e in events if e.kind == "annotation_deleted" and (e.payload or {}).get("source") in AI_SOURCES
               and e.annotation_id not in present]  # deleted and not brought back by Undo

    ai_images = {a.image_id for a in anns if a.source in AI_SOURCES} | {e.image_id for e in removed}
    done = {i.id for i in images if i.status == "done"}
    scope = ai_images & done if ai_images & done else ai_images

    zero = {"ai_kept": 0, "ai_fixed": 0, "ai_removed": 0, "hand": 0}
    per: dict[str, dict[str, int]] = defaultdict(lambda: dict(zero))
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"ai_kept": 0, "ai_fixed": 0, "ai_removed": 0})
    for a in anns:
        if a.image_id not in scope:
            continue
        key = {"ai": "ai_kept", "ai_edited": "ai_fixed"}.get(a.source, "hand")
        per[labels.get(a.label_id, "?")][key] += 1
        if key != "hand":
            by_model[made_by.get(a.id, "unknown")][key] += 1
    for e in {e.annotation_id: e for e in removed}.values():
        if e.image_id in scope:
            per[labels.get(committed_label.get(e.annotation_id), "?")]["ai_removed"] += 1  # type: ignore[arg-type]
            by_model[made_by.get(e.annotation_id or "", "unknown")]["ai_removed"] += 1

    total = dict(zero)
    for c in per.values():
        for k in total:
            total[k] += c[k]
    effort = project_stats(project_id, db)
    body = {
        "project": project.name,
        "images": len(scope),
        "images_scope": "done" if scope and scope <= done else "all with AI labels",
        "total": _rates(total),
        "per_label": {name: _rates(c) for name, c in sorted(per.items())},
        # Which detector's labels survived review (precision per model; recall needs "hand").
        "per_model": {m: {**c, "precision": (c["ai_kept"] + c["ai_fixed"]) / t if (t := sum(c.values())) else None}
                      for m, c in sorted(by_model.items())},
        "effort": {
            "seconds_per_object": {m: effort["per_mode"][m]["seconds_per_object"]["median"] for m in ("manual", "ai")},
            "speed_up_median": effort["speed_up_median"],
            "accept_rate": effort["proposals"]["accept_rate"],
        },
    }
    if download:
        return JSONResponse(body, headers={"Content-Disposition": f'attachment; filename="project_{project_id}_ai_scorecard.json"'})
    return body
