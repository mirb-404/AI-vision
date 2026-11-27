"""Mini "data engine" report: does fine-tuning on your own labels make the assistant more useful?

Idea from the SAM paper (Kirillov et al. 2023): label with the model, retrain on the corrected
labels, label more; time per object fell from 34 s to 14 s over 6 retrains. Here at small scale:

  Round 0  label batch A in the app with the pretrained YOLO (session name "round0")
           export YOLO zip -> prepare_app_export.py -> train.py --variant full -> export_yolo.py
  Round 1  restart the app with the fine-tuned model (YOLO_MODEL=yolo_round1.onnx), label a NEW
           batch B (session name "round1")
  Report   python training/data_engine.py project_1_events.csv

Full commands are in training/README.md ("Data engine round"). Rounds are told apart by the
session name typed when the labeling session starts ("participant"); --rounds picks which ones.

For each round, from the app's effort log (events.csv, "Download events" on the stats page):
  AI kept share   AI objects kept / all final objects (higher = the AI did more of the work)
  accept rate     suggestions accepted / suggestions reviewed (auto-accepted count as accepted)
  AI fix rate     AI objects the user later edited or deleted / AI objects added
  hand objects    objects the user still had to draw
  s / image       active time per image (first to last event on that image, gaps > 2 min cut)

Different images per round (needed, else the user remembers the images), so keep batches similar
in difficulty and say so in the report. Writes <out>/data_engine.md and data_engine.json.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

AI_SOURCES = {"ai", "ai_edited"}
IDLE_CAP_S = 120  # a pause longer than this (coffee break) is not labeling time


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["payload"] = json.loads(r["payload"] or "{}")
        r["ts"] = datetime.fromisoformat(r["ts"])
    return rows


def round_stats(rows: list[dict]) -> dict:
    ai_added: set[str] = set()
    hand = 0
    ai_fixed: set[str] = set()
    ai_deleted: set[str] = set()
    accepted = rejected = 0
    times: dict[str, list[datetime]] = defaultdict(list)
    for r in rows:
        p, kind, ann = r["payload"], r["kind"], r["annotation_id"]
        if r["image_id"]:
            times[r["image_id"]].append(r["ts"])
        if kind == "annotation_committed":
            if p.get("source") in AI_SOURCES:
                ai_added.add(ann)
            else:
                hand += 1
        elif kind == "proposal_accepted":
            accepted += 1
        elif kind == "proposal_rejected":
            rejected += 1
        elif kind == "annotation_edited" and p.get("source_before") in AI_SOURCES:
            ai_fixed.add(ann)
        elif kind == "annotation_deleted" and p.get("source") in AI_SOURCES:
            ai_deleted.add(ann)

    seconds = []
    for stamps in times.values():
        stamps.sort()
        active = sum(min((b - a).total_seconds(), IDLE_CAP_S) for a, b in zip(stamps, stamps[1:]))
        seconds.append(active)
    kept = len(ai_added - ai_deleted)
    final = kept + hand
    reviewed = accepted + rejected
    return {
        "images": len(times),
        "ai_added": len(ai_added),
        "ai_kept": kept,
        "hand_objects": hand,
        "ai_kept_share": kept / final if final else None,
        "accept_rate": accepted / reviewed if reviewed else None,
        "ai_fix_rate": len((ai_fixed | ai_deleted) & ai_added) / len(ai_added) if ai_added else None,
        "seconds_per_image": sorted(seconds)[len(seconds) // 2] if seconds else None,
        "seconds_per_object": sum(seconds) / final if final else None,
    }


def fmt(v, pct=False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.0f} %" if pct else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=Path, help="events.csv downloaded from the app")
    parser.add_argument("--rounds", nargs="+", default=["round0", "round1"], help="session names, in order")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rows = load(args.events)
    by_round = defaultdict(list)
    for r in rows:
        if r["participant"] in args.rounds and r["mode"] == "ai":
            by_round[r["participant"]].append(r)
    missing = [n for n in args.rounds if n not in by_round]
    if missing:
        found = sorted({r["participant"] for r in rows if r["participant"]})
        raise SystemExit(f"no AI-mode events for session(s) {missing}; sessions in the file: {found}")

    stats = {name: round_stats(by_round[name]) for name in args.rounds}
    lines = ["# Data engine: pretrained vs fine-tuned assistant", "",
             "| Round | Images | AI kept share | Accept rate | AI fix rate | Hand objects | s / image | s / object |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['images']} | {fmt(s['ai_kept_share'], True)} | {fmt(s['accept_rate'], True)} | "
                     f"{fmt(s['ai_fix_rate'], True)} | {s['hand_objects']} | {fmt(s['seconds_per_image'])} | "
                     f"{fmt(s['seconds_per_object'])} |")
    first, last = stats[args.rounds[0]], stats[args.rounds[-1]]
    if first["seconds_per_object"] and last["seconds_per_object"]:
        lines += ["", f"Time per object {args.rounds[0]} -> {args.rounds[-1]}: "
                      f"{first['seconds_per_object']:.1f} s -> {last['seconds_per_object']:.1f} s "
                      f"({first['seconds_per_object'] / last['seconds_per_object']:.2f}x)."]
    lines += ["", "Different image batches per round: compare with care (batch difficulty may differ).",
              "For a same-images comparison use training/evaluate.py on a held-out test set."]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_engine.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "data_engine.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
