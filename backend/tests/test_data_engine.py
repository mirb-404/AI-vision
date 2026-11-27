"""Round metrics of training/data_engine.py from a small fake effort log."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
import data_engine as de  # noqa: E402


def _write(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ts", "session_id", "participant", "mode", "kind", "image_id", "annotation_id", "payload"])
        for i, (sec, who, kind, img, ann, payload) in enumerate(rows):
            w.writerow([i, f"2026-11-26T10:00:{sec:02d}", 1, who, "ai", kind, img, ann, json.dumps(payload)])


def test_round_stats(tmp_path):
    p = tmp_path / "events.csv"
    _write(p, [
        (0, "round0", "proposal_accepted", 1, "a", {}),
        (1, "round0", "annotation_committed", 1, "a", {"source": "ai"}),
        (2, "round0", "proposal_rejected", 1, "", {}),
        (5, "round0", "annotation_committed", 1, "b", {"source": "manual"}),
        (9, "round0", "annotation_deleted", 1, "a", {"source": "ai"}),
        (0, "round1", "annotation_committed", 2, "c", {"source": "ai"}),
        (2, "round1", "annotation_committed", 2, "d", {"source": "ai"}),
    ])
    rows = de.load(p)
    r0 = de.round_stats([r for r in rows if r["participant"] == "round0"])
    r1 = de.round_stats([r for r in rows if r["participant"] == "round1"])
    assert r0["ai_kept"] == 0 and r0["hand_objects"] == 1 and r0["ai_kept_share"] == 0
    assert r0["accept_rate"] == 0.5 and r0["ai_fix_rate"] == 1.0 and r0["seconds_per_image"] == 9
    assert r1["ai_kept_share"] == 1.0 and r1["seconds_per_object"] == 1.0
