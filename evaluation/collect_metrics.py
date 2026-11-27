"""Collect every metric of the project into one bundle to share for analysis.

Run on the Linux machine from the repo root, while the app is running (for the app's numbers):
    python3 evaluation/collect_metrics.py                       # app at http://localhost:8000
    python3 evaluation/collect_metrics.py --app-url http://localhost:8000 --no-events

Writes metrics_bundle_<date>/ and metrics_bundle_<date>.zip with:
    METRICS.md        everything readable in one file (send this one if you can only send one)
    environment.json  git commit, GPU, Python, model files and sizes
    training/         training/results/**  (summary.md/csv, per_class.csv, metrics.json, data_engine.*)
    evaluation/       evaluation/results/** (NoC benchmark: summary.md, metrics.json, curve.csv, per_image.csv)
    app/              per project: stats.json, scorecard.json, events.csv; ai_status.json
Only text files (.md .csv .json); no images, weights or datasets. Standard library only.

Privacy: events.csv contains the participant names typed in the app. Use ids like P1, P2, or
pass --no-events to leave the raw log out (stats and scorecard are still included).
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".csv", ".json"}
MAX_INLINE = 60_000  # characters of one file copied into METRICS.md


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment() -> dict:
    weights = ROOT / "models" / "weights"
    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "model_files": {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(weights.glob("*"))
                        if p.is_file() and p.name != ".gitkeep"},
    }


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def collect_app(base: str, out: Path, events: bool) -> list[str]:
    """Stats, scorecard (and raw events) for every project. Returns notes for METRICS.md."""
    base = base.rstrip("/")
    try:
        projects = json.loads(get(f"{base}/api/projects"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return [f"App not reachable at {base} ({exc}); app numbers skipped. Start ./run.sh and run again."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "ai_status.json").write_bytes(get(f"{base}/api/ai/status"))
    notes = []
    for p in projects:
        d = out / f"project_{p['id']}"
        d.mkdir(exist_ok=True)
        (d / "project.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
        for name, path in (("stats.json", "stats"), ("scorecard.json", "scorecard")):
            try:
                (d / name).write_bytes(get(f"{base}/api/projects/{p['id']}/{path}"))
            except urllib.error.HTTPError as exc:
                notes.append(f"project {p['id']}: {path} failed ({exc.code})")
        if events:
            (d / "events.csv").write_bytes(get(f"{base}/api/projects/{p['id']}/events.csv"))
    notes.append(f"{len(projects)} project(s) collected from {base}")
    return notes


def copy_results(src: Path, dst: Path) -> list[Path]:
    copied = []
    if not src.is_dir():
        return copied
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix.lower() in TEXT and "runs" not in f.relative_to(src).parts[:-1]:
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-url", default="http://localhost:8000")
    parser.add_argument("--no-events", action="store_true", help="leave out the raw events.csv")
    parser.add_argument("--no-app", action="store_true", help="skip the app (only training/evaluation results)")
    args = parser.parse_args()

    name = f"metrics_bundle_{datetime.now():%Y-%m-%d_%H%M}"
    out = ROOT / name
    out.mkdir()
    env = environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    files = copy_results(ROOT / "training" / "results", out / "training")
    files += copy_results(ROOT / "evaluation" / "results", out / "evaluation")
    notes = ["App skipped (--no-app)"] if args.no_app else collect_app(args.app_url, out / "app", not args.no_events)

    # One readable file with everything (summaries first, raw tables after).
    lines = [f"# Metrics bundle {name}", "", "## Environment", "", "```json", json.dumps(env, indent=2), "```", "",
             "## Notes", "", *[f"- {n}" for n in notes], ""]
    all_files = sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in TEXT and p.name not in ("METRICS.md", "environment.json"))
    ordered = sorted(all_files, key=lambda p: (p.suffix != ".md", p.name == "events.csv", p.name == "per_image.csv", str(p)))
    for f in ordered:
        if f.name == "events.csv":
            rows = f.read_text(encoding="utf-8").count("\n") - 1
            lines += [f"## {f.relative_to(out)}", "", f"(raw log, {rows} events: see the file)", ""]
            continue
        text = f.read_text(encoding="utf-8")
        if len(text) > MAX_INLINE:
            text = text[:MAX_INLINE] + "\n... (cut, see the file)"
        fence = "" if f.suffix == ".md" else ("json" if f.suffix == ".json" else "csv")
        lines += [f"## {f.relative_to(out)}", ""]
        lines += [text.strip(), ""] if f.suffix == ".md" else [f"```{fence}", text.strip(), "```", ""]
    (out / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")

    archive = shutil.make_archive(str(out), "zip", out)
    print(f"training/evaluation files: {len(files)}")
    for n in notes:
        print(f"  {n}")
    print(f"wrote {out}/METRICS.md")
    print(f"wrote {archive}  ({Path(archive).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
