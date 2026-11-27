"""Magic select (SAM) quality: NoC@85 / NoC@90 with a simulated user, the standard interactive-
segmentation benchmark (RITM, Sofiiuk et al. 2021, arXiv:2102.06583; also used in the SAM paper).

The simulated user clicks like this:
  1. First click: the centre of the object (the point farthest from the object border).
  2. Every next click: the centre of the largest error region. Missed object pixels -> object click,
     wrong background pixels -> background click (whichever region reaches farther from its border).
  3. Stop at 20 clicks. NoC@85 = clicks needed to reach IoU 0.85 (20 if never reached), same for 0.90.

The model is the app's own SAM wrapper (backend/app/inference/sam.py), so the numbers measure what
users get. All clicks so far are sent each time; the previous mask is not fed back (the app does not
either), so results can be a little worse than published SAM numbers that use mask feedback.

Run on the Linux machine (needs models/weights/sam_encoder.onnx + sam_decoder.onnx):
    python evaluation/noc_benchmark.py --preset grabcut --root datasets/GrabCut
    python evaluation/noc_benchmark.py --preset berkeley --root datasets/Berkeley
    python evaluation/noc_benchmark.py --images my/imgs --masks my/masks   # any image + mask folders

Writes evaluation/results/<name>/: per_image.csv, curve.csv (mean IoU per click), summary.md,
metrics.json, and curve.png when matplotlib is installed.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.inference.sam import SamModel  # noqa: E402  (the app's own SAM code)

MAX_CLICKS = 20
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Folder layout of the RITM / f-BRS benchmark downloads (see evaluation/README.md).
PRESETS = {
    "grabcut": {"images": "data_GT", "masks": "boundary_GT", "ignore": 128},
    "berkeley": {"images": "images", "masks": "masks", "ignore": None},
}


def iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p, g = pred & valid, gt & valid
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0


def _farthest(region: np.ndarray) -> tuple[float, int, int]:
    """(distance to the region border, x, y) of the region's most interior pixel."""
    if not region.any():
        return 0.0, 0, 0
    padded = np.pad(region.astype(np.uint8), 1)  # the image edge counts as a border
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return float(dist[y, x]), int(x), int(y)


def next_click(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    """RITM click: centre of the larger error region. Returns (x, y, label 1=object / 0=background)."""
    fn = gt & ~pred & valid  # object pixels the mask misses
    fp = pred & ~gt & valid  # background pixels the mask wrongly covers
    d_fn, x_fn, y_fn = _farthest(fn)
    d_fp, x_fp, y_fp = _farthest(fp)
    if d_fn >= d_fp:
        return float(x_fn), float(y_fn), 1
    return float(x_fp), float(y_fp), 0


def run_image(sam: SamModel, key: str, image_path: Path, gt: np.ndarray, valid: np.ndarray) -> tuple[list[float], float]:
    """IoU after each of 1..MAX_CLICKS simulated clicks. Returns (ious, mean decoder ms)."""
    feats, (h, w), _, _ = sam.embed(key, image_path)
    if (h, w) != gt.shape:
        raise ValueError(f"mask size {gt.shape} differs from image size {(h, w)}")
    pred = np.zeros_like(gt)
    points: list[tuple[float, float, int]] = []
    ious, times = [], []
    for _ in range(MAX_CLICKS):
        points.append(next_click(pred, gt, valid))
        pred, _, ms = sam._decode(feats, h, w, points, None)
        times.append(ms)
        ious.append(iou(pred, gt, valid))
    return ious, float(np.mean(times))


def noc(ious: list[float], thr: float) -> int:
    """Clicks needed to reach IoU >= thr (MAX_CLICKS when never reached, as in RITM)."""
    return next((i + 1 for i, v in enumerate(ious) if v >= thr), MAX_CLICKS)


def manual_vertices(gt: np.ndarray) -> int:
    """Clicks a manual polygon would need, approximated by the vertex count of the object's outline
    simplified to 1 % of its perimeter (Polygon-RNN-style click counting; an approximation)."""
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total = 0
    for c in contours:
        if cv2.contourArea(c) < 10:
            continue
        total += len(cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True))
    return total


def pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXT}
    out = [(p, masks[p.stem]) for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXT and p.stem in masks]
    if not out:
        raise SystemExit(f"no image/mask pairs with the same file name in {images_dir} and {masks_dir}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="folder layout of a benchmark download")
    parser.add_argument("--root", type=Path, help="dataset folder (with --preset)")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--masks", type=Path, help="masks with the image's file name; non-zero = object")
    parser.add_argument("--ignore-value", type=int, default=None, help="mask value to ignore (GrabCut: 128)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images (quick check)")
    parser.add_argument("--name", default=None, help="results folder name")
    weights = ROOT.parent / "models" / "weights"
    parser.add_argument("--encoder", type=Path, default=weights / "sam_encoder.onnx")
    parser.add_argument("--decoder", type=Path, default=weights / "sam_decoder.onnx")
    parser.add_argument("--variant", default="sam2", choices=["sam2", "sam1"])
    args = parser.parse_args()

    if args.preset:
        if not args.root:
            parser.error("--preset needs --root")
        cfg = PRESETS[args.preset]
        images_dir, masks_dir = args.root / cfg["images"], args.root / cfg["masks"]
        ignore = cfg["ignore"] if args.ignore_value is None else args.ignore_value
    elif args.images and args.masks:
        images_dir, masks_dir, ignore = args.images, args.masks, args.ignore_value
    else:
        parser.error("give --preset and --root, or --images and --masks")
    name = args.name or args.preset or images_dir.name
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sam = SamModel(args.encoder, args.decoder, args.variant, cache_size=1)
    items = pairs(images_dir, masks_dir)
    if args.limit:
        items = items[: args.limit]

    rows, curves, encoder_ms, decoder_ms = [], [], [], []
    for k, (img_path, mask_path) in enumerate(items, 1):
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        valid = np.ones(raw.shape, bool) if ignore is None else raw != ignore
        gt = (raw > 0) & valid
        t0 = time.perf_counter()
        sam.embed(("enc", img_path.name), img_path)
        encoder_ms.append((time.perf_counter() - t0) * 1000)
        ious, dec_ms = run_image(sam, ("enc", img_path.name), img_path, gt, valid)
        decoder_ms.append(dec_ms)
        curves.append(ious)
        row = {"image": img_path.name, "noc85": noc(ious, 0.85), "noc90": noc(ious, 0.90),
               "iou@1": round(ious[0], 4), "iou@3": round(ious[2], 4), "iou@5": round(ious[4], 4),
               "iou@20": round(ious[-1], 4), "manual_vertices": manual_vertices(gt)}
        rows.append(row)
        print(f"[{k}/{len(items)}] {img_path.name}: NoC85 {row['noc85']}  NoC90 {row['noc90']}  IoU@1 {row['iou@1']:.3f}")

    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve = np.mean(np.asarray(curves), axis=0)
    with open(out_dir / "curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clicks", "mean_iou"])
        writer.writerows([[i + 1, round(float(v), 4)] for i, v in enumerate(curve)])

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    metrics = {
        "dataset": name, "images": len(rows), "max_clicks": MAX_CLICKS,
        "noc85": mean("noc85"), "noc90": mean("noc90"),
        "fail85": sum(r["noc85"] == MAX_CLICKS and r["iou@20"] < 0.85 for r in rows),
        "fail90": sum(r["noc90"] == MAX_CLICKS and r["iou@20"] < 0.90 for r in rows),
        "miou@1": mean("iou@1"), "miou@3": mean("iou@3"), "miou@5": mean("iou@5"),
        "manual_vertices": mean("manual_vertices"),
        "encoder_ms_median": float(np.median(encoder_ms)), "decoder_ms_median": float(np.median(decoder_ms)),
        "sam_variant": args.variant, "encoder": args.encoder.name, "decoder": args.decoder.name,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = f"""# Magic select (SAM) click benchmark: {name}

{len(rows)} images, simulated user (RITM protocol), at most {MAX_CLICKS} clicks.

| Metric | Value |
|---|---|
| NoC@85 (clicks to IoU 0.85, lower is better) | {metrics['noc85']:.2f} |
| NoC@90 | {metrics['noc90']:.2f} |
| Images never reaching 0.85 / 0.90 in {MAX_CLICKS} clicks | {metrics['fail85']} / {metrics['fail90']} |
| mIoU after 1 / 3 / 5 clicks | {metrics['miou@1']:.3f} / {metrics['miou@3']:.3f} / {metrics['miou@5']:.3f} |
| Manual polygon, approx. clicks (outline vertices) | {metrics['manual_vertices']:.1f} |
| Encoder / decoder latency (median) | {metrics['encoder_ms_median']:.0f} ms / {metrics['decoder_ms_median']:.1f} ms |

Clicks saved vs a manual polygon at IoU 0.90 ≈ {metrics['manual_vertices']:.1f} / {metrics['noc90']:.2f}
= {metrics['manual_vertices'] / max(metrics['noc90'], 1e-9):.1f}x (approximation: a manual vertex and a
SAM click are counted the same, which ignores that placing vertices accurately takes longer).
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(range(1, MAX_CLICKS + 1), curve, marker="o", ms=3)
        ax.axhline(0.85, ls="--", lw=0.8, c="grey")
        ax.axhline(0.90, ls=":", lw=0.8, c="grey")
        ax.set(xlabel="clicks", ylabel="mean IoU", title=f"Magic select: {name}", ylim=(0, 1), xticks=range(1, 21, 2))
        fig.tight_layout()
        fig.savefig(out_dir / "curve.png", dpi=150)
    except ImportError:
        pass
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
