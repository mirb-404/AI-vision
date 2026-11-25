"""Evaluate YOLO detectors on one held-out split with the same code for every model.

Metrics (pycocotools, the standard COCO implementation):
  mAP@0.5:0.95, mAP@0.5, AR@100, and AP per class.
At the app's operating threshold (--conf, default 0.25 = auto-label default), IoU 0.5:
  precision, recall, F1, and an annotation-effort estimate:
    manual   = draw every ground-truth box               -> GT actions
    assisted = delete each false positive + draw each miss -> FP + FN actions
    saved    = 1 - (FP + FN) / GT   (box-adjust edits on true positives are not counted)

A COCO-pretrained model is evaluated zero-shot by mapping its class names to the dataset's
class names (case-insensitive); dataset classes it does not know score 0.

Run on the Linux GPU machine:
    python training/evaluate.py --data african-wildlife.yaml --split test \
        --model zero-shot=yolo26n.pt --runs training/runs/african-wildlife
Writes to --out: summary.md, summary.csv, per_class.csv, metrics.json, failures/<model>/*.jpg
"""

import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, exif_size, img2label_paths

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- dataset

def list_images(entry: str | list[str]) -> list[Path]:
    """Image files of a split entry from a resolved dataset dict (dir, .txt list, or list of those)."""
    files: list[Path] = []
    for item in entry if isinstance(entry, list) else [entry]:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix[1:].lower() in IMG_FORMATS)
        elif p.suffix == ".txt" and p.is_file():
            for line in p.read_text().split():
                q = Path(line)
                files.append(q if q.is_absolute() else (p.parent / q).resolve())
        else:
            raise FileNotFoundError(f"split entry not found: {p}")
    return files


def read_label_file(path: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    """YOLO label file -> [(class, [x, y, w, h] in pixels)]. Accepts boxes or polygons."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            box = [x1, y1, bw * w, bh * h]
        else:  # polygon: x1 y1 x2 y2 ...
            xs, ys = np.array(vals[0::2]) * w, np.array(vals[1::2]) * h
            box = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
        out.append((cls, [float(v) for v in box]))
    return out


def build_gt(images: list[Path], names: dict[int, str]) -> dict:
    coco = {"info": {}, "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in names.items()]}
    label_files = img2label_paths([str(f) for f in images])
    ann_id = 1
    for img_id, (f, lf) in enumerate(zip(images, label_files), start=1):
        with Image.open(f) as im:
            w, h = exif_size(im)
        coco["images"].append({"id": img_id, "file_name": str(f), "width": w, "height": h})
        for cls, box in read_label_file(Path(lf), w, h):
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cls, "bbox": box,
                                        "area": box[2] * box[3], "iscrowd": 0})
            ann_id += 1
    return coco


# ---------------------------------------------------------------- prediction

def predict(weights: str, images: list[Path], names: dict[int, str], imgsz: int, device, batch: int):
    model = YOLO(weights)
    by_name = {n.strip().lower(): i for i, n in names.items()}
    lut = {mid: by_name[mn.strip().lower()] for mid, mn in model.names.items() if mn.strip().lower() in by_name}
    dets, ms = [], []
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        kwargs = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)
        if device is not None:
            kwargs["device"] = device
        for k, r in enumerate(model.predict(chunk, **kwargs)):
            ms.append(r.speed.get("inference", float("nan")))
            b = r.boxes
            for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                if int(c) in lut:
                    dets.append({"image_id": start + k + 1, "category_id": lut[int(c)],
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    mapped = sorted({names[d] for d in lut.values()})
    return dets, mapped, float(np.nanmean(ms)) if ms else float("nan")


# ---------------------------------------------------------------- metrics

def coco_metrics(gt: COCO, dets: list[dict]) -> tuple[dict, dict]:
    cat_ids = sorted(gt.getCatIds())
    zero = {"mAP50_95": 0.0, "mAP50": 0.0, "AR100": 0.0}
    if not dets:
        return zero, {c: {"AP50": 0.0, "AP50_95": 0.0} for c in cat_ids}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    overall = {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "AR100": float(s[8])}
    prec = ev.eval["precision"]  # [IoU thr, recall, class, area, maxDets]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p_all, p_50 = prec[:, :, k, 0, -1], prec[0, :, k, 0, -1]
        per_class[cid] = {"AP50": float(p_50[p_50 > -1].mean()) if (p_50 > -1).any() else float("nan"),
                          "AP50_95": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else float("nan")}
    return overall, per_class


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4], b: [M,4] as x, y, w, h -> IoU [N,M]."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def operating_point(coco_gt: dict, dets: list[dict], conf: float, iou_thr: float = 0.5):
    """Greedy per-image, per-class matching at a fixed confidence. Returns totals and per-image counts."""
    gts: dict[int, list] = {}
    for a in coco_gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    preds: dict[int, list] = {}
    for d in dets:
        if d["score"] >= conf:
            preds.setdefault(d["image_id"], []).append(d)
    per_image = {}
    for img in coco_gt["images"]:
        i = img["id"]
        g, p = gts.get(i, []), sorted(preds.get(i, []), key=lambda d: -d["score"])
        used, tp = set(), 0
        if g and p:
            ious = iou_matrix(np.array([d["bbox"] for d in p]), np.array([a["bbox"] for a in g]))
            for pi, d in enumerate(p):
                best, best_j = iou_thr, -1
                for j, a in enumerate(g):
                    if j not in used and a["category_id"] == d["category_id"] and ious[pi, j] >= best:
                        best, best_j = ious[pi, j], j
                if best_j >= 0:
                    used.add(best_j)
                    tp += 1
        per_image[i] = {"tp": tp, "fp": len(p) - tp, "fn": len(g) - tp, "gt": len(g)}
    tp, fp, fn, n_gt = (sum(v[k] for v in per_image.values()) for k in ("tp", "fp", "fn", "gt"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    totals = {"precision": precision, "recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "GT": n_gt,
              "manual_actions": n_gt, "assisted_actions": fp + fn,
              "actions_saved": 1 - (fp + fn) / n_gt if n_gt else 0.0}
    return totals, per_image


# ---------------------------------------------------------------- failure cases

def draw_failures(coco_gt: dict, dets: list[dict], per_image: dict, names: dict, conf: float, out_dir: Path, k: int):
    """Save the k images with the most errors (FP + FN). Green = ground truth, red = prediction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(per_image, key=lambda i: -(per_image[i]["fp"] + per_image[i]["fn"]))[:k]
    files = {im["id"]: im["file_name"] for im in coco_gt["images"]}
    for rank, i in enumerate(worst, start=1):
        c = per_image[i]
        if c["fp"] + c["fn"] == 0:
            break
        img = cv2.imread(files[i])
        if img is None:
            continue
        for a in (a for a in coco_gt["annotations"] if a["image_id"] == i):
            x, y, w, h = map(int, a["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        for d in (d for d in dets if d["image_id"] == i and d["score"] >= conf):
            x, y, w, h = map(int, d["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 230), 2)
            cv2.putText(img, f"{names[d['category_id']]} {d['score']:.2f}", (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1, cv2.LINE_AA)
        cv2.putText(img, f"FP {c['fp']}  FN {c['fn']}  GT {c['gt']}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{rank:02d}_{Path(files[i]).stem}.jpg"), img)


# ---------------------------------------------------------------- main

def discover(runs: Path) -> list[tuple[str, str]]:
    return [(p.parent.parent.name, str(p)) for p in sorted(runs.glob("*/weights/best.pt"))]


def fmt(v: float) -> str:
    return "n/a" if isinstance(v, float) and math.isnan(v) else f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Dataset YAML (same one used for training)")
    parser.add_argument("--split", default="test", help="test (default) or val; never evaluate on train")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=WEIGHTS",
                        help="Model to evaluate; repeatable. e.g. zero-shot=yolo26n.pt")
    parser.add_argument("--runs", default=None, help="Also evaluate every <runs>/*/weights/best.pt")
    parser.add_argument("--out", default=None, help="Output folder; default training/results/<dataset>_<split>")
    parser.add_argument("--conf", type=float, default=0.25, help="Operating threshold for P/R/effort (app default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--failures", type=int, default=8, help="Worst images to render per model")
    args = parser.parse_args()

    models = [tuple(m.split("=", 1)) if "=" in m else (Path(m).stem, m) for m in args.model]
    if args.runs:
        models += discover(Path(args.runs))
    if not models:
        parser.error("no models: pass --model and/or --runs")

    info = check_det_dataset(args.data)
    split = args.split
    if not info.get(split):
        print(f"warning: dataset has no '{split}' split, using 'val'")
        split = "val"
    names = {int(k): str(v) for k, v in info["names"].items()}
    images = list_images(info[split])
    out = Path(args.out) if args.out else HERE / "results" / f"{Path(args.data).stem}_{split}"
    out.mkdir(parents=True, exist_ok=True)

    gt_dict = build_gt(images, names)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
    print(f"{args.data} [{split}]: {len(images)} images, {len(gt_dict['annotations'])} boxes, {len(names)} classes")

    results, class_rows = [], []
    for name, weights in models:
        print(f"evaluating {name} ({weights})")
        dets, mapped, ms = predict(weights, images, names, args.imgsz, args.device, args.batch)
        overall, per_class = coco_metrics(gt, dets)
        op, per_image = operating_point(gt_dict, dets, args.conf)
        results.append({"model": name, "weights": weights, **overall, **op, "ms_per_image": ms,
                        "classes_known": f"{len(mapped)}/{len(names)}"})
        counts = {c: 0 for c in names}
        for a in gt_dict["annotations"]:
            counts[a["category_id"]] += 1
        for cid, v in per_class.items():
            class_rows.append({"model": name, "class": names[cid], "n_gt": counts[cid], **v})
        draw_failures(gt_dict, dets, per_image, names, args.conf, out / "failures" / name, args.failures)

    cols = ["model", "mAP50_95", "mAP50", "AR100", "precision", "recall", "F1", "TP", "FP", "FN", "GT",
            "actions_saved", "ms_per_image", "classes_known", "weights"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "class", "n_gt", "AP50", "AP50_95"])
        w.writeheader()
        w.writerows(class_rows)
    meta = {"data": args.data, "split": split, "images": len(images), "boxes": len(gt_dict["annotations"]),
            "classes": names, "conf": args.conf, "imgsz": args.imgsz}
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "models": results, "per_class": class_rows}, indent=2))

    head = ["model", "mAP50-95", "mAP50", f"P@{args.conf}", f"R@{args.conf}", "F1", "FP", "FN", "GT",
            "actions saved", "ms/img", "classes known"]
    lines = [f"# Detection results: {args.data} [{split}]", "",
             f"{len(images)} images, {len(gt_dict['annotations'])} boxes. "
             f"P/R/effort at conf {args.conf}, IoU 0.5. "
             "Actions saved = 1 - (FP + FN) / GT (estimate; box adjustments not counted).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        lines.append("| " + " | ".join([r["model"], fmt(r["mAP50_95"]), fmt(r["mAP50"]), fmt(r["precision"]),
                                        fmt(r["recall"]), fmt(r["F1"]), str(r["FP"]), str(r["FN"]), str(r["GT"]),
                                        f"{r['actions_saved']:.1%}", f"{r['ms_per_image']:.1f}",
                                        r["classes_known"]]) + " |")
    lines += ["", "## AP@0.5 per class", "", "| model | " + " | ".join(names.values()) + " |",
              "|---|" + "---|" * len(names)]
    for name, _ in models:
        row = {r["class"]: r["AP50"] for r in class_rows if r["model"] == name}
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(n, float("nan"))) for n in names.values()) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
