"""Annotation export to standard formats (COCO JSON, YOLO txt), zipped with images."""

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ..tables import Annotation, Image, Label


def export_name(image: Image) -> str:
    """Unique, filesystem-safe file name for an image inside the export."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(image.filename).stem)[:80] or "image"
    return f"{image.id:06d}_{stem}{Path(image.stored_name).suffix}"


def _clip_box(a: Annotation, image: Image) -> tuple[float, float, float, float] | None:
    """Axis-aligned box (x, y, w, h) of an annotation, clipped to the image."""
    g = a.geometry
    if a.kind == "box":
        x1, y1, x2, y2 = g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]
    else:
        xs, ys = g["points"][0::2], g["points"][1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, x2 = max(0.0, x1), min(float(image.width), x2)
    y1, y2 = max(0.0, y1), min(float(image.height), y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _polygon(a: Annotation, image: Image) -> list[float]:
    if a.kind == "box":
        x, y, w, h = a.geometry["x"], a.geometry["y"], a.geometry["w"], a.geometry["h"]
        pts = [x, y, x + w, y, x + w, y + h, x, y + h]
    else:
        pts = a.geometry["points"]
    out = []
    for i, v in enumerate(pts):
        limit = image.width if i % 2 == 0 else image.height
        out.append(round(min(max(float(v), 0.0), float(limit)), 2))
    return out


def polygon_area(points: list[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


def coco_dict(project_name: str, images: list[Image], labels: list[Label],
              annotations: dict[int, list[Annotation]]) -> dict:
    coco_images, coco_anns = [], []
    ann_id = 1
    for image in images:
        coco_images.append({"id": image.id, "file_name": f"images/{export_name(image)}",
                            "width": image.width, "height": image.height})
        for a in annotations.get(image.id, []):  # type: ignore[arg-type]
            box = _clip_box(a, image)
            if box is None:
                continue
            poly = _polygon(a, image)
            coco_anns.append({
                "id": ann_id,
                "image_id": image.id,
                "category_id": a.label_id,
                "bbox": [round(v, 2) for v in box],
                "segmentation": [poly],
                "area": round(polygon_area(poly) if a.kind == "polygon" else box[2] * box[3], 2),
                "iscrowd": 0,
                # Non-standard but harmless extras: provenance for quality analysis.
                "attributes": {"type": a.kind, "source": a.source, "score": a.score, "uid": a.id},
            })
            ann_id += 1
    return {
        "info": {"description": f"{project_name} - exported by AI Annotator",
                 "date_created": datetime.now(timezone.utc).isoformat()},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": l.id, "name": l.name, "supercategory": "none"} for l in labels],
    }


def write_coco_zip(zf: zipfile.ZipFile, project_name: str, images: list[Image], labels: list[Label],
                   annotations: dict[int, list[Annotation]], images_dir: Path) -> None:
    zf.writestr("annotations.json", json.dumps(coco_dict(project_name, images, labels, annotations), indent=2))
    for image in images:
        zf.write(images_dir / image.stored_name, f"images/{export_name(image)}")


def write_yolo_zip(zf: zipfile.ZipFile, images: list[Image], labels: list[Label],
                   annotations: dict[int, list[Annotation]], images_dir: Path) -> None:
    """Ultralytics detection layout: images/, labels/<stem>.txt (class cx cy w h, normalised), data.yaml."""
    index = {l.id: i for i, l in enumerate(labels)}
    for image in images:
        name = export_name(image)
        lines = []
        for a in annotations.get(image.id, []):  # type: ignore[arg-type]
            box = _clip_box(a, image)
            if box is None:
                continue
            x, y, w, h = box
            lines.append(f"{index[a.label_id]} {(x + w / 2) / image.width:.6f} {(y + h / 2) / image.height:.6f} "
                         f"{w / image.width:.6f} {h / image.height:.6f}")
        zf.write(images_dir / image.stored_name, f"images/{name}")
        zf.writestr(f"labels/{Path(name).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: {json.dumps(l.name)}" for i, l in enumerate(labels))
    zf.writestr("data.yaml", f"path: .\ntrain: images\nval: images\nnames:\n{names}\n")
    zf.writestr("classes.txt", "\n".join(l.name for l in labels) + "\n")
