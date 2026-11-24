"""Load a tiny demo dataset (Ultralytics coco8: 8 COCO images) into a running server.

    python -m scripts.load_demo http://127.0.0.1:8000

Downloads coco8.zip once into DATA_DIR/demo (git-ignored), then creates the
project "Demo: coco8" with one label per COCO class that appears in the images
and uploads the images. Does nothing when that project already exists.

The ground-truth YOLO labels stay in DATA_DIR/demo/coco8/labels for evaluation;
they are not imported, so the images start unlabelled.

Source: https://github.com/ultralytics/assets/releases (coco8.zip), a subset of
COCO val2017 (https://cocodataset.org). COCO annotations are CC BY 4.0; the
images are from Flickr under their own Creative Commons licenses (see the COCO
terms of use).
"""

import argparse
import colorsys
import sys
import urllib.request
import zipfile
from pathlib import Path

import httpx

from app import config

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
PROJECT = "Demo: coco8"

# COCO 80-class names in YOLO class-id order.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def fetch(dest: Path) -> Path:
    root = dest / "coco8"
    if (root / "images").is_dir():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "coco8.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():  # refuse paths that would escape dest
            if name.startswith("/") or ".." in Path(name).parts:
                sys.exit(f"unsafe path in zip: {name}")
        zf.extractall(dest)
    zpath.unlink()
    return root


def color(i: int, n: int) -> str:
    r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url")
    args = parser.parse_args()

    api = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=60)
    if any(p["name"] == PROJECT for p in api.get("/api/projects").raise_for_status().json()):
        print(f"Demo project '{PROJECT}' already loaded")
        return

    root = fetch(config.DATA_DIR / "demo")
    images = sorted((root / "images").rglob("*.jpg"))
    class_ids = sorted({int(line.split()[0])
                        for txt in (root / "labels").rglob("*.txt")
                        for line in txt.read_text().splitlines() if line.strip()})

    project = api.post("/api/projects", json={"name": PROJECT}).raise_for_status().json()
    for i, cid in enumerate(class_ids):
        api.post(f"/api/projects/{project['id']}/labels",
                 json={"name": COCO_NAMES[cid], "color": color(i, len(class_ids))}).raise_for_status()

    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in images]
    result = api.post(f"/api/projects/{project['id']}/images", files=files).raise_for_status().json()
    if result["errors"]:
        print(f"upload errors: {result['errors']}")
    print(f"Loaded '{PROJECT}': {len(result['uploaded'])} images, {len(class_ids)} labels "
          f"({', '.join(COCO_NAMES[c] for c in class_ids)})")


if __name__ == "__main__":
    main()
