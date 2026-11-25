"""Turn a YOLO export from the app into a train/val/test dataset for training/train.py.

The app's "YOLO (.zip)" export has images/, labels/ and a data.yaml with train = val = all
images. This script makes a seeded, reproducible split so evaluation uses held-out images.
Images whose label file is empty are skipped by default: in the app they usually mean
"not annotated yet", and training on them as background would teach the model to miss objects.

    python training/prepare_app_export.py export.zip --out training/datasets/my_project
    python training/train.py --data training/datasets/my_project/data.yaml --variant full
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            p = PurePosixPath(member)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="YOLO .zip exported from the app (or an extracted folder)")
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with no boxes as background")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        parser.error(f"{out} already exists; pick a new --out or delete it first")
    src = Path(args.export)
    work = out / "_src"
    if src.is_dir():
        shutil.copytree(src, work)
    else:
        work.mkdir(parents=True)
        safe_extract(src, work)

    names = yaml.safe_load((work / "data.yaml").read_text())["names"]
    pairs, skipped = [], 0
    for img in sorted((work / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        label = work / "labels" / f"{img.stem}.txt"
        has_boxes = label.is_file() and label.read_text().strip() != ""
        if not has_boxes and not args.keep_empty:
            skipped += 1
            continue
        pairs.append((img, label))

    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    n_test, n_val = round(n * args.test), round(n * args.val)
    splits = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val], "train": pairs[n_test + n_val:]}
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit(f"only {n} annotated images: too few for train/val/test. Annotate more first.")

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img, label in items:
            shutil.copy2(img, out / "images" / split / img.name)
            target = out / "labels" / split / f"{img.stem}.txt"
            if label.is_file():
                shutil.copy2(label, target)
            else:
                target.write_text("")
    shutil.rmtree(work)

    data = {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    counts = ", ".join(f"{k} {len(v)}" for k, v in splits.items())
    print(f"wrote {out}/data.yaml: {counts}; skipped {skipped} images without boxes")
    if n < 100:
        print("warning: fewer than 100 images; expect noisy metrics and report that as a limitation")


if __name__ == "__main__":
    main()
