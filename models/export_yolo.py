"""Export an Ultralytics YOLO detector to ONNX for the backend.

Run on the Linux machine (downloads the .pt weights on first use):
    python models/export_yolo.py --weights yolo26n.pt
Writes models/weights/yolo.onnx (model input 640x640, class names in metadata).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo26n.pt", help="Ultralytics weights name or path (e.g. yolo11n.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=str(WEIGHTS_DIR / "yolo.onnx"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
