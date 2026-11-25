"""Train one YOLO detector variant for the transfer-learning experiment.

Variants (same data, epochs, image size and seed, so only the starting point differs):
  scratch  random weights from the model YAML (no transfer learning)
  head     COCO-pretrained weights, first 10 layers (backbone) frozen: feature extraction
  full     COCO-pretrained weights, every layer trained: full fine-tuning

The detection head is re-initialised automatically when the dataset has a different
number of classes than COCO. Use --fraction to train on only part of the training split
(learning curve: how many labelled images are needed).

Run on the Linux GPU machine:
    python training/train.py --data african-wildlife.yaml --variant full
Writes training/runs/<tag>/<name>/ (weights/best.pt, results.csv, plots, train_config.json).
"""

import argparse
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
VARIANTS = ("scratch", "head", "full")
BACKBONE_LAYERS = 10


def run_name(variant: str, fraction: float, seed: int) -> str:
    name = variant
    if fraction < 1.0:
        name += f"_f{round(fraction * 100):03d}"
    return f"{name}_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML (name like african-wildlife.yaml, or a path)")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--model", default="yolo26n", help="Model family/scale without extension, e.g. yolo26n or yolo11n")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="Share of the training split to use (0-1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    parser.add_argument("--tag", default=None, help="Run group folder; default: dataset name")
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")

    tag = args.tag or Path(args.data).stem
    project = HERE / "runs" / tag
    name = run_name(args.variant, args.fraction, args.seed)

    scratch = args.variant == "scratch"
    model = YOLO(f"{args.model}.yaml" if scratch else f"{args.model}.pt")
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        seed=args.seed,
        deterministic=True,
        pretrained=not scratch,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
    )
    if args.variant == "head":
        train_kwargs["freeze"] = BACKBONE_LAYERS
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    best = Path(model.trainer.best)
    run_dir = project / name
    config = {
        **vars(args),
        "tag": tag,
        "name": name,
        "init": f"{args.model}.yaml (random)" if scratch else f"{args.model}.pt (COCO pretrained)",
        "frozen_layers": BACKBONE_LAYERS if args.variant == "head" else 0,
        "best_weights": str(best),
        "ultralytics_version": ultralytics.__version__,
    }
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
