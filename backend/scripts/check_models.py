"""Smoke-test the exported ONNX models with the backend's own inference code.

Run from backend/ after placing weights in models/weights:
    python -m scripts.check_models path/to/image.jpg [--x 320 --y 240]

Loads SAM and YOLO exactly as the API does (same env vars: MODELS_DIR,
SAM_VARIANT, ...), runs one click and one detection, and prints latency.
"""

import argparse
import time
from pathlib import Path

from app import config
from app.inference import registry
from app.inference.sam import load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--x", type=float, help="click x in pixels (default: image centre)")
    parser.add_argument("--y", type=float, help="click y in pixels (default: image centre)")
    args = parser.parse_args()

    rgb = load_rgb(args.image)
    h, w = rgb.shape[:2]
    x, y = args.x if args.x is not None else w / 2, args.y if args.y is not None else h / 2
    print(f"image {args.image} ({w}x{h}); models dir {config.MODELS_DIR}")
    print(registry.status())

    try:
        sam = registry.get_sam()
        t0 = time.perf_counter()
        _, _, enc_ms, _ = sam.embed("check", args.image)
        polygon, score, dec_ms, _ = sam.segment("check", args.image, [(x, y, 1)], None)
        print(f"SAM ({config.SAM_VARIANT}): encoder {enc_ms:.0f} ms, decoder {dec_ms:.0f} ms, "
              f"score {score:.3f}, polygon vertices {len(polygon) // 2}, total {(time.perf_counter() - t0) * 1000:.0f} ms")
        if not polygon:
            print("  WARNING: empty mask; check SAM_VARIANT and the I/O contract in models/README.md")
    except registry.ModelUnavailable as exc:
        print(f"SAM unavailable: {exc}")

    try:
        yolo = registry.get_yolo()
        detections, ms = yolo.detect(rgb, conf=0.25, iou=0.45)
        print(f"YOLO: {ms:.0f} ms, {len(detections)} detections")
        for d in detections[:10]:
            b = d["box"]
            print(f"  {d['label']:15s} {d['score']:.2f}  x={b['x']:.0f} y={b['y']:.0f} w={b['w']:.0f} h={b['h']:.0f}")
    except registry.ModelUnavailable as exc:
        print(f"YOLO unavailable: {exc}")


if __name__ == "__main__":
    main()
