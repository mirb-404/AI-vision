#!/usr/bin/env bash
# Export SAM 2.1 (hiera tiny by default) to encoder + decoder ONNX with samexporter.
# Run on the Linux machine:  bash models/export_sam.sh [model_type]
# model_type: sam2.1_hiera_tiny (default) | sam2.1_hiera_small | sam2.1_hiera_base_plus | sam2.1_hiera_large
#
# samexporter's CLI can change between releases. If a flag fails, check
# `python -m samexporter.export_sam2 --help`, then run backend/scripts/check_models.py.
set -euo pipefail

MODEL_TYPE="${1:-sam2.1_hiera_tiny}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$HERE/checkpoints"
OUT_DIR="$HERE/weights"
mkdir -p "$CKPT_DIR" "$OUT_DIR"

CKPT="$CKPT_DIR/${MODEL_TYPE}.pt"
if [ ! -f "$CKPT" ]; then
  # Official Meta checkpoint location used by facebookresearch/sam2 download_ckpts.sh
  wget -O "$CKPT" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${MODEL_TYPE}.pt"
fi

python -m samexporter.export_sam2 \
  --checkpoint "$CKPT" \
  --output_encoder "$OUT_DIR/sam_encoder.onnx" \
  --output_decoder "$OUT_DIR/sam_decoder.onnx" \
  --model_type "$MODEL_TYPE"

echo "wrote $OUT_DIR/sam_encoder.onnx and $OUT_DIR/sam_decoder.onnx (set SAM_VARIANT=sam2)"
