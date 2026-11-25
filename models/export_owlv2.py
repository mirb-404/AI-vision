"""Export OWLv2 (image-guided, one-shot detection) to ONNX for the backend's "find alike objects".

Run on the Linux machine (downloads ~600 MB from the Hugging Face hub on first use):
    python models/export_owlv2.py
Writes models/weights/owlv2.onnx with one input (pixel_values, 1x3x960x960) and per-patch outputs:
    class_embeds (1, N, D)  unit class embeddings (the space queries live in)
    logit_shift  (1, N, 1)  class-head shift
    logit_scale  (1, N, 1)  class-head scale (elu + 1 applied)
    pred_boxes   (1, N, 4)  cx, cy, w, h normalised to the padded square input
    objectness   (1, N)     objectness logits
Query selection and scoring run in numpy (backend/app/inference/owl.py), so the graph needs no
query input. The script checks the ONNX outputs against PyTorch before finishing.

Model: google/owlv2-base-patch16-ensemble, Apache-2.0 (Minderer et al., NeurIPS 2023).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Owlv2ForObjectDetection

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.inference.owl import INPUT_SIZE, preprocess  # noqa: E402  (same preprocessing as serving)

OUTPUTS = ["class_embeds", "logit_shift", "logit_scale", "pred_boxes", "objectness"]


class PatchOutputs(torch.nn.Module):
    """Owlv2ForObjectDetection without text/query inputs: everything per image patch."""

    def __init__(self, model: Owlv2ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        m = self.model
        feature_map = m.image_embedder(pixel_values=pixel_values)[0]  # (B, h, w, D)
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        head = m.class_head
        embeds = head.dense0(feats)
        embeds = embeds / (torch.linalg.norm(embeds, dim=-1, keepdim=True) + 1e-6)
        shift = head.logit_shift(feats)
        scale = head.elu(head.logit_scale(feats)) + 1
        boxes = m.box_predictor(feats, feature_map)
        objectness = m.objectness_predictor(feats)
        return embeds, shift, scale, boxes, objectness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--out", default=str(ROOT / "weights" / "owlv2.onnx"))
    parser.add_argument("--check-image", default=None, help="Image for the ONNX-vs-PyTorch check (default: random)")
    args = parser.parse_args()

    model = Owlv2ForObjectDetection.from_pretrained(args.model).eval()
    wrapper = PatchOutputs(model).eval()
    if args.check_image:
        from PIL import Image
        rgb = np.asarray(Image.open(args.check_image).convert("RGB"))
    else:
        rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    pixels, _ = preprocess(rgb)
    assert pixels.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = [t.numpy() for t in wrapper(torch.from_numpy(pixels))]
        torch.onnx.export(wrapper, (torch.from_numpy(pixels),), str(out), input_names=["pixel_values"],
                          output_names=OUTPUTS, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(OUTPUTS, {"pixel_values": pixels})
    for name, a, b in zip(OUTPUTS, reference, got):
        diff = float(np.abs(a - b).max())
        print(f"  {name:13s} {tuple(b.shape)}  max |onnx - torch| = {diff:.2e}")
        if diff > 1e-2:
            raise SystemExit(f"ONNX output {name} differs from PyTorch by {diff}")
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
