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
query input.

It also writes the text tower for "find by name" (type "car", the AI marks every car):
    owlv2_text.onnx          input_ids, attention_mask (1x16 int64) -> text_embeds (1, D), not normalised
    owlv2_tokenizer.json     CLIP tokenizer (Hugging Face `tokenizers` format)
The script checks every ONNX output against PyTorch before finishing.

Model: google/owlv2-base-patch16-ensemble, Apache-2.0 (Minderer et al., NeurIPS 2023).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Owlv2ForObjectDetection

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.inference.owl import INPUT_SIZE, TEXT_LEN, TEXT_PROMPT, preprocess  # noqa: E402  (same as serving)

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


class TextEmbeds(torch.nn.Module):
    """OWLv2 text tower: token ids -> projected text embedding (the query for a class name)."""

    def __init__(self, model: Owlv2ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        owl = self.model.owlv2
        pooled = owl.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]  # EOS token state
        return owl.text_projection(pooled)


def export_text(model: Owlv2ForObjectDetection, name: str, out: Path, tok_out: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    tokenizer.backend_tokenizer.save(str(tok_out))
    print(f"wrote {tok_out}")
    enc = tokenizer([TEXT_PROMPT.format("cricket ball")], padding="max_length", max_length=TEXT_LEN,
                    truncation=True, return_tensors="pt")
    ids, mask = enc["input_ids"].long(), enc["attention_mask"].long()

    # The backend tokenises with `tokenizers` directly: check it gives the same ids as transformers.
    from tokenizers import Tokenizer
    fast = Tokenizer.from_file(str(tok_out))
    fast.enable_truncation(TEXT_LEN)
    fast.enable_padding(length=TEXT_LEN, pad_id=0, pad_token="!")
    got_ids = fast.encode(TEXT_PROMPT.format("cricket ball")).ids
    if got_ids != ids[0].tolist():
        raise SystemExit(f"tokenizer mismatch: {got_ids} vs {ids[0].tolist()}")

    wrapper = TextEmbeds(model).eval()
    with torch.no_grad():
        reference = wrapper(ids, mask).numpy()
        torch.onnx.export(wrapper, (ids, mask), str(out), input_names=["input_ids", "attention_mask"],
                          output_names=["text_embeds"], opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(["text_embeds"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
    diff = float(np.abs(reference - got).max())
    print(f"  text_embeds   {tuple(got.shape)}  max |onnx - torch| = {diff:.2e}")
    if diff > 1e-2:
        raise SystemExit(f"ONNX text output differs from PyTorch by {diff}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--out", default=str(ROOT / "weights" / "owlv2.onnx"))
    parser.add_argument("--text-out", default=str(ROOT / "weights" / "owlv2_text.onnx"))
    parser.add_argument("--tokenizer-out", default=str(ROOT / "weights" / "owlv2_tokenizer.json"))
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
    export_text(model, args.model, Path(args.text_out), Path(args.tokenizer_out))
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
