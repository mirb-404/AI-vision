# models/

Model weights and the scripts that produce them. **Weights are never committed** (`*.onnx`, `*.pt` are git-ignored) and are **not downloaded on the dev machine**. Download and export them on the Linux cloud machine.

The inference code itself lives in `backend/app/inference/`. This folder only holds the files that code loads.

## Expected files

| File in `models/weights/` | Used by | Env var to rename |
|---|---|---|
| `sam_encoder.onnx` | SAM image encoder (once per image) | `SAM_ENCODER` |
| `sam_decoder.onnx` | SAM prompt decoder (per click/box) | `SAM_DECODER` |
| `yolo.onnx` | YOLO detector (box pre-labels) | `YOLO_MODEL` |
| `owlv2.onnx` | OWLv2: find every object of the same kind as a marked example | `OWL_MODEL` |
| `owlv2_text.onnx` | OWLv2 text tower: find every object matching a typed name | `OWL_TEXT_MODEL` |
| `owlv2_tokenizer.json` | CLIP tokenizer for the text tower | `OWL_TOKENIZER` |

`SAM_VARIANT=sam2` (default) for SAM 2.x exports; `SAM_VARIANT=sam1` for SAM v1 / MobileSAM exports.
The app runs without any of these: AI buttons show "unavailable" and manual labeling still works.

## Export on Linux

```bash
python -m venv .venv-export && source .venv-export/bin/activate
pip install -r models/requirements-export.txt

# YOLO detector (COCO classes): writes models/weights/yolo.onnx
python models/export_yolo.py --weights yolo26n.pt      # or yolo11n.pt / yolov8n.pt

# SAM 2.1 tiny: downloads the Meta checkpoint, writes sam_encoder.onnx + sam_decoder.onnx
bash models/export_sam.sh sam2.1_hiera_tiny

# OWLv2 base (find alike + find by name): downloads ~600 MB from Hugging Face, writes owlv2.onnx,
# owlv2_text.onnx and owlv2_tokenizer.json, then checks every ONNX output against PyTorch
python models/export_owlv2.py

# Check the I/O of the files, then test them with the backend code
python models/inspect_onnx.py models/weights/*.onnx
cd backend && python -m scripts.check_models /path/to/some/photo.jpg
```

## I/O contract (what `backend/app/inference` expects)

**SAM encoder:** one input.
- Rank 4 `[1, 3, 1024, 1024]` float: the backend resizes, normalises (ImageNet mean/std), and pads. For `sam2` it resizes to a 1024x1024 square. For `sam1` it resizes the longest side to 1024 and zero-pads.
- Rank 3 `[H, W, 3]`: the graph normalises the image itself. The backend passes a 0–255 image resized to 1024 on its longest side.

The encoder may have any number of outputs. Each one is passed to the decoder input that has the same name. The samexporter SAM 2 encoder, for example, outputs `image_embed`, `high_res_feats_0` and `high_res_feats_1`.

**SAM decoder:** inputs are the encoder outputs (matched by name) plus the standard SAM prompt inputs:

| Input | Value |
|---|---|
| `point_coords` | `[1, N, 2]` in model-input pixels |
| `point_labels` | `[1, N]`: 1 = foreground, 0 = background, 2/3 = box corners, -1 = padding |
| `mask_input` | `[1, 1, 256, 256]` zeros |
| `has_mask_input` | `[1]` = 0 |
| `orig_im_size` | `[2]` = (H, W). Optional: the samexporter SAM 2 decoder has no such input and returns 256x256 masks, which the backend resizes |

Data types are cast to whatever the graph declares. Outputs are `masks` (logits; > 0 counts as foreground) and `iou_predictions`. The mask with the best predicted IoU is kept. If its size is not (H, W), it is resized.

**YOLO:** an Ultralytics ONNX export with input `[1, 3, 640, 640]`. Two output layouts are supported:
- end-to-end `[1, N, 6]` (YOLO26 / v10)
- classic `[1, 4+C, anchors]` (YOLOv8 / 11). The backend runs NMS on this one.

Class names come from the ONNX metadata (`names`).

**OWLv2:** input `pixel_values [1, 3, 960, 960]` (padded to a square with grey, CLIP-normalised; see
`backend/app/inference/owl.py`). Per-patch outputs: `class_embeds [1, N, D]` (unit), `logit_shift`
and `logit_scale [1, N, 1]`, `pred_boxes [1, N, 4]` (cx, cy, w, h normalised to the square), and
`objectness [1, N]`. Picking the query from the example box and scoring the patches happen in numpy.

**OWLv2 text:** inputs `input_ids`, `attention_mask [1, 16]` int64 (CLIP tokens of "a photo of a <name>",
padded with id 0), output `text_embeds [1, D]` (normalised in numpy). The backend tokenises with the
`tokenizers` package and `owlv2_tokenizer.json`; the export script checks that the ids match `transformers`.

## Licenses / citations (for the report)

- SAM 2 / 2.1: Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714. Apache-2.0 (github.com/facebookresearch/sam2).
- MobileSAM (optional): Zhang et al., arXiv:2306.14289. Apache-2.0.
- YOLO26 / YOLO11 / YOLOv8: Ultralytics, AGPL-3.0 (github.com/ultralytics/ultralytics).
- OWLv2: Minderer, Gritsenko, Houlsby, *Scaling Open-Vocabulary Object Detection*, NeurIPS 2023. Checkpoint `google/owlv2-base-patch16-ensemble`, Apache-2.0. The image-guided (one-shot) query follows OWL-ViT (Minderer et al., ECCV 2022) and Hugging Face `transformers` (Apache-2.0). Text queries use the CLIP tokenizer via Hugging Face `tokenizers` (Apache-2.0).
- samexporter (ONNX export tool only): github.com/vietanhdev/samexporter. Check its license.

Verify each license against the repository's LICENSE file before submission.
