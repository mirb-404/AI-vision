# Training and fine-tuning

This folder holds the transfer-learning experiment for the YOLO pre-label model. It is kept apart from the app so the results can be checked and reproduced on their own. Everything here runs on the **Linux GPU machine (Datalab)**. Nothing is trained or downloaded on the Windows dev machine.

## Approach

**What is fine-tuned:** the YOLO detector that makes box pre-labels in the app. It is not SAM. SAM is class-agnostic and already reaches an IoU of about 0.97 from one click, so the gain from fine-tuning it is small, and it would need masks plus far more GPU time. The detector is different: a COCO-trained YOLO only knows the 80 COCO classes, so on a new domain it misses objects or gives them the wrong class. Fine-tuning fixes exactly that, and every extra correct pre-label is one less box to draw by hand.

**Dataset (default): Ultralytics African Wildlife.** It has 4 classes (buffalo, elephant, rhino, zebra) and about 1,500 images with fixed train/val/test splits. COCO contains elephant and zebra but **not** buffalo or rhino, so the zero-shot COCO model scores partly but cannot be complete. That makes the effect of transfer learning visible and easy to explain. It downloads automatically (about 100 MB).

**Experiment.** Every run uses the same data, epochs, image size and seed. Only the starting point changes (lecture 08, Transfer Learning):

| Run | Start weights | Trained layers | Answers |
|---|---|---|---|
| `zero-shot` | COCO `yolo26n.pt` | none | How good is the model the app uses today on this domain? |
| `scratch` | random (`yolo26n.yaml`) | all | What does pre-training give us? |
| `head` | COCO | backbone (first 10 layers) frozen | Is feature extraction enough? |
| `full` | COCO | all | Full fine-tuning |
| `full_f010/025/050` | COCO | all, on 10/25/50 % of train | How many labelled images are needed (learning curve)? |

**Evaluation** (`evaluate.py`). The same script scores every model on the **held-out test split**. That split is never used for training or for model selection, which uses val.
- Model quality: mAP@0.5:0.95, mAP@0.5, AR@100 and AP per class, computed with pycocotools (the standard COCO implementation).
- At the app's auto-label threshold (conf 0.25, IoU 0.5): precision, recall and F1.
- **Effort estimate:** labelling by hand means drawing every box (GT actions). With pre-labels, the annotator deletes each false positive and draws each missed box (FP + FN actions). `actions saved = 1 - (FP + FN) / GT`. This is a lower-level proxy only: box-adjust edits are not counted. The in-app user study measures real time and clicks.
- Failure cases: the 8 worst test images per model, drawn with ground truth (green) and predictions (red), in `results/<tag>/failures/`.

## Run

```bash
bash training/run_all.sh                                  # whole experiment, ~1 h on an RTX A6000
SEEDS="0 1 2" bash training/run_all.sh                    # 3 seeds for mean ± std (about 3x time)
EPOCHS=5 FRACTIONS="" VARIANTS=full bash training/run_all.sh   # quick smoke test
```

Single steps:

```bash
. .venv-train/bin/activate
python training/train.py --data african-wildlife.yaml --variant full
python training/evaluate.py --data african-wildlife.yaml --split test \
    --model zero-shot=yolo26n.pt --runs training/runs/african-wildlife
```

### Training on data annotated with the app

The workflow is: auto-label, correct in the app, export, fine-tune, then auto-label better.

```bash
# in the app: Export -> YOLO (.zip), copy the zip to the Linux machine
python training/prepare_app_export.py my_export.zip --out training/datasets/my_project
DATA=training/datasets/my_project/data.yaml bash training/run_all.sh
```

`prepare_app_export.py` makes a seeded 70/15/15 split and drops images without boxes (in the app those are usually not annotated yet).

### Use the fine-tuned model in the app

```bash
. .venv-export/bin/activate
python models/export_yolo.py --weights training/runs/african-wildlife/full_s0/weights/best.pt \
    --out models/weights/yolo_wildlife.onnx
YOLO_MODEL=yolo_wildlife.onnx ./run.sh
```

The class names travel inside the ONNX metadata, so auto-label creates the new labels by name. The default `yolo.onnx` (COCO) stays untouched.

## Data engine round (does fine-tuning on your own labels help the assistant?)

A small version of the SAM data engine (Kirillov et al. 2023): label, fine-tune on the corrected labels,
label new images with the better model, compare. Use two batches of new images of similar difficulty.

```bash
# Round 0: start a labeling session named "round0" (AI-assisted), label batch A with the stock model.
#   Tip: "Label these next" on the project page picks the images the model is least sure about.
# Export "YOLO (.zip)" with "Only images marked done", then:
python training/prepare_app_export.py round0_yolo.zip --out training/datasets/round0 --test 0
python training/train.py --data training/datasets/round0/data.yaml --variant full --tag round0
python models/export_yolo.py --weights training/runs/round0/full_s0/weights/best.pt \
    --out models/weights/yolo_round1.onnx

# Round 1: restart with the fine-tuned model, session named "round1", label batch B.
YOLO_MODEL=yolo_round1.onnx ./run.sh

# Report: download events.csv from the Effort stats page, then
python training/data_engine.py project_1_events.csv     # writes results/data_engine.md
```

The report shows, per round: share of final objects the AI provided, accept rate, share of AI labels
the user had to fix, objects drawn by hand, and time per image and per object. Batches differ per
round, so state this in the report and back it with `evaluate.py` on a fixed held-out test set.

## How to verify the results

- `results/<tag>/summary.md`: the main table (copy it into the report).
- `results/<tag>/summary.csv`, `per_class.csv`, `metrics.json`: the raw numbers, including dataset size and thresholds.
- `results/<tag>/runs/<run>/`: `results.csv` / `results.png` (loss and mAP per epoch), `args.yaml` (exact Ultralytics settings), `train_config.json` (variant, start weights, frozen layers, Ultralytics version).
- To re-check one number, run `evaluate.py` again on a `best.pt`. Training uses `deterministic=True` and a fixed seed, but GPU kernels can still vary slightly between machines.

`runs/` (weights) and `datasets/` are git-ignored. `results/` is committed.

## Limits

- Single seed by default. Use `SEEDS="0 1 2"` before claiming small differences between variants.
- `fraction` uses the first N % of the training list, not a random sample.
- The effort number is an estimate from detections, not measured annotator time.
- The `scratch` run gets the same epoch budget as the others, which is fair for comparing cost but not its best possible result.

## References

- Ultralytics YOLO (YOLO26 / YOLO11), AGPL-3.0: https://github.com/ultralytics/ultralytics
- African Wildlife dataset (Ultralytics), AGPL-3.0: https://docs.ultralytics.com/datasets/detect/african-wildlife/
- COCO metrics: pycocotools, https://github.com/cocodataset/cocoapi. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
- Data engine: Kirillov et al., "Segment Anything", ICCV 2023 (arXiv:2304.02643).
- Uncertainty sampling ("Label these next"): Settles, "Active Learning Literature Survey", 2009.
- Transfer learning: Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014.
