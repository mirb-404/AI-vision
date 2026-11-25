# Project report: 2026-11-25

## Summary

Checked the exam requirements in the lecture slides (`AI_Vision/01 - Introduction Fundamentals.pdf`) against the current state of the app. The biggest gap is that there is no measured evidence yet: no model-quality numbers apart from one SAM IoU on `bus.jpg`, and no manual-vs-assisted effort numbers. The deadline is **Sunday, 30 November 2026, 23:59**.

Today's work:

1. A separate `training/` folder with a full, reproducible transfer-learning experiment for the YOLO pre-label model.
2. An "always-on" AI assistant in the editor, with a gradient glow animation whenever a model is working.
3. A beginner-friendly simple view as the default. The AI takes over the hard parts (marking sure objects, naming shapes). The full tool set moves behind an "Expert tools" switch.

No commits yet. The training and AI flows still need a run on Datalab.

## Exam requirements vs. state

| Requirement (slides) | State |
|---|---|
| Upload and label images | Done |
| At least one model helps inside the app | Done (SAM 2.1 tiny, YOLO26n), now always on |
| Annotations saved and exportable | Done (COCO, YOLO, annotated images) |
| Model quality (mAP, IoU) | Scripts written today (`training/evaluate.py`), **not run yet** |
| Effort saved, manual vs assisted, with failure cases | Effort estimate and failure images in `training/evaluate.py`. In-app user study **not run yet** |
| Runs with one command, eval scripts in repo | `./run.sh`, `bash training/run_all.sh`. Not verified on Datalab |
| Report, demo video, exported dataset | Not started |

## Done

### Training and fine-tuning (`training/`)

**What is fine-tuned and why.** The YOLO detector is fine-tuned, not SAM.
- SAM does not know classes and already reaches about 0.97 IoU from one click, so fine-tuning it would gain little for a lot of GPU time.
- The COCO-trained YOLO only knows 80 classes. On a new domain it misses objects or names them wrong, and every miss is a box the user has to draw by hand. Fine-tuning targets exactly that.

**Dataset.** Ultralytics African Wildlife: 4 classes (buffalo, elephant, rhino, zebra), about 1,500 images, fixed train/val/test splits, downloaded automatically.
- COCO has elephant and zebra but not buffalo or rhino. The zero-shot COCO model can therefore only partly succeed, which makes the effect of transfer learning visible.

**Experiment.** Every run uses the same data, epochs, image size and seed; only the starting point changes (lecture 08, Transfer Learning):
- `zero-shot`: COCO `yolo26n.pt`, no training (the model the app uses today).
- `scratch`: random weights. Shows what pre-training is worth.
- `head`: COCO weights with the first 10 layers (backbone) frozen. Feature extraction only.
- `full`: COCO weights, all layers trained. Full fine-tuning.
- `full_f010/025/050`: full fine-tuning on 10 / 25 / 50 % of the training split. Learning curve: how many labelled images are needed.

**Evaluation (`training/evaluate.py`).** One script scores every model the same way on the held-out test split. The test split is never used for training or model selection.
- pycocotools mAP@0.5:0.95, mAP@0.5, AR@100 and AP per class.
- Precision, recall and F1 at the app's auto-label threshold (conf 0.25, IoU 0.5).
- Effort estimate: `actions saved = 1 - (FP + FN) / GT`. Manual labelling means drawing every box; with pre-labels the user deletes false positives and draws the misses. Box-adjust edits are not counted, so this is an estimate. The in-app study measures real time and clicks.
- The 8 worst test images per model, rendered with ground truth (green) and predictions (red), as failure cases for the report.
- The zero-shot model is mapped to the dataset classes by name. Classes it does not know score 0.

**Other files.**
- `training/run_all.sh`: the whole experiment in one command, Linux only. Options: `DATA`, `MODEL`, `EPOCHS`, `SEEDS`, `VARIANTS`, `FRACTIONS`.
- `training/train.py`: one variant.
- `training/prepare_app_export.py`: turns a YOLO zip exported from the app into a seeded 70/15/15 split.
  - Images without boxes are dropped by default, because in the app they usually mean "not annotated yet".
  - This closes the loop: auto-label, correct, export, fine-tune, better auto-labels.
- `training/README.md`: approach, commands, how to check the results, limits, references.

**Results stay checkable.**
- `training/results/<tag>/` is committed: `summary.md`, `summary.csv`, `per_class.csv`, `metrics.json`, failure images, and per run `results.csv`, `args.yaml` and `train_config.json`.
- `training/runs/` (weights) and `training/datasets/` are git-ignored.

**Using the fine-tuned model in the app.** Export it with `models/export_yolo.py --weights .../best.pt --out models/weights/yolo_wildlife.onnx`, then start with `YOLO_MODEL=yolo_wildlife.onnx ./run.sh`. Class names travel in the ONNX metadata, so auto-label creates the new labels by name. The default COCO `yolo.onnx` stays untouched.

### Always-on AI assistant and AI animations

- In AI mode, YOLO runs by itself on every image that opens. There is no button to remember. An "Always on" switch turns this off; it is remembered per project.
- Suggestions that overlap an existing annotation (IoU ≥ 0.5) are not proposed again.
- Results that come back after the user has moved to another image are dropped. The old code could show the previous image's boxes on the new one.
- **Animation while a model works** (finding objects, reading the image for SAM, tracing an outline): a rotating multicolour gradient ring around the canvas, a soft halo, a light band sweeping over the image, and a status pill with shimmering text and a ✦ sparkle.
- When results arrive, a toast says what the AI did, and the suggestion list fades in item by item.
- The overlay never blocks input. Animations are reduced when the device asks for reduced motion.
- The style is a generic gradient "AI glow" (inspired by phone AI features), with no third-party branding.

### Simple view for beginners (default) and Expert tools

The goal is that someone who knows nothing about annotation can label images. The AI and backend do the complex work; the user gets only the settings that matter.

**AI takes over:**
- **Sure objects are marked automatically.** Detections with score ≥ 0.5 become annotations straight away (switch: "Mark sure objects for me"). Only unsure ones are shown, as "Is this a zebra? ✓ Yes / ✕ No".
- **Shapes are named by the AI.** A box or Magic-select outline that overlaps something the detector saw (IoU ≥ 0.5, score ≥ 0.25) gets that name automatically.
- **No label setup needed first.** If the AI cannot name a shape, a "What is this?" card asks. It offers existing names as chips; typing a new name creates the label. Enter reuses the last name; Esc discards the shape.
- Suggestions that the user's own shape makes redundant are removed.

**Simplified UI:**
- Four tools, easiest first: ✦ Magic select, Draw box, Move / fix, Pan. Magic select is the starting tool when SAM is available. Polygon is an expert tool.
- Plain wording everywhere:
  - "Done, next image ›" and "Saved ✓";
  - "Tap any object. AI outlines it for you." and "✓ Looks right / Start over";
  - "Objects on this image", with "you / ✦ AI / ✦ AI · fixed" tags.
- The tour is rewritten in plain language and gains a stop for Expert tools.
- Hidden in the simple view: the AI/manual mode switch (still on the project page), label manager, confidence slider, score percentages, keyboard shortcuts, image download, and model names.
- An **Expert tools** checkbox in the header shows all of it again. It is remembered per project.

**Effort study.** Auto-marked objects are logged as `proposal_accepted` with `auto: true` and committed with tool `auto`. The report can therefore separate them from accepts the user clicked. Manual mode is unchanged, so the baseline stays clean.

## Tests

- Training scripts: `py_compile` and `bash -n` pass. A unit test of the TP / FP / FN matching passed with stubbed dependencies. It covered a duplicate box, a wrong class and a below-threshold box; result TP 1, FP 2, FN 2, as expected.
- Frontend: TypeScript check and production build pass.
- Browser check in Chrome (local backend, no models, scratch data folder):
  - The simple view shows 4 tools and no label or mode controls.
  - Drawing a box opens "What is this?". Typing a name created the label, and the box saved with the "you" tag.
  - The second box offered the existing name as a chip. Esc discarded it.
  - Expert tools brought back the mode switch, label manager, polygon tool, download and shortcuts.

## Not yet verified

- The training experiment on Datalab: `bash training/run_all.sh`. First a smoke test: `EPOCHS=5 FRACTIONS="" VARIANTS=full bash training/run_all.sh`.
- Always-on detection, auto-marking, AI naming and the glow animation with real models (no models on the dev machine).
- `yolo26n.yaml` for the scratch run resolving in the installed Ultralytics version.

## Risks and limits

- Training uses one seed by default. Differences of a point or two between variants need `SEEDS="0 1 2"`.
- `fraction` takes the first N % of the training list, not a random sample.
- The effort number from `evaluate.py` is an estimate from detections, not measured annotator time.
- While the "What is this?" card is open, the shape just drawn is not shown on the image.
- When the AI is unavailable, the two AI switches look "off" even though the saved setting is still on.
- Auto-marking at 0.5 can add wrong boxes on an unfamiliar domain. The fine-tuned model should reduce this; the user can still delete them, and they are tagged "✦ AI".

## Next

1. Run the training experiment on Datalab and commit `training/results/`.
2. Test the always-on assistant and simple view with real models, on desktop and phone.
3. Run the in-app user study (manual vs AI-assisted) for measured time and clicks.
4. Annotate and export the submission dataset.
5. Write the report and record the demo video.
