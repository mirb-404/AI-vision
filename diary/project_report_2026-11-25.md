# Project report: 2026-11-25

## Summary

Checked the exam requirements in the lecture slides (`AI_Vision/01 - Introduction Fundamentals.pdf`) against the current state of the app. The biggest gap is that there is no measured evidence yet: no model-quality numbers apart from one SAM IoU on `bus.jpg`, and no manual-vs-assisted effort numbers. The deadline is **Sunday, 30 November 2026, 23:59**.

Today's work:

1. A separate `training/` folder with a full, reproducible transfer-learning experiment for the YOLO pre-label model.
2. An "always-on" AI assistant in the editor, with a gradient glow animation whenever a model is working.
3. A beginner-friendly simple view as the default. The AI takes over the hard parts (marking sure objects, naming shapes). The full tool set moves behind an "Expert tools" switch.
4. Smoother, much subtler AI animations.
5. **Find alike objects:** mark one object (for example one cricket ball) and the AI marks every object of that kind in all images, using OWLv2 image-guided detection.
6. A `samples/` folder with 10 test images, and a fix for the SAM 2 install on Datalab.

Items 1–3 and the first animation version are in commit `a2bad9f`. Items 4–6 are not committed yet. The training and all AI flows still need a run on Datalab.

## Exam requirements vs. state

| Requirement (slides) | State |
|---|---|
| Upload and label images | Done |
| At least one model helps inside the app | Done (SAM 2.1 tiny, YOLO26n), now always on. OWLv2 added for find alike objects |
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
- **Animation while a model works** (finding objects, reading the image for SAM, tracing an outline, searching for alike objects): a slowly turning pastel gradient ring around the canvas, a faint halo, a soft light band sweeping over the image, and a status pill with shimmering text and a ✦ sparkle.
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

### Subtler animations

The first version felt too fast and jarring. Every motion is now slower, smaller and eased:
- The gradient ring turns once every 9 s (was 2.6 s), is thinner (2 px) and 70 % opaque, with pastel colours instead of saturated ones.
- The halo is much dimmer (opacity 0.16, was 0.45). The light sweep is about a third as bright and takes 6 s (was 1.8 s).
- The overlay fades in and out over 700 ms (was 300 ms). The shimmer takes 6 s; the ✦ sparkle barely pulses (scale 1.08, was 1.3).
- The AI toast stays 4.2 s and fades without moving. Tool switches no longer bounce or overshoot. Fade-ins are slower and move only 3 px. Suggestions appear 80 ms apart.
- The reduced-motion setting still turns animation off.

### Find alike objects (OWLv2, one-shot)

**What the user does.** Mark one object, for example one cricket ball, and name it. A card asks "Mark every 'ball' in all images? [Find all]". The search then runs image by image with a progress bar and a Stop button, and ends with "Marked 5 more 'ball' in 3 images [Undo]". The ⧉ button on any object in the side list starts the same search. The goal is every object of the **same kind**, including ones that look different (another colour or size), not only copies of the example.

**How we got there.** Two earlier versions were built and replaced today. Both are kept here because they explain the model choice:
1. Matching SAM encoder features to the example (a PerSAM-style prototype) and grouping matching cells into blobs. That suits texture areas like grass, but not separate objects.
2. The same SAM features, with one peak per object and SAM segmenting each peak. That found separate objects, but SAM features mostly encode *appearance*, so it only found near-identical copies.

Only a model trained on categories finds "same kind, different look". That led to OWLv2.

**How it works.**
- OWLv2 (Minderer et al., NeurIPS 2023; `google/owlv2-base-patch16-ensemble`, Apache-2.0) predicts, for every image patch, a box, an objectness score and a class embedding. The class embedding lives in the same space as CLIP text and image queries.
- **Query.** On the source image, the predicted boxes whose overlap with the example is at least 0.8 × the best overlap are candidates. The one least like the image's average embedding becomes the query; this is the same rule as Hugging Face `embed_image_query`.
- **Scoring.** Every patch of every image gets `sigmoid((cos(patch, query) + shift) × scale)`, using OWLv2's own class head. Results pass a threshold and NMS (IoU 0.3).
- **Self-calibrating threshold.** Image-query scores are not probabilities, so the cut-off is set per example. It sits between the source image's background level (95th percentile of patches away from the example) and the example's own score: 30–80 % of the way, chosen by a strictness setting, default 55 %.
- **Duplicates.** The example and objects that already have that label (box IoU ≥ 0.5) are skipped. Running the search twice adds nothing.
- **Output kind.** A box example gives boxes. An outline example gives outlines: SAM turns each found box into an outline, or the box rectangle is used when SAM is missing.
- **Flow.** One request per image, so progress is live and no request hits the ~100 s tunnel limit. The open image's results go into the editor, where normal undo works. Other images are saved by the server. `POST /api/similar/undo` deletes only the AI annotations the search created.

**Files.**
- `backend/app/inference/owl.py`: preprocessing (grey square padding, 960 px, CLIP normalisation), query selection, threshold, scoring, NMS, and the embedding cache.
- `models/export_owlv2.py`: exports the per-patch outputs to ONNX and checks them against PyTorch.
- `POST /api/images/{id}/similar` in `routes/ai.py`.
- Wiring: an `owl` entry in `/api/ai/status`, `OWL_MODEL` in config, `./run.sh` setup exports OWLv2, `transformers` added to the export requirements, licence and citation in `models/README.md`.
- Frontend: `SimilarSearch.tsx`, the ⧉ button, and the offer shown only when OWLv2 is present.
- The SAM look-alike code from the first two versions was removed. SAM's decoder is now split into a reusable `_decode`, and click / box segmentation is unchanged.

**Exam guidelines.** Allowed: the rules permit pretrained models and libraries if cited. OWLv2 is a pretrained model, not an annotation tool. The search logic, calibration, dedupe, undo and UI are our own code. The report must cite OWLv2 and OWL-ViT, justify the choice (category-aware rather than appearance-only, shown by the two earlier versions), and evaluate the feature: precision / recall of found objects per example, clicks saved, failure cases.

### Sample images (`samples/`)

- `samples/images/`: 8 Ultralytics coco8 photos (COCO val2017), plus `bus.jpg` (4 people) and `zidane.jpg` (2 people). The last two are good tests for find alike objects.
- `samples/labels/`: COCO ground-truth boxes for the 8 coco8 photos, in YOLO format.
- `samples/README.md`: sources and licences (COCO annotations CC BY 4.0; images under their Flickr licences; Ultralytics example images). About 640 KB in total.

### Datalab setup fix

`./run.sh` stopped during model export with `No matching distribution found for sam2`. Meta's SAM 2 repository names its package `SAM-2`, while the requirement said `sam2`. Newer pip rejects that mismatch and then looks for an unrelated `sam2` on PyPI. The requirement is now `SAM-2 @ git+https://github.com/facebookresearch/sam2.git`.

## Tests

- Training scripts: `py_compile` and `bash -n` pass. A unit test of the TP / FP / FN matching passed with stubbed dependencies. It covered a duplicate box, a wrong class and a below-threshold box; result TP 1, FP 2, FN 2, as expected.
- Frontend: TypeScript check and production build pass.
- Backend: 15 tests pass. The 4 find-alike tests use fake OWLv2 outputs:
  - the query comes from the marked object, and an example outside every box is rejected;
  - a ball of a different look is found, but not a bat and not the background;
  - several predicted boxes per ball merge into one; a stricter setting finds no more;
  - boxes are scaled back to original pixels with the square padding, and embeddings are cached;
  - the API returns 503 without weights, shows `owl` in the status, saves boxes for a box example and outlines for an outline example, adds no duplicates on a second run, and Undo deletes only AI annotations.
- The `SAM-2` requirement fix is not tested here: there is no Linux environment on the dev machine.
- Browser check in Chrome (local backend, no models, scratch data folder):
  - The simple view shows 4 tools and no label or mode controls.
  - Drawing a box opens "What is this?". Typing a name created the label, and the box saved with the "you" tag.
  - The second box offered the existing name as a chip. Esc discarded it.
  - Expert tools brought back the mode switch, label manager, polygon tool, download and shortcuts.

## Not yet verified

- The training experiment on Datalab: `bash training/run_all.sh`. First a smoke test: `EPOCHS=5 FRACTIONS="" VARIANTS=full bash training/run_all.sh`.
- Always-on detection, auto-marking, AI naming and the glow animation with real models (no models on the dev machine).
- `yolo26n.yaml` for the scratch run resolving in the installed Ultralytics version.
- `./run.sh setup` on Datalab with the `SAM-2` fix, and the OWLv2 export (`python models/export_owlv2.py`, about 600 MB download), including its ONNX-vs-PyTorch check.
- Find alike objects on real photos (`samples/images/bus.jpg`), including whether the self-calibrating threshold is too loose or too strict.
- The softer animations seen with real models running.
- `docker compose up` with `owlv2.onnx` present.

## Risks and limits

- Training uses one seed by default. Differences of a point or two between variants need `SEEDS="0 1 2"`.
- `fraction` takes the first N % of the training list, not a random sample.
- The effort number from `evaluate.py` is an estimate from detections, not measured annotator time.
- While the "What is this?" card is open, the shape just drawn is not shown on the image.
- When the AI is unavailable, the two AI switches look "off" even though the saved setting is still on.
- Auto-marking at 0.5 can add wrong boxes on an unfamiliar domain. The fine-tuned model should reduce this; the user can still delete them, and they are tagged "✦ AI".
- Find alike objects: the threshold rule is a design choice, not yet tuned on data. Very small objects may be missed at 960 px input. OWLv2 base is about 600 MB and slow on CPU (fine on the A6000).
- Ultralytics YOLO is AGPL-3.0. Name it in the report.

## Next

1. Run the training experiment on Datalab and commit `training/results/`.
2. Run `./run.sh setup` (SAM-2 fix, OWLv2 export). Test the always-on assistant, simple view and find alike objects with real models, on desktop and phone.
3. Evaluate find alike objects for the report: one example per class, precision / recall of found objects, clicks saved, failure cases.
4. Run the in-app user study (manual vs AI-assisted) for measured time and clicks.
5. Annotate and export the submission dataset.
6. Write the report and record the demo video.
