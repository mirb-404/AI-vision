# Project report: 2026-11-27

## Summary

Today was about getting ready for the Datalab runs, and about one design question: **how can the assistant be useful for many kinds of objects, not only the 80 COCO classes?**

Built today:

1. `evaluation/collect_metrics.py`: one command gathers every metric into one bundle (`METRICS.md` + zip) for analysis. Committed in `3a3d099`.
2. **A bigger auto-label model by default:** YOLOv8s trained on Open Images V7 (601 classes) replaces YOLO26n COCO (80 classes) as the auto-label detector.
3. **Open-vocabulary fallback:** project labels that the detector does not know are searched by name with OWLv2. Suggestions and "Auto-label all images" now work for any label name.

Items 2–3 are in commit `84bcf31`. A first test on Datalab then changed the plan; see "Update (later the same day)" at the end.

## Thought process

### Why fine-tune at all, when the product is an annotation platform?

The question came up before starting training on Datalab. The exam does not ask for fine-tuning. It asks for:
- a model that helps inside the app;
- measured model quality;
- measured effort saved.

Fine-tuning is still worth doing, for two reasons:

- **Product:** the assistant only helps with classes it knows. With the stock COCO model, a user labeling rhinos or product defects gets no suggestions and draws every box by hand. The way commercial tools solve this is the loop "label some images, retrain on them, the assistant gets better". The `training/` folder and `training/data_engine.py` show that loop with numbers.
- **Evidence:** zero-shot vs fine-tuned mAP gives the model-quality number. Fewer missed objects means fewer manual boxes, which gives the effort number. The scratch / head-only / full comparison is lecture 08 (transfer learning) put into practice.

**Decision:** keep training. It runs unattended on the GPU, so it costs little time, but it is not on the critical path. If it fails, the core grade still stands on the app, the click benchmark and the user study.

### There are too many classes: how to make the assistant good for many things?

No single trained model knows every class. Options considered:

| Option | Covers | Cost | Decision |
|---|---|---|---|
| Open-vocabulary detector driven by the label name (OWLv2, already in the app for "Find by name") | any name | backend wiring only; slower and less precise than a trained detector | **Done today** as the fallback for unknown labels |
| A detector trained on more classes: YOLOv8 Open Images V7 (601 classes) | 601 everyday classes | no code, only a different export | **Done today**: new default |
| YOLO-World / YOLOE (fast open-vocabulary YOLO) | any name, fast | new export script, re-export when labels change, untested export path | Not now: too risky with 3 days left. Future work |
| Fine-tuning on the user's own labels (data engine) | the user's classes, high accuracy | a training round | Already built (`training/`) |

**Resulting design, in two layers:**
1. **From day one, any class.** The 601-class detector covers common objects. OWLv2 covers every other label name.
2. **Over time, accurate on the user's domain.** Fine-tuning on the labels made in the app.

This is the same "label, then retrain" story as the SAM data engine, and it answers "why fine-tune" as part of the product.

## Done

### Default detector: YOLOv8s Open Images V7

- `run.sh` now exports `YOLO_WEIGHTS=yolov8s-oiv7.pt` by default (override with `YOLO_WEIGHTS=yolo26n.pt` for COCO).
- A new file `models/weights/yolo.source` records which weights `yolo.onnx` came from. When `YOLO_WEIGHTS` changes, `run.sh` re-exports automatically. The existing COCO `yolo.onnx` on Datalab will therefore be replaced on the next `./run.sh`.
- The backend needed no change: it already reads class names from the ONNX metadata and decodes YOLOv8 output.
- The "s" size was picked over "n" because the Datalab GPU makes the extra cost small and the 601-class task is harder. Ultralytics reports a clearly higher mAP for s than n on Open Images V7. **Check the exact numbers in their docs before citing them.**
- The label hints in the UI no longer say "use COCO names". They now say any plain name works.
- Citation added (`rules.md`, `models/README.md`): Open Images V7 (Kuznetsova et al., IJCV 2020, CC BY 4.0 annotations), Ultralytics YOLOv8 (AGPL-3.0).

### Open-vocabulary fallback for unknown labels

- `POST /api/images/{id}/detect` takes `names` (the project's label names). The editor's always-on assistant now sends them.
- Names the detector does not know (case-insensitive) are searched with OWLv2 text queries, at most 20 names per image. They come back as normal suggestions with `class_id = -1`.
- The cut-off is the "normal" find-by-name threshold (0.2).
- **Auto-accept is unchanged:** only suggestions scoring 0.5 or more are added without a click. OWLv2 text scores are usually lower than YOLO's, so most OWLv2 suggestions wait for a Yes/No. That is the safer default for a weaker model.
- "Auto-label all images" does the same for the project's labels.
- Without OWLv2 weights everything behaves as before (YOLO only, no error).
- Inference time is logged as `owlv2/detect_text` and `owlv2/auto_label_text`, so the stats page shows the extra latency.

### Metrics bundle (`evaluation/collect_metrics.py`)

- One command on Datalab (with the app running) writes `metrics_bundle_<date>/` and a `.zip`:
  - `METRICS.md`: all numbers in one file;
  - `environment.json`: commit, GPU, model files;
  - training and click-benchmark results;
  - per project: stats, AI scorecard and `events.csv`.
- Text files only. Bundles are git-ignored because `events.csv` contains participant names; `--no-events` leaves the raw log out.
- Tested locally against a running backend.

## Tests

- Backend: **22 passed**. New test: detect returns YOLO boxes for known labels and OWLv2 boxes only for unknown ones (case-insensitive, no duplicate names), works without OWLv2, and auto-label uses the same fallback. The fake detector in `test_api.py` now has class names, like the real one.
- Frontend: `tsc` and `vite build` pass. `run.sh` passes `bash -n`.

## Not yet verified

- Export of `yolov8s-oiv7.pt` on Datalab. The file name is from the Ultralytics docs and has not been downloaded yet.
- Suggestion quality and latency of the OWLv2 fallback on real photos. Several unknown labels mean several text queries per image. The scoring pass is cheap, but the OWLv2 image pass (cached per image) is heavier than YOLO, especially on CPU.
- Everything from yesterday's "not yet verified" list.

## Risks and limits

- Open Images class names differ from COCO's (for example "Human face", "Person" capitalised). Label matching ignores case but not wording, so "people" will not match "Person". OWLv2 then searches "people" by name, which still gives suggestions.
- The training experiments in `training/` still use YOLO26n COCO as the zero-shot baseline. That is fine: they measure transfer learning, not the app's default detector. Say so in the report so the two are not mixed up.
- With 3 days left, today's code is the last feature work. From now on: Datalab runs, user study, report, video.

## Update (later the same day): first real test on Datalab, and a change of approach

### What was seen

Both detectors were tried in the app on Datalab with the same photo: elephants with a person.

| Detector | What it marked | Strength | Weakness |
|---|---|---|---|
| YOLO26n, COCO, 80 classes (old default) | **Only the person.** The elephants got no box, although COCO has an "elephant" class | Clean, no noise | Too little: one "person" box per image is not a useful annotation product |
| YOLOv8s, Open Images V7, 601 classes (new) | **Many more things, with finer names**: the label set is far richer (Open Images names things like Man/Woman, Human face, Human hand, Footwear, Tree) | Much more "sophisticated" labelling, closer to what a real dataset needs | **Missed the elephants too**, and leans heavily on scenery classes: it drew a "Tree" box that covers almost the whole image |

### Reasoning

- First reaction was to go back to the 80-class model as the default and show only the project's own labels. That change is in commit `6c27352`, which is the current state.
- That fixes the noise, but it throws away the reason the 601-class model was tried. On the same image it was clearly more useful: more objects, more detail.
- The product goal is not "mark the human and move on". It is an assistant that does the detailed labelling and leaves the user only the checking.
- So the two models are not rivals. Each covers the other's weakness:
  - the 601-class model brings breadth and detail;
  - a second model catches what it misses;
  - a simple rule removes scenery boxes that fill the whole image.
- Precision of the shapes matters as much as the names. Boxes are rough; SAM is already in the app, so every AI label can be traced as an exact outline at almost no extra cost.

### Scope check against the exam

- The brief allows pretrained models and libraries if they are cited. Feature set and models are our choice.
- "At least one model" is a minimum, not a limit. Two cited detectors are fine; only forking or wrapping annotation tools is forbidden.
- No new training and no new UI concept. Both models are already exported. The change is one backend merge function plus one setting.
- It also strengthens the evidence. The report can compare 80-class vs 601-class vs merged on the same images (hits, misses, noise), and boxes vs outlines in the user study (time and edits per object).
- VRAM is not a constraint. The extra detector adds roughly 0.2–0.5 GB (estimate). The whole app with all models is about 3–5 GB next to training (3–6 GB) on a 48 GB A6000. Check with `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

### New approach (decided, to build next)

1. **Merge both detectors.**
   - Run the 601-class model (default again) and the 80-class model on each image.
   - Where boxes from the two overlap strongly (the same object, possibly named differently, such as "person" vs "Man"), keep the more confident one.
   - Drop boxes from the 601-class model that cover more than 80 % of the image (whole-image "Tree").
   - The cost is about 2× detector time: tens of milliseconds on the GPU.
2. **Auto-outline every AI label with SAM.**
   - Suggestions arrive as precise outlines instead of boxes. SAM reads the image once already; tracing each box costs a few milliseconds.
   - A simple setting "AI draws: Outlines / Boxes" (default Outlines). YOLO-format export still writes boxes.
3. Keep what already works:
   - "Is this a …? ✓ / ✕" for unsure items;
   - Magic select and Find by name for anything missed;
   - Remove AI labels and Undo for mistakes;
   - OWLv2 for label names neither detector knows.

What the user should experience: open an image, and people, faces, helmets and elephants are already traced. Confirm the unsure ones, tap anything missed, next image.

**Out of scope on purpose:** attributes (occluded, truncated), part hierarchies, video. They are more product, not more exam value.

**Time budget:** about one day including tests. After that no more features: two days for the Datalab runs, user study, report and video.

## Next

1. Build the merged detector and auto-outlines (see above), with tests.
2. Push. On Datalab: `git pull`, start `training/run_all.sh` in the background, then `./run.sh` (exports both detectors).
3. Re-test the elephant photo with the 80-class, 601-class and merged detectors. Save screenshots for the report.
4. Click benchmark, user study, then `python3 evaluation/collect_metrics.py` and send `METRICS.md` for analysis.
