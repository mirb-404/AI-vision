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

### Datalab `./run.sh datalab` result (after `6c27352`)

- Passed:
  - unit tests (22);
  - model smoke test (SAM 324 ms encoder / 67 ms decoder, score 0.969; YOLO 5 detections: bus + 4 persons);
  - UI build;
  - all end-to-end API checks up to SAM and YOLO detect.
- **Failed: "YOLO auto-label project: nothing saved".** Cause: the end-to-end check creates a label called "object". With the new "only suggest the project's labels" rule, bus and person were correctly skipped. The rule works; the check was out of date.
  - Fix: the check now adds a label named after the first class YOLO detected before running auto-label.
- **`./run.sh: line 1: srv: unbound variable`** after the failure. Cause: the clean-up trap used a `local` variable, which no longer exists when the trap runs after a failed check (`set -u`).
  - Fix: global `E2E_SRV` / `E2E_DATA`.
- The ONNX Runtime warnings in the log (shape merge, Memcpy nodes, ScatterND) are harmless and were there before.

### Built: merged detector and exact outlines

- **Merged detector** (`backend/app/inference/yolo.py`: `merge_detections`, `MergedDetector`):
  - `yolo.onnx` = YOLOv8s Open Images V7 (main) and `yolo_coco.onnx` = YOLO26n COCO (extra), both exported by `run.sh`. Each has a `.source` file, so changing `YOLO_WEIGHTS` / `YOLO_EXTRA_WEIGHTS` re-exports only that model.
  - Merge rules:
    1. Boxes from the main model covering more than 80 % of the image are dropped (scenery).
    2. Boxes from different models with IoU ≥ 0.7 are one object: the more confident one is kept.
    3. Boxes of the same model never suppress each other, so a face inside a person stays.
  - Every box is tagged with the model that found it (`model`: `yolo`, `yolo_coco`, `owlv2`). This shows which model contributed what, for the report.
  - The rest of the app sees one detector with the same interface as before. If `yolo_coco.onnx` is missing, or `YOLO_EXTRA_WEIGHTS=""` / `YOLO_EXTRA_MODEL=""`, it runs the main model alone.
- **Exact outlines** (editor, setting "Exact outlines", on by default, needs SAM):
  - Every accepted AI suggestion, including auto-accepted ones, is traced by SAM along the object's edge and saved as an outline. A box SAM cannot trace stays a box.
  - The SAM decoder runs per box on the already-read image, about 67 ms each on Datalab, all in parallel.
  - Not changed: "Auto-label all images" on the project page still saves boxes. Outlines there would mean many SAM encoder runs per project, which is deliberately left out.
- Tests: backend **25 passed**. New: merge (scenery dropped, "Man" vs "person" de-duplicated, face kept, elephant filled in; one model keeps big boxes) and `MergedDetector` (model tags, unique class ids). Frontend `tsc` and build pass; `run.sh` passes `bash -n`.

### Datalab check of the merged detector: richer labels, and the next question

The merged detector was run on Datalab. The result looks **clearly more sophisticated**: the 601-class model contributes finer names, and the merge keeps the output clean.

That raised the next question: is "more classes" enough, or how deep do professional annotations go? Before adding anything, we checked what industry leaders and the big public datasets actually record per object.

### What industry leaders annotate (research, 27 Nov)

| Layer | Industry practice | Our app before today |
|---|---|---|
| Class vocabulary | COCO 80 → Open Images 600 box classes (20k image-level) → LVIS 1,203 classes with a long tail (rare / common / frequent) | 601 + 80 merged, plus OWLv2 for typed names |
| Class hierarchy | Open Images classes form a tree (e.g. Man under Person) | none |
| Shape | box → mask (Open Images: 2.8M masks in 350 classes) → keypoints | box + SAM outline |
| Per-object attributes | Scale: occlusion level per vehicle or pedestrian (categorical / number / angle / text attributes). Labelbox: nested classifications, e.g. a tree marked healthy or unhealthy. V7: attributes, text, instance ID | none |
| Relationships | Open Images: 3.3M relations ("person rides horse"). Labelbox: named links between two annotations | none |
| Image-level tags, captions | Open Images image-level labels, localized narratives (voice + mouse trace) | none |
| Completeness rule | LVIS: every instance of each *checked* class is labelled, not every class on every image | same idea as our "only suggest the project's labels" rule |

For scale, Open Images has per image on average: 6.7 image-level labels, 8.3 boxes, 1.5 masks, 1.7 relations and 34.8 point labels.

Sources: Labelbox ontology docs; Google Research blog on Open Images V7; Scale API docs on annotation attributes; AWS blog on the LAVA driving dataset; LVIS paper (Gupta et al., CVPR 2019); V7 docs on sub-annotations and properties.

### Decision: what to add, and why

**Added: four layers** that move the labels from "a name and a box" towards what professional datasets record. None needs training. Each reuses a model already in the app.

1. **Per-object attributes: occluded, truncated, and a free note.**
   - This is the most common attribute set in industry (Scale's own example is occlusion level). It is also the VOC standard (`<truncated>`, `<occluded>`).
   - The AI pre-fills both, as an assistant should:
     - truncated = the box touches the image edge;
     - occluded = another object covers 25–90 % of it. Over 90 % means it sits *inside* the other object, like a face in a person: a part, not hidden.
   - Pre-filled values carry a ✦. Changing one is the user's decision, and it does **not** turn the AI label into "fixed", so the scorecard stays honest.
2. **Class hierarchy in exports.**
   - The COCO export's `supercategory` is now real, for example Man → Person. It is not "none" any more.
   - Open Images parents come from Google's published hierarchy (`models/export_class_tree.py`, run on Datalab). COCO's 12 supercategories are built in.
3. **More kinds of things (1,203 LVIS classes).**
   - The idea: OWLv2 already reads each image once. Scoring every patch against all 1,203 LVIS class names is one matrix product, so a large-vocabulary detector comes almost for free.
   - It is **off by default** (a switch in the AI panel). OWLv2's open-vocabulary precision is lower than a trained detector's, and with no project labels yet, 1,203 classes can add noise.
   - Its boxes go through the same merge (same object twice → the more confident one; whole-image scenery dropped).
   - The threshold (0.3) is **untuned**: this is a Datalab test item.
4. **Which model found each label, recorded on acceptance.**
   - The scorecard now has a "Found by model" table with kept / fixed / removed and precision per detector: Open Images, COCO, OWLv2 names, OWLv2 LVIS, find alike, find by name.
   - This gives the report direct evidence for the merge decision.

**Not added, on purpose:**
- **Relationships, captions / localized narratives, keypoints.** Each is a new annotation type with its own UI. That is large work with little exam value, and the deadline is 30 Nov.
- **Attributes, the vocabulary and outlines in "Auto-label all images".** That batch path still saves plain boxes; the editor is where review happens.

### Built and checked

- **Database:** an `attributes` JSON column on annotations. `db._migrate()` adds it to existing databases (the one on Datalab). It was checked on an old local database and is safe to run twice.
- **API:** attributes are validated (unknown keys dropped, wrong types → 422).
- **Exports:**
  - COCO carries the attributes and the supercategory;
  - VOC writes `<truncated>` and `<occluded>`.
- **Detect:** `vocab: true` merges the LVIS boxes in. `/api/ai/status` reports `owl_vocab`.
- **Export side:**
  - `export_owlv2.py` also writes `owlv2_lvis.npz` (`--vocab-only` for Datalab, where OWLv2 already exists);
  - `export_class_tree.py` writes `class_tree.json`;
  - `run.sh` runs both when the files are missing.
- **Tests:** backend **30 passed**. New tests cover:
  - attributes: round trip, validation, COCO and VOC output, supercategory;
  - class tree: Open Images before COCO;
  - LVIS scoring;
  - detect with the vocabulary;
  - scorecard per model.
- **Frontend:** `tsc` and build pass.
- **Browser (local, no weights):** selected an AI object, switched on "Cut off at edge", reloaded. The ⇥ marker was still there and the label stayed "✦ AI" (not "fixed").

### Start over (reset annotations, keep images)

- **Why:** testing the new detectors over and over meant deleting and re-uploading images each time.
- **What:** a "Start over" card on the project page, with the button "Reset all annotations". It asks for confirmation first, because the reset cannot be undone. It deletes every annotation of the project and marks all images *new* again; the images stay.
- **Two optional checkboxes:**
  - clear the effort stats (events + labeling sessions), so test runs do not mix into the user-study numbers;
  - remove the labels.
- It also forgets the "AI labels removed" marks, so the assistant labels every image again.
- Endpoint: `POST /api/projects/{id}/reset`.
- Test added (31 passed). Checked in the browser on scratch data: 3 annotations removed, the image is new again, the image is kept.

## Next

1. Push. On Datalab: `git pull`, start `training/run_all.sh` in the background, then `./run.sh`. It exports `owlv2_lvis.npz` and `class_tree.json`; the database gets the new column automatically.
2. Elephant photo: compare the 80-class, 601-class and merged detectors, then switch on "More kinds of things". Check that the ✦ attributes look right (people at the image border = cut off). Save screenshots.
3. Review 10–20 images, then send `METRICS.md` (`evaluation/collect_metrics.py`). Its "Found by model" table shows which detector earns its place.
4. Click benchmark, user study.
