# Project report: 2026-11-26

## Summary

Read the research notes in `research_notes/AI image annotation landscape/` (competitors, models and evaluation, industry practice, job market) and compared them with the app. The app already had most of what the notes recommend: an effort meter, auto-accept by confidence, AI/you provenance badges, COCO/YOLO export and find alike. The rest was built today:

1. **Find by name**: type "car", the AI marks every car in all images (OWLv2 text queries).
2. **Magic select click benchmark** (`evaluation/noc_benchmark.py`): NoC@85/90 and mIoU per click with a simulated user (RITM protocol).
3. **Data engine round** (`training/data_engine.py`): label, fine-tune, label new images, compare (SAM-paper idea at small scale).
4. **Label these next**: the AI ranks new images by how unsure it is (uncertainty sampling).
5. **AI scorecard** on the stats page: AI labels vs the user's final labels (kept / fixed / removed / drawn by hand, precision, recall), downloadable as JSON.
6. **Pascal VOC export**, the third standard format.
7. **Confidence chip** on the Magic select preview: SAM's own predicted IoU in plain words.

Also new, kept out of git on purpose: `TODO.md` (experiments mapped to each lecture) and `rules.md` (the exam's citation rule and everything that must be cited). The "Remove AI labels" button from yesterday is in commit `b62f0fb`.

None of today's work is committed yet. All AI parts still need the model export and a run on Datalab.

## Exam requirements vs. state

| Requirement (slides) | State |
|---|---|
| Upload and label images | Done |
| At least one model helps inside the app | Done: SAM 2.1, YOLO26n, OWLv2 (find alike + find by name) |
| Annotations saved and exportable | Done: COCO, YOLO, **Pascal VOC** (new), annotated images |
| Model quality (mAP, IoU) | Scripts ready: `training/evaluate.py` (YOLO mAP), **`evaluation/noc_benchmark.py` (SAM NoC/mIoU, new)**, AI scorecard in the app (new). **Not run yet** |
| Effort saved, manual vs assisted, failure cases | Effort meter + scorecard + `training/data_engine.py` (new). User study **not run yet** |
| Runs with one command, eval scripts in repo | `./run.sh` now also exports the OWLv2 text tower. Not verified on Datalab |
| Report, demo video, exported dataset | Not started |

## Done

### Find by name (OWLv2 text queries)

- **What the user sees.** In AI mode, the editor sidebar has a "✦ Find by name" box. Typing "helmet" and pressing Find creates (or reuses) the label "helmet". The AI then searches every image, the open one first. It uses the same progress card, Stop and Undo as find alike. Results are boxes marked AI.
- **Why.** The research notes show text-prompt labeling is what the leading tools ship (SAM 3 in V7 and Roboflow, Grounding DINO / YOLO-World in X-AnyLabeling). It also covers classes YOLO's 80 COCO classes do not know.
- **How.**
  - `models/export_owlv2.py` now also exports the OWLv2 text tower (`owlv2_text.onnx`, input 16 CLIP tokens) and saves the CLIP tokenizer (`owlv2_tokenizer.json`).
  - It checks that the `tokenizers` ids equal the `transformers` ids, and that the ONNX output matches PyTorch.
  - The backend (`OwlTextEncoder` in `backend/app/inference/owl.py`) turns "a photo of a <name>" into a unit query embedding. The image side is unchanged.
  - Score cut-off: 0.10 / 0.20 / 0.30 for loose / normal / strict. Text scores behave more like probabilities than image-query scores (Hugging Face uses 0.1 by default). **Not tuned on real photos yet.**
- **Refactor.** The code that turns OWLv2 boxes into regions now lives in one helper, `_store_found`, shared by find alike and find by name. That code skips duplicates, outlines with SAM when needed, and saves. Behaviour is unchanged: the old tests still pass.
- New endpoint: `POST /api/images/{id}/text-search`. `/api/ai/status` reports `owl_text`. New dependency: `tokenizers` (backend requirements).

### Magic select click benchmark (`evaluation/`)

- **Protocol** (RITM, Sofiiuk et al. 2021):
  - The first click goes to the object centre (the point farthest from its border).
  - Every next click goes to the centre of the biggest error. A missed object region gives an object click; a wrongly covered background region gives a background click.
  - The run stops at 20 clicks.
- **Metrics:** NoC@85, NoC@90, failures, mIoU after 1/3/5 clicks, an IoU-per-click curve (CSV + PNG), and encoder/decoder latency.
- It uses the app's own `SamModel`, so it measures what users get.
- It also counts the vertices of the ground-truth outline as an estimate of how many clicks a manual polygon needs (Polygon-RNN-style). The summary prints "clicks saved" with the caveat that a vertex and a click are not the same effort.
- Datasets: GrabCut (50) and Berkeley (96 images / 100 masks), from the RITM / f-BRS downloads. Other image + mask folders work too. `evaluation/datasets/` is git-ignored.

### Data engine round (`training/data_engine.py`)

The procedure (also in `training/README.md`):
1. Round 0: label batch A in the app with the stock model, in a session named `round0`.
2. Export and fine-tune: `prepare_app_export.py --test 0`, then `train.py --variant full`, then `export_yolo.py`.
3. Round 1: restart with `YOLO_MODEL=yolo_round1.onnx` and label batch B in a session named `round1`.
4. Run `data_engine.py` on the downloaded `events.csv`.

Per round, the report gives:
- the share of final objects the AI provided;
- the accept rate;
- the share of AI labels the user had to fix or delete;
- the number of objects drawn by hand;
- the time per image and per object (pauses over 2 min are cut).

`prepare_app_export.py` now accepts `--test 0`: every image trains, and the next round acts as the test.

Caveat: the two rounds use different images. Otherwise the user would remember them. The report must say this and back it with `evaluate.py` on a fixed test set.

### Label these next (uncertainty sampling)

- Project page, "✦ Label these next" → "Ask the AI". YOLO scores every image that has status *new*. The 6 images it is least sure about are shown, each with a reason: "AI unsure about 2 objects" or "AI sees nothing it knows".
- **Uncertainty** = the mean of 1 − |2s − 1| over the detector's boxes (0 = all sure, 1 = coin flips). An image with no box scores 1: the model sees nothing it knows, so a human look teaches it the most.
- Endpoint: `POST /api/projects/{id}/next-images`.

### AI scorecard

- Stats page, new section, plus a JSON download (`GET /api/projects/{id}/scorecard?download=true`).
- It counts, per label and in total:
  - AI labels kept unchanged;
  - AI labels fixed (edited);
  - AI labels removed (deleted, and not brought back by Undo);
  - labels the user drew by hand.
- Precision = kept ÷ all AI labels the user saw. Recall = AI share of the final objects. Exactly right = kept unchanged ÷ final objects.
- Only images where the AI labeled something are counted, and only the ones marked done when any exist. Manual-mode images do not drag the recall down.
- This is a quality number against the user's own final labels, with no extra ground truth needed.

### Smaller items

- **Pascal VOC export**:
  - The zip holds `JPEGImages/`, `Annotations/*.xml` (1-based pixel boxes; outlines stored as their bounding box), `ImageSets/Main/default.txt` and `labels.txt`.
  - There is a new button on the project page.
- **Confidence chip** in the Magic select hint: "✦ 93% sure", "check the edges" or "unsure, tap the missing part", based on SAM's predicted IoU (≥ 0.85 / ≥ 0.6 / below).
- Project page model list now shows both OWLv2 parts. README, `models/README.md`, `training/README.md` and a new `evaluation/README.md` are updated with commands and citations.

## Tests

- Backend: **21 passed**, up from 15. New tests:
  - find-by-name API with a fake text encoder: finds both balls, no duplicates, wrong name finds nothing, empty text rejected;
  - text threshold;
  - NoC click simulator (first click at the centre, background click when the mask is too big, NoC counting, rectangle = 4 vertices);
  - data-engine round metrics;
  - VOC export;
  - scorecard (a deleted-then-restored label counts as kept);
  - next-images ranking.
- Frontend: `tsc` and `vite build` pass.
- Browser (local backend, no weights): checked that the project page shows the VOC button, both OWLv2 badges and "Label these next", and that the stats page shows the scorecard with correct numbers (2 kept, 1 drawn, 100 % precision, 67 % recall). The find-by-name box only appears when the OWLv2 text weights are present, so it was not seen in the browser yet.

## Not yet verified

- `models/export_owlv2.py` text-tower export and the tokenizer check (needs Datalab).
- Find by name on real photos, and its threshold (0.2).
- The NoC benchmark with real SAM weights. The GrabCut/Berkeley download links come from the RITM / f-BRS release and were not checked today.
- A data-engine round: needs about 2 × 30 labeled images and one training run.
- "Label these next" with the real YOLO model.

## Risks and limits

- Text-query quality depends on the name. OWLv2 knows many words, but rare or very specific terms may fail. Show failure cases in the report.
- Scorecard precision and recall are measured against the user's own final labels, not independent ground truth. Users who accept AI labels without looking inflate them. Pair them with `training/evaluate.py` on a ground-truth test set.
- The data-engine comparison uses different image batches per round.
- 4 days left. Today added features. From now on the priority is running the experiments and writing the report.

## Next

1. Datalab: export all weights (incl. the OWLv2 text tower), then run `training/run_all.sh` and `evaluation/noc_benchmark.py` on GrabCut + Berkeley.
2. Try find by name and find alike on `samples/` and tune the thresholds.
3. One data-engine round (`round0` / `round1`).
4. User study (manual vs AI), then the report, demo video and exported dataset.
