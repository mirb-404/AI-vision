# Project Plan — AI-Assisted Image Annotation Platform

Status: draft for review (2026-11-20)
Inputs: course requirements (SRH Data Engineering 1), `reports/AI image annotation landscape.md`, `research_notes/AI image annotation landscape/*`

---

## 1. Direction in one paragraph

A browser app where users upload images, label them with boxes and polygons, and get help from two models:
**SAM 2.1-tiny / MobileSAM** turns clicks or a box into a mask, and a **YOLO detector** pre-labels boxes that the user accepts, edits or deletes.
Every action is logged, so the app measures its own value: time, clicks and corrections per object in *Manual* vs *AI-assisted* mode, plus model quality (IoU, NoC, mAP) against ground truth.
The measured effort meter is the differentiator — commercial tools claim "10x" with no method; this project shows a number with a method.

## 2. Target users and problem

- **Who:** students, researchers and small ML teams who need a few hundred to a few thousand labeled images for a custom detector/segmenter.
- **Problem:** manual polygon labeling is slow (~30+ s per mask); existing tools are either heavy to self-host (CVAT, Label Studio + separate ML backend) or credit-metered (Roboflow).
- **Promise:** zero-config AI assistance, local compute, standard export, and an honest report of how much time the AI actually saved.

## 3. Requirement → feature mapping

| Course requirement | Feature that satisfies it |
|---|---|
| Upload images and label them | Project + class list, image upload (drag & drop), box / polygon / point tools, undo/redo |
| AI model actively helps, integrated in app | Click-to-segment (SAM), detector pre-labels (YOLO) with confidence slider, box → SAM refine chain |
| Annotations saved and exportable | SQLite persistence, export COCO JSON (+ YOLO txt, Pascal VOC optional) as a zip with images |
| Evidence it works | Effort logger + Manual/AI toggle, in-app stats page, offline eval scripts (IoU, NoC@85/90, mAP) |
| Runs with one command | `docker compose up` |
| Reproducible evaluation | `eval/` scripts + pinned datasets download script |

## 4. Feature set (prioritised)

**Must (MVP)**
1. Projects, class labels with colours, image upload and gallery with progress (unlabeled / in progress / done).
2. Annotation canvas: pan/zoom, bounding box, polygon, select/move/delete, vertex edit, undo/redo, keyboard shortcuts.
3. **Smart Select (SAM):** left click = positive point, right click = negative point, or drag a box; mask preview updates per click; Enter accepts and converts to polygon.
4. **Auto-label (YOLO):** one button runs the detector; proposals shown dashed with confidence; slider hides low-confidence ones; user accepts (A), edits, or deletes each proposal.
5. Persistence (SQLite) and export: COCO JSON (bbox + polygon segmentation) and YOLO txt, zipped with images.
6. **Effort logger:** per annotation records mode (manual / AI), time from start to accept, clicks, vertices, and outcome of AI proposals (accepted / edited / deleted / added manually).
7. **Mode toggle** Manual vs AI-assisted per session (needed for the user study).

**Should**
8. Provenance badge per annotation (AI-proposed, AI-edited, manual) and simple history.
9. Stats dashboard: seconds per object (manual vs AI), speed-up ratio, accept/edit/delete rates, model latency.
10. Ground-truth comparison: import a COCO GT file and show IoU of user labels vs GT.

**Stretch (only if time remains)**
11. Text-prompt boxes (Grounding DINO / YOLO-World).
12. One fine-tune round of YOLO on corrected labels; show accept rate rising.

**Out of scope:** video tracking, multi-user review/consensus, auth, SAM 3 (848M params, gated, GPU-only).

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite | Typed, fast dev server, standard |
| Styling | Tailwind CSS (+ a few headless components, e.g. Radix UI) | Fast, consistent UI without a design system build |
| Canvas | Konva via `react-konva` | Shapes, hit testing, drag handles, zoom built in; avoids writing raw canvas math |
| Client state | Zustand | Small store for canvas/tool state; undo/redo via history stack |
| Server state | TanStack Query | Caching, loading states for API calls |
| Backend | **Python FastAPI** (recommended, see §5.1) | Same language as models and eval; serves REST API |
| Inference | ONNX Runtime (CPU, optional CUDA) | One runtime for SAM and YOLO; no PyTorch in the serving image |
| Storage | SQLite (SQLModel) + images on disk volume | Zero setup, file-based, enough for one-user tool |
| Eval | Python: pycocotools, numpy, scipy, pandas, matplotlib | Standard tooling, trusted numbers |
| Packaging | Docker Compose: `web` (nginx serving built React) + `api` (FastAPI) | One command to run |

### 5.1 Backend decision — needs your call

The research notes proposed **Rust (axum + `ort`)**. For a prototype I recommend **Python FastAPI** instead:

- Pros of Python: one language for serving, ONNX export, and evaluation; `ultralytics` / `sam2` reference code is Python; fastest path to a working demo.
- Pros of Rust: single small binary, predictable latency, CV niche advantage (Vitronic, Tensordyne).
- Plan: build in FastAPI now. The React frontend talks plain HTTP/JSON, so the backend can be swapped for axum later without touching the UI.

## 6. Architecture

```
Browser (React + TS + Tailwind + Konva)
   │  REST/JSON  (images as multipart, masks as polygons/RLE)
   ▼
FastAPI  ──►  SQLite (projects, images, labels, annotations, events)
   │       └► /data/images (uploaded files)
   ▼
Model service (in-process)
   ├─ SAM encoder  : runs once per image → embedding cached (LRU, in memory + .npy on disk)
   ├─ SAM decoder  : runs per click (~tens of ms on CPU)
   └─ YOLO detector: runs on demand per image
```

Key design points:
- **Encoder/decoder split:** when an image is opened, the frontend calls `POST /images/{id}/embed` in the background. Clicks then hit only the cheap decoder. This mirrors SAM's own design.
- **Mask → polygon** on the server (OpenCV `findContours` + `approxPolyDP`), so all annotations are stored as polygons/boxes.
- **Events table** is append-only: every tool action is logged with timestamp; stats are computed from it, not from UI state.
- Latency for encoder and decoder is logged per call for the report.

### 6.1 API sketch

| Method | Path | Purpose |
|---|---|---|
| POST/GET | `/projects` | Create / list projects |
| POST/GET | `/projects/{id}/labels` | Class list |
| POST | `/projects/{id}/images` | Upload (multipart, multiple) |
| GET | `/projects/{id}/images` | Gallery with status |
| GET | `/images/{id}/file` | Image bytes |
| GET/PUT | `/images/{id}/annotations` | Load / save annotations |
| POST | `/images/{id}/embed` | Precompute SAM embedding |
| POST | `/images/{id}/segment` | Body: points + labels and/or box → polygon + score |
| POST | `/images/{id}/detect` | Body: conf threshold → box proposals |
| POST | `/events` | Batch of effort-log events |
| GET | `/projects/{id}/stats` | Effort and quality summary |
| GET | `/projects/{id}/export?format=coco|yolo` | Zip download |

### 6.2 Data model

- `Project(id, name, created_at)`
- `Label(id, project_id, name, color)`
- `Image(id, project_id, filename, width, height, status)`
- `Annotation(id, image_id, label_id, type[box|polygon], geometry(json), source[manual|ai|ai_edited], score, created_at, updated_at)`
- `Event(id, session_id, image_id, annotation_id?, mode[manual|ai], kind[click|vertex|accept|edit|delete|start|finish], ts, payload(json))`
- `Session(id, project_id, mode, participant, started_at)`

## 7. Models

| Model | Role | License | Notes |
|---|---|---|---|
| SAM 2.1 hiera_tiny (38.9M) | Click/box → mask, primary | Apache 2.0 | Export encoder + decoder to ONNX |
| MobileSAM (9.66M) | CPU-fast alternative | Apache 2.0 | Fallback if SAM 2.1-tiny too slow on CPU |
| YOLO26n / YOLOv8n (COCO) | Box pre-labels | AGPL-3.0 | Fine for open-source repo; state it in report |
| SAM 2.1 hiera_large | Offline quality comparison only | Apache 2.0 | Eval scripts, not served |

Fine-tuning: none for MVP. Stretch: fine-tune YOLO on the project's own corrected dataset if the chosen domain has non-COCO classes.

**Dataset to annotate for submission** (decide in review): pick a domain with COCO-overlapping classes so YOLO helps (e.g. street scenes, kitchen objects, animals), ~100–200 own or CC-licensed images. A domain where YOLO fails (non-COCO class) is a useful failure case for the report.

## 8. Evaluation plan

**Part A — model quality (offline, scripts in `eval/`)**
- SAM click simulator (RITM protocol): first click at mask interior centre, next clicks at largest error region, up to 20 clicks.
- Metrics: mIoU@1/3/5 clicks, NoC@85, NoC@90, failure count at 20 clicks. Datasets: GrabCut (50), Berkeley (100), DAVIS-345 if time.
- Compare MobileSAM vs SAM 2.1-tiny vs SAM 2.1-large; record encoder/decoder latency on own hardware.
- YOLO: mAP@0.5, mAP@0.5:0.95, precision/recall @0.5 with pycocotools on a 200–500 image COCO val2017 subset **and** on own annotated dataset (unseen domain).

**Part B — effort saved (user study, data from the app's events table)**
- 5–8 participants, ~20 images each, 2–5 objects per image.
- Within-subject: half images manual, half AI-assisted; image sets and order counterbalanced.
- Report: median/IQR seconds per object, speed-up ratio, clicks per object, accept/edit/delete/add rates, final IoU vs GT in both modes (compare speed at matched quality), Wilcoxon signed-rank test, per-class breakdown.
- Compare against anchors: SAM paper 34 s → 14 s per mask; extreme clicking 7 s vs 34.5 s per box.
- Failure cases: small objects, thin structures, low contrast, non-COCO classes.
- Threats to validity: learning effects, small n, builder as participant, hardware-dependent latency.

`eval/study_analysis.py` reads an exported events CSV and produces all tables and plots for the report.

## 9. Repository layout

```
/frontend        React + TS + Tailwind (Vite)
  src/components  Canvas, Toolbar, Gallery, LabelPanel, StatsPage
  src/store       Zustand stores (tool, annotations, history)
  src/api         Typed API client
/backend         FastAPI app
  app/routes      projects, images, annotations, ai, export, stats
  app/models      SQLModel tables
  app/inference   sam.py, yolo.py, onnx session cache
  app/export      coco.py, yolo.py
/models          download_models.sh + ONNX export scripts (weights git-ignored)
/eval            click simulator, mAP script, study analysis, datasets download
/docs            report assets, figures
docker-compose.yml
README.md
```

## 10. Milestones

| # | Milestone | Output | Done when |
|---|---|---|---|
| M0 | **UI prototype** (this next step) | React + TS + Tailwind app with mock API: gallery, canvas, box/polygon tools, label panel, fake AI proposals | You can click through the full flow in the browser and judge the direction |
| M1 | Backend core | FastAPI + SQLite, upload, save/load annotations, COCO/YOLO export | Real labels survive reload and export opens in a COCO viewer |
| M2 | SAM integration | ONNX export, embed + segment endpoints, Smart Select tool | Click on object → polygon in < 1 s after embedding |
| M3 | YOLO integration | Detect endpoint, proposal review UI, confidence slider, box → SAM chain | Proposals can be accepted/edited/deleted |
| M4 | Effort logging | Events, sessions, Manual/AI toggle, stats page | Stats page shows per-object time for both modes |
| M5 | Packaging | Dockerfiles, `docker compose up`, README | Fresh clone runs with one command |
| M6 | Evaluation | Click simulator, mAP script, user study run, analysis notebook | Tables and plots ready for report |
| M7 | Deliverables | Annotated dataset export, demo video, report | All four submission items ready |

### M0 prototype scope (what we build first)

- Vite + React + TS + Tailwind scaffold in `/frontend`.
- Screens: **Projects** → **Gallery** (upload, status) → **Annotate** (canvas, toolbar, labels sidebar, annotation list, mode toggle) → **Stats** (placeholder charts) → **Export** dialog.
- Mock API layer (in-memory, same interface as the real API) so the backend can be plugged in at M1 without UI changes.
- Fake "AI": Smart Select returns a polygon around the click; Auto-label returns random boxes with confidences — only to test the interaction design.
- Images kept in browser memory (object URLs); no persistence yet.

## 11. Risks

| Risk | Mitigation |
|---|---|
| SAM encoder slow on CPU (seconds per image) | Precompute embedding on image open / in background for next image; fall back to MobileSAM |
| ONNX export of SAM 2.1 tricky | Use known community export scripts (cite them) or start with MobileSAM, whose ONNX export is documented |
| YOLO useless on chosen domain | Pick dataset with COCO classes; report non-COCO class as failure case |
| Too few study participants | Plan 5 minimum early; author's own runs reported separately |
| Polygon editing UX takes long | Keep tools minimal; Konva handles hit testing and drag |
| "Forking annotation tools" rule | All UI, data model and export code written here; only inference libraries used and cited |

## 12. Open questions for review

1. Backend: Python FastAPI (recommended) or keep Rust axum from the research notes?
2. Dataset domain for the submitted export?
3. Deadline / available weeks — sets how far into "Should" and "Stretch" we go.
4. Segmentation masks as polygons only, or also store RLE in COCO export?
5. GPU available for development/eval, or CPU only?
