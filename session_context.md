# Session context

Read this first in a new session. Last updated: 2026-11-23.

## Project

University exam project (SRH Heidelberg, Data Engineering 1): an **AI-assisted image annotation web app**.
Requirements:
- upload and label images
- at least one model helps *inside* the app
- annotations are saved and exportable
- measure model quality and the effort saved (manual vs assisted)
- runs with one command
- includes eval scripts
- no forking or wrapping of existing annotation tools; pretrained models are fine if cited

Deliverables: report, repo + README, demo video, exported annotated dataset.

## Key decisions

- Stack: React 19 + TypeScript + Tailwind v4 + react-konva + zustand (frontend). FastAPI + SQLite/SQLModel + ONNX Runtime (backend). Docker Compose.
- Backend in Python, not Rust as the research notes proposed. The UI uses plain HTTP, so Rust remains possible later.
- Models: SAM 2.1 hiera_tiny for click/box → mask (MobileSAM via `SAM_VARIANT=sam1`). YOLO (yolo26n / yolo11n) for box pre-labels. SAM 3 is rejected: too large, gated, GPU only.
- What sets the project apart: built-in effort measurement (time and clicks per object, accept/edit/reject rates, per mode).
- **Folders stay separate:** `frontend/`, `backend/` (inference code in `backend/app/inference/`), `models/` (export scripts + `weights/`).
- **Do not download or train models on the Windows dev machine.** All weights are produced on the Linux cloud machine. `*.onnx` and `*.pt` are git-ignored.
- Daily progress goes in `diary/project_report_YYYY-MM-DD.md`.

## Current state (end of 2026-11-23)

- MVP built (2026-11-20). Backend: 9 tests pass. Frontend builds. Manual tools were smoke-tested in the browser.
- **Linux machine:** a JupyterHub container (`jovyan`, `~/vault/AI-vision`). No Docker, an RTX A6000 (CUDA 13 driver), Python 3.12, Node 18 (run.sh installs Node 22 via nvm).
- **Models exported and verified on the GPU** with `onnxruntime-gpu` 1.32: SAM 2.1 tiny gave IoU 0.969 on bus.jpg, and YOLO26n found the bus and 4 people. The weights are in `models/weights/` on Linux only.
- `./run.sh` is the one-command launcher. It sets up whatever is missing, then serves UI + API on 127.0.0.1:8000. The UI uses hash routing and relative URLs so it works behind the Jupyter proxy (`/user/<name>/proxy/8000/`).
- Not yet verified: `run.sh` on Linux, access through the Jupyter proxy, Smart select / Auto-label in the browser, the Docker build.
- Next: M6, the evaluation scripts (click simulator NoC@85/90, mAP, user-study analysis).

## Key files

| Path | What |
|---|---|
| `plan.md` | Full project plan |
| `backend/app/main.py` | FastAPI app, router wiring |
| `backend/app/routes/` | projects, images, annotations, ai, events, stats, export |
| `backend/app/inference/` | `sam.py`, `yolo.py`, `postprocess.py`, `registry.py` |
| `backend/app/export/formats.py` | COCO / YOLO export |
| `backend/scripts/check_models.py` | Smoke-test weights (Linux) |
| `frontend/src/pages/AnnotatePage.tsx` | Annotation logic, autosave, keyboard, AI actions |
| `frontend/src/components/AnnotationCanvas.tsx` | Konva canvas and tools |
| `frontend/src/store/annotate.ts` | zustand store, undo/redo |
| `frontend/src/lib/effort.ts` | Effort logger (how time per object is defined) |
| `models/README.md` | Export steps, ONNX I/O contract, licenses |
| `run.sh` | One-file Linux setup + launcher |
| `research_notes/`, `reports/` | Market and model research |

## Commands

```bash
./run.sh                                        # Linux, no Docker: setup + serve on :8000
./run.sh check | test | dev | setup              # other modes
docker compose up --build                       # http://localhost:8080
cd backend && pytest                            # tests
cd backend && uvicorn app.main:app --reload     # dev API :8000
cd frontend && npm run dev                      # dev UI :5173
python models/export_yolo.py --weights yolo26n.pt   # Linux only
bash models/export_sam.sh sam2.1_hiera_tiny         # Linux only
```

## Open questions for the user

1. Dataset domain for the submitted export.
2. Deadline / weeks available.
3. Store RLE masks in addition to polygons?
4. GPU available on the Linux machine?
