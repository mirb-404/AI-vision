# Session context

Read this first in a new session. Last updated: 2026-11-20.

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

## Current state (end of 2026-11-20)

- MVP built. Backend: 9 pytest tests pass. Frontend `npm run build` passes. Browser smoke test passed for manual tools, autosave, effort logging and the stats page.
- Not verified: SAM/YOLO inference with real weights, the Docker image build, the GPU path.
- Plan: `plan.md` (milestones M0–M7). MVP covers M0–M5 apart from those unverified items. M6 (evaluation scripts) is next.

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
| `research_notes/`, `reports/` | Market and model research |

## Commands

```bash
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
