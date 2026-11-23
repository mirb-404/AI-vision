# Project report — 2026-11-20

## Summary

Wrote the project plan and built the MVP of the AI-assisted annotation app. The repo now has three separate parts: `frontend/`, `backend/` and `models/`. No models were downloaded or trained on this machine. That happens on the Linux cloud machine.

## Done

**Planning**
- Wrote `plan.md`. It covers the target users, how each feature maps to a requirement, the stack, the architecture, the API and data model, the models, the evaluation protocol, the milestones and the risks.
- Chose Python FastAPI instead of Rust axum, which the research notes had proposed. Python is the same language as the models and the eval tooling. The frontend talks plain HTTP, so the backend stays swappable.

**Backend (`backend/`, FastAPI + SQLite + SQLModel)**
- Projects, labels, image upload. Uploads are validated, EXIF orientation is baked into the pixels, and the size limit is set by `MAX_UPLOAD_MB`.
- Annotations: boxes and polygons are stored as JSON geometry. A PUT replaces all annotations of an image. Each annotation records its source: `manual`, `ai` or `ai_edited`.
- Inference code in `backend/app/inference/`:
  - `sam.py`: SAM encoder/decoder over ONNX Runtime. Embeddings are cached per image (LRU). Encoder outputs are fed to decoder inputs by name, so the samexporter SAM 2 and SAM1/MobileSAM layouts both work.
  - `yolo.py`: Ultralytics ONNX with letterboxing. Reads class names from the ONNX metadata.
  - `postprocess.py`: mask → polygon (OpenCV), numpy NMS, YOLO decoding for both output layouts (end-to-end and classic).
  - `registry.py`: models load lazily. The API runs without weights and returns 503 with a clear message.
- Effort logging: `LabelingSession` (mode manual/ai, participant) plus an append-only `Event` table. Model latency is logged server-side.
- Stats endpoint: median and IQR of seconds and clicks per object for each mode, the speed-up, proposal accept/edit rates, and latency per model and stage. Also an events CSV export.
- Export: COCO JSON (polygons + bbox + provenance attributes) and YOLO txt with `data.yaml`. Both are zipped with the images.
- `backend/scripts/check_models.py`: smoke test for the weights on Linux.
- Tests: 9 pytest tests (API flow, validation, AI 503 without weights, stats, mask→polygon, NMS, YOLO decoding). **All pass.**

**Frontend (`frontend/`, React 19 + TS + Tailwind v4 + react-konva + zustand)**
- Pages: Projects, Project (labels, upload by drag & drop, gallery, session mode, export, model status), Annotate, Stats.
- Annotate canvas: zoom and pan, box, polygon, select/move/resize, vertex drag, Alt+click to delete a vertex, undo/redo, keyboard shortcuts.
- AI tools: Smart select with SAM (positive/negative points, box prompt, live preview, Enter to accept), YOLO auto-label with a confidence slider and accept/reject per proposal or for all, and "refine box with SAM".
- Autosave, Mark done → next image, Manual/AI mode toggle.
- Effort tracker: time and clicks per object, with proposal accept/reject and edit/delete events.
- `npm run build` passes. Browser smoke test passed: box, polygon, select, vertex drag, undo, autosave, effort events reaching stats, no console errors.

**Models (`models/`)**
- `export_yolo.py`, `export_sam.sh` (samexporter, SAM 2.1 tiny) and `inspect_onnx.py`.
- `README.md` with the I/O contract and license notes. Weights are git-ignored.

**Packaging**
- `docker-compose.yml`: api + web (nginx serves the build and proxies `/api`). `docker-compose.gpu.yml` is the GPU override. Root `README.md` added.

## Not verified / risks

- **SAM/YOLO inference has not run end to end.** No weights exist on this machine. The export flags and I/O names for samexporter come from memory of its docs. Verify them on Linux with `models/inspect_onnx.py` and `backend/scripts/check_models.py`.
- **Docker images not built.** The Docker daemon was not running here. Run `docker compose up --build` on Linux.
- The GPU build (`onnxruntime-gpu`) may need CUDA/cuDNN libraries in the image. Not tested.
- Only the largest connected region of a SAM mask is kept, so occluded objects that split into parts lose those parts.
- In dev mode, React StrictMode creates an extra empty labeling session per page load. The production build does not.

## Next steps

1. On Linux: export the weights, run `check_models.py`, then `docker compose up --build`.
2. Test Smart select and Auto-label on real images. Tune the polygon simplification tolerance.
3. Choose the dataset domain for the submitted export.
4. M6: evaluation scripts (click simulator NoC@85/90, mAP with pycocotools, user-study analysis from `events.csv`).
