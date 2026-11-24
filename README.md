# AI Annotator

A web app for labeling images with AI assistance. Upload images, then draw boxes and polygons. **SAM 2.1** turns a click or a box into a mask, and **YOLO** pre-labels boxes that you accept or reject. Labels export as **COCO JSON** or **YOLO**. Every action is logged, so the app can measure how much time the AI saves compared with manual labeling.

## Repository layout

```
frontend/   React + TypeScript + Tailwind (Vite), canvas via react-konva
backend/    FastAPI + SQLite REST API; ONNX inference code in backend/app/inference/
models/     export scripts + weights/ folder (weights are git-ignored, produced on Linux)
diary/      daily progress reports
plan.md     project plan
```

## Run on Linux without Docker (one command)

```bash
./run.sh            # first run: venvs, model export, Node 22 via nvm if needed, UI build; then serves UI + API on :8000
```

`run.sh` skips every step that is already done, so later runs start in seconds. Other modes:
- `./run.sh dev`: hot-reload dev servers
- `./run.sh setup`: set up only, don't start
- `./run.sh check [img]`: smoke-test the models
- `./run.sh test`: run the backend tests

On a GPU machine it uses `onnxruntime-gpu` automatically. It binds to `127.0.0.1` by default because the app has no login. Open it through a tunnel:
- **JupyterHub:** `https://<hub>/user/<name>/proxy/8000/`. The script prints the exact path. This needs `jupyter-server-proxy`.
- **VS Code / SSH:** forward port 8000.

## Run with Docker (one command)

```bash
docker compose up --build        # http://localhost:8080
```

Without model weights the app still runs: manual labeling and export work, and the AI buttons show "unavailable". To enable AI, export the weights on the Linux machine first. See [models/README.md](models/README.md).

GPU (NVIDIA Container Toolkit needed):
```bash
ORT_PACKAGE=onnxruntime-gpu docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Develop without Docker

```bash
# backend (Python 3.11+)
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt onnxruntime
uvicorn app.main:app --reload --port 8000
pytest                                               # API + post-processing tests, no weights needed

# frontend
cd frontend
npm install
npm run dev                                          # http://localhost:5173, proxies /api to :8000
```

Data (SQLite DB and uploaded images) goes to `./data/`. Change the location with the `DATA_DIR` env var.

## Workflow

1. Create a project and add labels. Use COCO class names, such as `person` or `car`, so YOLO proposals map to your labels automatically. A proposal with no matching label creates one.
2. Upload images by dragging them in or choosing files.
3. Choose **AI-assisted** or **Manual** mode and enter an optional participant id for the effort study.
4. Annotate:
   - **Box (B)**: drag a rectangle.
   - **Polygon (P)**: click to add points. Press Enter or click the first point to close.
   - **Smart (S)**: click an object for SAM. Right-click marks background. Drag to give a box. Press Enter to accept.
   - **Auto-label**: runs YOLO. Use the confidence slider, then accept (A) or reject (Del) each proposal, or all at once.
   - The ✦ button on a box refines it into a SAM mask.
5. Click **Mark done**, which moves you to the next image. Annotations autosave.
6. Export COCO or YOLO zips (images included) from the project page.
7. **Effort stats** shows seconds and clicks per object for each mode, the speed-up, proposal accept and edit rates, and model latency. You can also download the raw event CSV.

## API

FastAPI docs: `http://localhost:8000/docs` (dev) or `http://localhost:8080/api/...` (docker).
Main routes: `/api/projects`, `/api/projects/{id}/images`, `/api/images/{id}/annotations`, `/api/images/{id}/segment`, `/api/images/{id}/detect`, `/api/projects/{id}/export?format=coco|yolo`, `/api/projects/{id}/stats`, `/api/projects/{id}/events.csv`.

## Models and citations

See [models/README.md](models/README.md). In short: SAM 2.1 (Apache-2.0), YOLO by Ultralytics (AGPL-3.0), ONNX Runtime (MIT). The UI, data model, export and effort logging are written from scratch. No existing annotation tool is forked or wrapped.
