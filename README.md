# AI Annotator

A web app for labeling images with AI assistance. Upload images, then draw boxes and polygons. **SAM 2.1** turns a click or a box into a mask, and **YOLO** pre-labels boxes that you accept or reject. **OWLv2** finds every object like one you marked, or every object matching a typed name. Labels export as **COCO JSON**, **YOLO** or **Pascal VOC**. Every action is logged, so the app can measure how much time the AI saves compared with manual labeling.

## Repository layout

```
frontend/   React + TypeScript + Tailwind (Vite), canvas via react-konva
backend/    FastAPI + SQLite REST API; ONNX inference code in backend/app/inference/
models/     export scripts + weights/ folder (weights are git-ignored, produced on Linux)
training/   YOLO fine-tuning experiments + data-engine round report
evaluation/ Magic select (SAM) click benchmark: NoC@85/90, mIoU per click
samples/    sample images for trying the app
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

1. Create a project and add labels. Auto-label merges two detectors: YOLOv8s trained on Open Images V7 (601 classes, e.g. Man, Human face, Helmet, Footwear) and YOLO26n on COCO (80 classes, fills in what the first misses). Matching names map automatically. Once a project has labels, the AI only suggests those. With **Exact outlines** on (default), SAM traces every AI label along the object's edge. Labels it does not know ("rhino", "scratch") are searched by name with OWLv2, so suggestions work for custom labels too.
2. Upload images by dragging them in or choosing files.
3. Choose **AI-assisted** or **Manual** mode and enter an optional participant id for the effort study.
4. Annotate:
   - **Box (B)**: drag a rectangle.
   - **Polygon (P)**: click to add points. Press Enter or click the first point to close.
   - **Smart (S)**: click an object for SAM. Right-click marks background. Drag to give a box. Press Enter to accept.
   - **Auto-label**: runs YOLO. Use the confidence slider, then accept (A) or reject (Del) each proposal, or all at once.
   - The ✦ button on a box refines it into a SAM mask.
   - **⤓ Image** downloads the current image as a JPEG with its annotations drawn on.
   - **⧉ Find alike**: after marking one object, the AI offers to mark every object of that kind in all images.
   - **✦ Find by name**: type a name ("car", "helmet"); the AI marks every one in all images, also classes YOLO does not know.
   - **✦ Remove AI labels** (top right of the image) removes everything the AI added to this image, with Undo.
   - **Attributes**: select an object to mark it *Partly hidden* (occluded) or *Cut off at edge* (truncated) and add a note. The AI pre-fills both (✦) from the object's position and overlaps.
   - **More kinds of things** (AI panel, off by default): OWLv2 also looks for the 1,203 LVIS classes. Richer labels, slower and less sure.
5. Click **Mark done**, which moves you to the next image. Annotations autosave.
6. Optional: **Auto-label all images** on the project page runs YOLO over every image and saves the boxes as AI annotations. By default it skips images that already have annotations.
7. Export from the project page: COCO, YOLO or Pascal VOC zips (images plus labels), or **Annotated images** (JPEGs with the labels drawn on). **Label these next** asks the AI which new images it is least sure about (label those first before fine-tuning).
8. **Effort stats** shows seconds and clicks per object for each mode, the speed-up, proposal accept and edit rates, and model latency. The **AI scorecard** compares the AI's labels with your final labels (kept, fixed, removed, drawn by you; precision and recall). You can download the raw event CSV and the scorecard JSON.

The **?** button on the project and editor pages starts a guided tour that spotlights each control. The tour opens by itself on the first visit; press `?` in the editor to replay it.

## API

FastAPI docs: `http://localhost:8000/docs` (dev) or `http://localhost:8080/api/...` (docker).
Main routes: `/api/projects`, `/api/projects/{id}/images`, `/api/images/{id}/annotations`, `/api/images/{id}/segment`, `/api/images/{id}/detect`, `/api/projects/{id}/export?format=coco|yolo|voc|images`, `/api/images/{id}/similar`, `/api/images/{id}/text-search`, `/api/projects/{id}/next-images`, `/api/projects/{id}/scorecard`, `/api/projects/{id}/auto-label`, `/api/images/{id}/rendered`, `/api/projects/{id}/stats`, `/api/projects/{id}/events.csv`.

## Models and citations

See [models/README.md](models/README.md). In short: SAM 2.1 (Apache-2.0), YOLO by Ultralytics (AGPL-3.0), OWLv2 (Apache-2.0), ONNX Runtime (MIT), Hugging Face tokenizers (Apache-2.0). The UI, data model, export and effort logging are written from scratch. No existing annotation tool is forked or wrapped.
