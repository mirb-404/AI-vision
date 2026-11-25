"""Runtime configuration, read from environment variables.

Defaults suit a local run from the repo root; docker-compose overrides them.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "annotator.db"

# Model weights live outside the backend (see models/README.md).
MODELS_DIR = Path(os.getenv("MODELS_DIR", REPO_ROOT / "models" / "weights"))
SAM_ENCODER = MODELS_DIR / os.getenv("SAM_ENCODER", "sam_encoder.onnx")
SAM_DECODER = MODELS_DIR / os.getenv("SAM_DECODER", "sam_decoder.onnx")
# "sam2": SAM 2.x export (square 1024x1024 resize).
# "sam1": SAM v1 / MobileSAM export (longest side 1024, zero padding).
SAM_VARIANT = os.getenv("SAM_VARIANT", "sam2")
YOLO_MODEL = MODELS_DIR / os.getenv("YOLO_MODEL", "yolo.onnx")
# OWLv2 image-guided detection for "find alike objects" (models/export_owlv2.py).
OWL_MODEL = MODELS_DIR / os.getenv("OWL_MODEL", "owlv2.onnx")

# Comma-separated ONNX Runtime providers, tried in order.
ORT_PROVIDERS = [p.strip() for p in os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]

# Number of SAM image embeddings kept in memory.
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "8"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Built frontend (frontend/dist). Served at "/" when it exists, so one port serves UI + API.
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
