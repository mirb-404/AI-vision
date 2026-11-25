#!/usr/bin/env bash
# One-file setup + launcher for Linux (no Docker needed; works in JupyterHub containers).
#
#   ./run.sh              set up whatever is missing, then serve UI + API on one port (default 8000)
#   ./run.sh dev          same setup, then hot-reload dev servers (API :8000, UI :5173)
#   ./run.sh setup        set up only (venvs, model export, npm install, UI build), don't start
#   ./run.sh check [img]  smoke-test the exported models with the backend's inference code
#   ./run.sh test         run the backend test suite
#   ./run.sh datalab      full test on Datalab: setup, unit tests, model smoke test,
#                         end-to-end HTTP check (throwaway data dir), then serve with a
#                         demo dataset + Cloudflare quick tunnel, printing the public URL
#                         (NO_SERVE=1 to stop after the checks, TUNNEL=0 for no tunnel)
#
# Every step is skipped when already done, so re-running is cheap.
#
# Environment overrides:
#   PORT=8000             API / single-port UI
#   HOST=127.0.0.1        bind address. Keep 127.0.0.1 behind Jupyter proxy / SSH tunnel: the app has no login.
#   TUNNEL=1              also open a Cloudflare quick tunnel (public URL, default on for datalab)
#   DEMO=1                load the 8-image coco8 demo project once (default on for datalab)
#   SKIP_MODELS=1         don't export model weights
#   FORCE_CPU=1           use CPU onnxruntime even if a GPU is present
#   YOLO_WEIGHTS=yolo26n.pt   SAM_MODEL=sam2.1_hiera_tiny

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-serve}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
WEIGHTS="$ROOT/models/weights"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
hash_of() { sha256sum | cut -d' ' -f1; }

has_gpu() {
  [ -z "${FORCE_CPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

check_python() {
  command -v python3 >/dev/null || die "python3 not found"
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python 3.11+ needed ($(python3 --version))"
}

# ---------------------------------------------------------------- models
export_models() {
  if [ -n "${SKIP_MODELS:-}" ]; then log "Skipping model export (SKIP_MODELS set)"; return; fi
  if [ -f "$WEIGHTS/yolo.onnx" ] && [ -f "$WEIGHTS/sam_encoder.onnx" ] && [ -f "$WEIGHTS/sam_decoder.onnx" ]      && [ -f "$WEIGHTS/owlv2.onnx" ]; then
    log "Model weights present in models/weights"
    return
  fi
  log "Exporting model weights (one time, downloads ~1-3 GB of export tooling)"
  [ -d .venv-export ] || python3 -m venv .venv-export
  mkdir -p models/checkpoints
  (
    # shellcheck disable=SC1091
    . .venv-export/bin/activate
    pip install -q --upgrade pip
    SAM2_BUILD_CUDA=0 pip install -q -r models/requirements-export.txt
    if [ ! -f "$WEIGHTS/yolo.onnx" ]; then
      # Ultralytics downloads the .pt into the cwd; keep it in models/checkpoints.
      (cd models/checkpoints && python ../export_yolo.py --weights "${YOLO_WEIGHTS:-yolo26n.pt}")
    fi
    if [ ! -f "$WEIGHTS/sam_encoder.onnx" ] || [ ! -f "$WEIGHTS/sam_decoder.onnx" ]; then
      bash models/export_sam.sh "${SAM_MODEL:-sam2.1_hiera_tiny}"
    fi
    if [ ! -f "$WEIGHTS/owlv2.onnx" ]; then
      python models/export_owlv2.py
    fi
  )
}

# ---------------------------------------------------------------- backend
setup_backend() {
  local ort="onnxruntime"
  has_gpu && ort="onnxruntime-gpu[cuda,cudnn]"
  local want
  want="$( (cat backend/requirements.txt backend/requirements-dev.txt; echo "$ort") | hash_of)"
  [ -d backend/.venv ] || python3 -m venv backend/.venv
  if [ "$(cat backend/.venv/.deps-stamp 2>/dev/null)" != "$want" ]; then
    log "Installing backend dependencies ($ort)"
    backend/.venv/bin/pip install -q --upgrade pip
    # CPU and GPU builds conflict; remove both before installing the wanted one.
    backend/.venv/bin/pip uninstall -q -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    backend/.venv/bin/pip install -q -r backend/requirements-dev.txt "$ort"
    echo "$want" > backend/.venv/.deps-stamp
  else
    log "Backend dependencies up to date"
  fi
  if has_gpu; then
    export ORT_PROVIDERS="${ORT_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
  fi
}

# ---------------------------------------------------------------- frontend
node_major() {
  if command -v node >/dev/null 2>&1; then node -p 'process.versions.node.split(".")[0]'; else echo 0; fi
}

ensure_node() {
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ "$(node_major)" -lt 20 ] && [ -s "$NVM_DIR/nvm.sh" ]; then
    set +u; . "$NVM_DIR/nvm.sh"; nvm use 22 >/dev/null 2>&1 || true; set -u
  fi
  [ "$(node_major)" -ge 20 ] && return
  log "Node $(node --version 2>/dev/null || echo 'missing') is too old; installing Node 22 with nvm (user-local, no root)"
  command -v curl >/dev/null || die "curl needed to install nvm; install Node 20+ manually"
  # PROFILE=/dev/null: don't edit ~/.bashrc; this script sources nvm itself.
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | PROFILE=/dev/null bash
  set +u; . "$NVM_DIR/nvm.sh"; nvm install 22; nvm use 22; set -u
  [ "$(node_major)" -ge 20 ] || die "could not get Node 20+"
}

setup_frontend() {
  ensure_node
  local want
  want="$( (cat frontend/package-lock.json; node --version) | hash_of)"
  if [ "$(cat frontend/node_modules/.deps-stamp 2>/dev/null)" != "$want" ]; then
    log "Installing frontend dependencies (npm ci)"
    (cd frontend && npm ci --no-audit --no-fund)
    echo "$want" > frontend/node_modules/.deps-stamp
  else
    log "Frontend dependencies up to date"
  fi
}

build_frontend() {
  local want
  want="$( (find frontend/src frontend/index.html frontend/vite.config.ts frontend/package-lock.json -type f -print0 \
            | sort -z | xargs -0 cat) | hash_of)"
  if [ -f frontend/dist/index.html ] && [ "$(cat frontend/dist/.build-stamp 2>/dev/null)" = "$want" ]; then
    log "UI build up to date"
    return
  fi
  log "Building UI"
  (cd frontend && npm run build)
  echo "$want" > frontend/dist/.build-stamp
}

# ---------------------------------------------------------------- run
print_urls() {
  local port="$1"
  echo
  echo "  Local:       http://localhost:${port}/"
  if [ -n "${JUPYTERHUB_SERVICE_PREFIX:-}" ]; then
    echo "  JupyterHub:  https://<your-hub-host>${JUPYTERHUB_SERVICE_PREFIX}proxy/${port}/"
    echo "               (open the same host you use for JupyterLab; needs jupyter-server-proxy, keep the trailing /)"
  fi
  echo "  VS Code / SSH: forward port ${port}, e.g. ssh -L ${port}:localhost:${port} <host>"
  echo
}

# ---------------------------------------------------------------- cloudflare tunnel
# Quick tunnel (no Cloudflare account): gives a random public https://*.trycloudflare.com URL.
ensure_cloudflared() {
  CLOUDFLARED="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"
  [ -x "$CLOUDFLARED" ] && return
  local arch
  case "$(uname -m)" in
    x86_64|amd64) arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) die "no cloudflared build for $(uname -m); install it manually" ;;
  esac
  log "Downloading cloudflared ($arch) to $CLOUDFLARED (user-local, no root)"
  mkdir -p "$(dirname "$CLOUDFLARED")"
  curl -fsSL -o "$CLOUDFLARED.part" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}"
  chmod +x "$CLOUDFLARED.part"
  mv "$CLOUDFLARED.part" "$CLOUDFLARED"
}

start_tunnel() {
  ensure_cloudflared
  local logf url=""
  logf="$(mktemp -t cloudflared.XXXXXX.log)"
  log "Opening Cloudflare quick tunnel to 127.0.0.1:${PORT}"
  "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >"$logf" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 60); do
    url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$logf" | head -1 || true)"
    [ -n "$url" ] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    sleep 1
  done
  if [ -z "$url" ]; then
    printf '\033[1;33mwarning:\033[0m tunnel URL not found; see %s. App still runs locally.\n' "$logf"
    return
  fi
  echo
  printf '  \033[1;32mPublic URL:  %s/\033[0m\n' "$url"
  echo "  WARNING: the app has no login. Anyone with this URL can view and change data."
  echo "           Stop with Ctrl+C when done; the URL changes every run. Log: $logf"
  echo
}

serve() {
  build_frontend
  log "Starting AI Annotator (UI + API) on ${HOST}:${PORT}"
  print_urls "$PORT"
  if [ "${TUNNEL:-0}" != 1 ] && [ "${DEMO:-0}" != 1 ]; then
    cd backend
    exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
  fi
  (cd backend && exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT") &
  API_PID=$!
  TUNNEL_PID=""
  # Globals, not locals: the EXIT trap runs after this function's scope is gone.
  trap 'kill "$API_PID" ${TUNNEL_PID:+"$TUNNEL_PID"} 2>/dev/null || true' EXIT INT TERM
  for _ in $(seq 60); do
    curl -fs "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1 && break
    kill -0 "$API_PID" 2>/dev/null || die "server exited during startup"
    sleep 1
  done
  if [ "${DEMO:-0}" = 1 ]; then
    log "Loading demo dataset (coco8, 8 images)"
    (cd backend && .venv/bin/python -m scripts.load_demo "http://127.0.0.1:${PORT}") \
      || printf '\033[1;33mwarning:\033[0m demo dataset not loaded; app still runs.\n'
  fi
  [ "${TUNNEL:-0}" = 1 ] && start_tunnel
  wait "$API_PID"
}

dev() {
  log "Starting dev servers: API ${HOST}:${PORT} (reload), UI ${HOST}:5173 (hot reload)"
  echo "  Dev mode needs direct port access (VS Code / SSH forwarding of 5173); behind the Jupyter proxy use ./run.sh"
  (cd backend && exec .venv/bin/uvicorn app.main:app --reload --host "$HOST" --port "$PORT") &
  local api_pid=$!
  trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
  print_urls 5173
  cd frontend
  VITE_API_PROXY="http://localhost:${PORT}" npx vite --host "$HOST" --port 5173
}

check_models() {
  local img="${1:-}"
  if [ -z "$img" ]; then
    img=/tmp/annotator-check.jpg
    [ -f "$img" ] || curl -fsSL -o "$img" https://ultralytics.com/images/bus.jpg
  fi
  log "Smoke-testing models on $img"
  (cd backend && .venv/bin/python -m scripts.check_models "$(realpath "$img")")
}

# Full test on Datalab: setup, unit tests, model smoke test, then an end-to-end
# HTTP run against a throwaway server (its own DATA_DIR, so real data is untouched).
# Serves the app afterwards unless NO_SERVE=1.
datalab() {
  local img=/tmp/annotator-check.jpg e2e_port="${E2E_PORT:-8765}" e2e_data no_models=()
  [ -f "$img" ] || curl -fsSL -o "$img" https://ultralytics.com/images/bus.jpg
  [ -n "${SKIP_MODELS:-}" ] && no_models=(--no-models)

  log "1/4 Backend unit tests"
  (cd backend && .venv/bin/python -m pytest -q)

  if [ -n "${SKIP_MODELS:-}" ]; then
    log "2/4 Model smoke test skipped (SKIP_MODELS set)"
  else
    check_models "$img"
  fi

  build_frontend
  log "3/4 End-to-end API + UI check on 127.0.0.1:${e2e_port}"
  e2e_data="$(mktemp -d)"
  (cd backend && DATA_DIR="$e2e_data" exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$e2e_port" \
     --log-level warning) &
  local srv=$!
  trap 'kill "$srv" 2>/dev/null || true; rm -rf "$e2e_data"' EXIT
  for _ in $(seq 60); do
    curl -fs "http://127.0.0.1:${e2e_port}/api/health" >/dev/null 2>&1 && break
    kill -0 "$srv" 2>/dev/null || die "test server exited during startup"
    sleep 1
  done
  (cd backend && .venv/bin/python -m scripts.e2e_check "http://127.0.0.1:${e2e_port}" "$img" "${no_models[@]}")
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  rm -rf "$e2e_data"
  trap - EXIT

  log "4/4 All checks passed"
  [ -n "${NO_SERVE:-}" ] && return
  TUNNEL="${TUNNEL:-1}" DEMO="${DEMO:-1}" serve
}

check_python
case "$MODE" in
  serve) export_models; setup_backend; setup_frontend; serve ;;
  dev)   export_models; setup_backend; setup_frontend; dev ;;
  setup) export_models; setup_backend; setup_frontend; build_frontend; log "Setup complete. Start with ./run.sh" ;;
  check) setup_backend; check_models "${2:-}" ;;
  test)  setup_backend; (cd backend && .venv/bin/python -m pytest -q) ;;
  datalab) export_models; setup_backend; setup_frontend; datalab ;;
  *)     sed -n '2,23p' "$0"; exit 1 ;;
esac
