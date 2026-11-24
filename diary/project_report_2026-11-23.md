# Project report: 2026-11-23

## Summary

Brought the app up on the Linux cloud machine: a JupyterHub container with no Docker and an RTX A6000. Exported and verified both models on the GPU. Added `run.sh`, a single script that sets up and starts everything. Made the UI work behind the Jupyter proxy.

## Done

**Linux environment (JupyterHub container, `jovyan`)**
- No Docker, Python 3.12.3, Node 18.19 (too old), RTX A6000 with driver 580 / CUDA 13.0.

**Model export (on Linux)**
- YOLO: `yolo26n.pt` exported to `models/weights/yolo.onnx`. Input is `images` [1,3,640,640]; output is `output0` [1,84,8400], the classic layout, so the backend runs NMS.
- SAM 2.1 hiera_tiny exported via samexporter to `sam_encoder.onnx` and `sam_decoder.onnx`.
  - Encoder: input `image` [1,3,1024,1024]; outputs `image_embed`, `high_res_feats_0`, `high_res_feats_1`.
  - Decoder: takes those three feature maps plus `point_coords`, `point_labels`, `mask_input` and `has_mask_input`. It has **no `orig_im_size` input**. It returns 3 masks at 256×256 plus `iou_predictions`.
  - This matches the backend's by-name I/O design. No code change was needed.
- Fix: samexporter needs Meta's `sam2` package, which is not on PyPI. Added `sam2 @ git+https://github.com/facebookresearch/sam2.git` to `models/requirements-export.txt`. Install it with `SAM2_BUILD_CUDA=0`. `export_sam.sh` now checks for it before running.

**Backend on GPU (verified)**
- `onnxruntime-gpu[cuda,cudnn]` 1.32 (CUDA 13 wheels) reports the `CUDAExecutionProvider`. pytest: 9 passed.
- `check_models` on Ultralytics `bus.jpg` (810×1080), first call including CUDA warm-up:
  - SAM: encoder 1413 ms, decoder 130 ms, predicted IoU **0.969**, 81-vertex polygon.
  - YOLO: 1170 ms. Found bus 0.90 and person 0.85 / 0.85 / 0.84 / 0.56.
  - Harmless ORT warnings: MergeShapeInfo, Memcpy nodes, ScatterND.

**`run.sh`: one-file automation**
- Modes: `serve` (default), `dev`, `setup`, `check [img]`, `test`.
- It only reruns a step when that step's inputs have changed (it compares hashes):
  - model export runs only if weights are missing
  - backend virtualenv: picks the CPU or GPU onnxruntime build automatically
  - Node 22 via nvm when Node is older than 20 (user-local, no root)
  - `npm ci`
  - UI build
- Binds to 127.0.0.1 by default, because the app has no auth. Prints the JupyterHub proxy URL from `JUPYTERHUB_SERVICE_PREFIX`.
- `.gitattributes` forces LF line endings for `*.sh`. Scripts are marked executable in git.

**Proxy-friendly UI (single port)**
- The frontend now uses `HashRouter`, relative `api` URLs and Vite `base: './'`, so it works under any path prefix such as `/user/<name>/proxy/8000/`.
- The backend serves `frontend/dist` at `/` when it exists (`FRONTEND_DIST`), so one port serves both UI and API. Docker/nginx is unaffected.
- Verified on Windows: the FastAPI-served build loads at `/#/projects/1/annotate/1` with no console errors. Tests still pass.

## Not verified / risks

- `run.sh` has not run on Linux yet; only a `bash -n` syntax check was done.
- The Jupyter proxy path (`/proxy/8000/`) is untested. It depends on `jupyter-server-proxy` being installed on the hub.
- First-call latencies include CUDA warm-up. Measure steady-state latency later for the report.

## Next steps

1. Commit and push, `git pull` on Linux, run `./run.sh`, and open the UI through the Jupyter proxy.
2. Test Smart select and Auto-label in the browser on real images.
3. Choose the dataset domain.
4. Start M6: evaluation scripts.
