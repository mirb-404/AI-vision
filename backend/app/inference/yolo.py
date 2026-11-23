"""YOLO object detector (Ultralytics ONNX export) for box pre-labels."""

import ast
import time
from pathlib import Path

import cv2
import numpy as np

from .onnx_utils import make_session
from .postprocess import decode_yolo


class YoloModel:
    def __init__(self, path: Path):
        self.session = make_session(path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W]; dims may be symbolic
        h, w = shape[2], shape[3]
        self.input_hw = (h if isinstance(h, int) else 640, w if isinstance(w, int) else 640)
        self.names = self._read_names()

    def _read_names(self) -> dict[int, str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta.get("names", "{}"))
            return {int(k): str(v) for k, v in dict(names).items()}
        except (ValueError, SyntaxError, TypeError):
            return {}

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        th, tw = self.input_hw
        h, w = rgb.shape[:2]
        r = min(th / h, tw / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (tw - nw) / 2, (th - nh) / 2
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        x0, y0 = int(round(left - 0.1)), int(round(top - 0.1))
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return tensor, r, x0, y0

    def detect(self, rgb: np.ndarray, conf: float, iou: float) -> tuple[list[dict], float]:
        """Returns ([{label, class_id, score, box: {x, y, w, h}}], latency_ms) in image pixels."""
        h, w = rgb.shape[:2]
        tensor, r, left, top = self._letterbox(rgb)
        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        latency = (time.perf_counter() - t0) * 1000
        rows = decode_yolo(output, conf, iou)
        detections = []
        for x1, y1, x2, y2, score, cls in rows:
            x1 = float(np.clip((x1 - left) / r, 0, w))
            x2 = float(np.clip((x2 - left) / r, 0, w))
            y1 = float(np.clip((y1 - top) / r, 0, h))
            y2 = float(np.clip((y2 - top) / r, 0, h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            class_id = int(cls)
            detections.append({
                "label": self.names.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "score": float(score),
                "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            })
        return detections, latency
