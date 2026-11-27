"""YOLO object detectors (Ultralytics ONNX exports) for box pre-labels, alone or merged."""

import ast
import time
from pathlib import Path

import cv2
import numpy as np

from .onnx_utils import make_session
from .postprocess import decode_yolo


class YoloModel:
    def __init__(self, path: Path, tag: str = "yolo"):
        self.tag = tag  # which detector a box came from (shown in the report / stats)
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
                "model": self.tag,
                "score": float(score),
                "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            })
        return detections, latency


GIANT_BOX = 0.8  # a box from the broad model covering more of the image than this is scenery ("Tree")
SAME_OBJECT_IOU = 0.7  # boxes from different models overlapping this much describe the same object


def _iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def merge_detections(per_model: list[list[dict]], image_area: float) -> list[dict]:
    """Merge the boxes of several detectors into one list, most confident first.

    per_model[0] is the broad model (Open Images, 601 classes): it names many more things but also
    draws scenery boxes over the whole image, so its boxes covering > GIANT_BOX of the image are
    dropped. The other models (COCO, 80 classes) fill in objects the broad model misses.
    Boxes from different models that overlap by IoU >= SAME_OBJECT_IOU are the same object under two
    names ("person" / "Man"): the more confident one is kept. Boxes of the same model are already
    de-duplicated by that model's NMS, so they never suppress each other here (a face inside a
    person stays).
    """
    pool = []
    for m, dets in enumerate(per_model):
        for d in dets:
            b = d["box"]
            if m == 0 and len(per_model) > 1 and b["w"] * b["h"] > GIANT_BOX * image_area:
                continue
            pool.append((m, d))
    pool.sort(key=lambda md: -md[1]["score"])
    kept: list[tuple[int, dict]] = []
    for m, d in pool:
        if any(km != m and _iou(d["box"], kd["box"]) >= SAME_OBJECT_IOU for km, kd in kept):
            continue
        kept.append((m, d))
    return [d for _, d in kept]


class MergedDetector:
    """Several YOLO models behind the YoloModel interface (names, detect)."""

    def __init__(self, models: list[YoloModel], tags: list[str]):
        self.models = models
        self.tags = tags
        self.names = {}
        offset = 0
        for model in models:  # class ids made unique across models
            for k, v in model.names.items():
                self.names[offset + k] = v
            offset += max(model.names, default=-1) + 1

    def detect(self, rgb: np.ndarray, conf: float, iou: float) -> tuple[list[dict], float]:
        per_model, total, offset = [], 0.0, 0
        for model, tag in zip(self.models, self.tags):
            dets, ms = model.detect(rgb, conf, iou)
            total += ms
            for d in dets:
                d["class_id"] += offset
                d["model"] = tag
            per_model.append(dets)
            offset += max(model.names, default=-1) + 1
        h, w = rgb.shape[:2]
        return merge_detections(per_model, float(w * h)), total
