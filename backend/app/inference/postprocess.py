"""Pure numpy/OpenCV post-processing, independent of any model file."""

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, tolerance: float = 0.0015) -> list[float]:
    """Largest outer contour of a binary mask as a flat [x0, y0, x1, y1, ...] list.

    `tolerance` is the Douglas-Peucker epsilon as a fraction of the contour
    perimeter; it trades vertex count against boundary accuracy.
    Returns [] when the mask has no region with at least 3 vertices.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, tolerance * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return []
    return [float(v) for v in approx.flatten()]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression on xyxy boxes. Returns kept indices."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo(output: np.ndarray, conf: float, iou: float) -> np.ndarray:
    """Decode an Ultralytics ONNX detection output into rows [x1, y1, x2, y2, score, class].

    Handles both layouts:
    * end-to-end (YOLO26 / YOLOv10 style): (1, N, 6) already NMS-free, xyxy.
    * classic (YOLOv8 / YOLO11): (1, 4 + C, A) with cx, cy, w, h + class scores; NMS applied here.
    Coordinates stay in model-input pixels.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[-1] == 6 and arr.shape[0] != 6:
        rows = arr[arr[:, 4] >= conf]
        return rows.astype(np.float32)

    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # -> (A, 4 + C)
    cls_scores = arr[:, 4:]
    cls = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(arr)), cls]
    mask = scores >= conf
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    cx, cy, w, h = arr[mask, :4].T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores, cls = scores[mask], cls[mask]
    # Class-aware NMS: shift boxes of different classes apart so they never overlap.
    offset = cls[:, None].astype(np.float32) * (boxes.max() + 1.0)
    keep = nms(boxes + offset, scores, iou)
    return np.concatenate([boxes[keep], scores[keep, None], cls[keep, None].astype(np.float32)], axis=1)
