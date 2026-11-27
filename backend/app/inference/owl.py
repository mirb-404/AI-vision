"""OWLv2 image-guided detection via ONNX Runtime: mark one object, find every object of that kind.

OWLv2 (Minderer et al., "Scaling Open-Vocabulary Object Detection", NeurIPS 2023) predicts, for
every image patch, a box, an objectness score and a class embedding that lives in the same space
as CLIP text and image-query embeddings. Image-guided ("one-shot") detection, as in the OWL-ViT
paper and the Hugging Face `image_guided_detection`:

1. Source image: pick the predicted box that best overlaps the user's example box; its class
   embedding is the query ("what this kind of object looks like" to the model).
2. Any image: score every patch against the query with the model's class head
   (cosine similarity, plus per-patch logit shift and scale), sigmoid, threshold, NMS.

Because the class embeddings are trained for open-vocabulary *categories*, matches are objects
of the same kind (another ball of another colour), not only near-identical copies.

The ONNX graph (models/export_owlv2.py) returns per-patch outputs; the query selection and the
class head arithmetic run here in numpy, so the graph has no query input.
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .onnx_utils import make_session, output_dict
from .sam import load_rgb

INPUT_SIZE = 960  # owlv2-base-patch16: 60 x 60 patches
MIN_SCORE = 0.05  # never accept matches below this, however low the background is
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """OWLv2 input: pad bottom/right to a grey square, resize to 960, CLIP-normalise. Returns (NCHW, side)."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 0.5, dtype=np.float32)
    canvas[:h, :w] = rgb.astype(np.float32) / 255.0
    interp = cv2.INTER_AREA if side > INPUT_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
    return ((resized - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32), side


def box_iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """IoU of each xyxy box in (N, 4) with one xyxy box."""
    iw = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = iw * ih
    union = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return inter / np.maximum(union, 1e-9)


def select_query(patches: dict[str, np.ndarray], example: np.ndarray) -> np.ndarray:
    """Query embedding for an example box (xyxy pixels).

    Same rule as Hugging Face `Owlv2ForObjectDetection.embed_image_query`: among predicted boxes
    whose IoU with the example is at least 0.8 x the best IoU, take the one whose embedding is
    least like the image's mean embedding (the most "object-like", least background-like).
    """
    ious = box_iou(patches["boxes"], example)
    best = float(ious.max())
    if best <= 0:
        raise ValueError("no model box overlaps the marked object")
    candidates = np.nonzero(ious >= 0.8 * best)[0]
    embeds = patches["class_embeds"]
    mean = embeds.mean(axis=0)
    pick = candidates[int(np.argmin(embeds[candidates] @ mean))]
    q = embeds[pick]
    return q / (np.linalg.norm(q) + 1e-6)


def score_patches(patches: dict[str, np.ndarray], query: np.ndarray) -> np.ndarray:
    """OWLv2 class head for one query: sigmoid((cos + shift) * scale) per patch."""
    logits = (patches["class_embeds"] @ query + patches["logit_shift"]) * patches["logit_scale"]
    return 1.0 / (1.0 + np.exp(-logits))


def auto_threshold(patches: dict[str, np.ndarray], query: np.ndarray, example: np.ndarray,
                   strictness: float) -> float:
    """Score cut-off calibrated on the source image, since image-query scores are not probabilities.

    bg  = 95th percentile of scores of patches whose box does not touch the example (background
          level; a few other instances in the same image barely move this percentile),
    top = score of the example's own best patch.
    The cut-off sits between them: 30 % of the way at strictness 0, 80 % at strictness 1.
    """
    scores = score_patches(patches, query)
    away = box_iou(patches["boxes"], example) < 0.1
    bg = float(np.percentile(scores[away], 95)) if away.any() else 0.0
    top = float(scores[box_iou(patches["boxes"], example) >= 0.5].max(initial=scores.max()))
    frac = 0.3 + 0.5 * float(np.clip(strictness, 0.0, 1.0))
    return max(MIN_SCORE, bg + (top - bg) * frac)


def detect_like(patches: dict[str, np.ndarray], query: np.ndarray, threshold: float,
                nms_iou: float = 0.3, max_dets: int = 100) -> list[tuple[np.ndarray, float]]:
    """Boxes (xyxy pixels) of objects of the query's kind, after threshold and NMS, best first."""
    scores = score_patches(patches, query)
    keep = np.nonzero(scores >= threshold)[0]
    if keep.size == 0:
        return []
    boxes = patches["boxes"][keep]
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    picked = cv2.dnn.NMSBoxes(xywh, scores[keep].astype(float).tolist(), threshold, nms_iou)
    picked = np.asarray(picked).reshape(-1)
    out = [(boxes[i], float(scores[keep][i])) for i in picked]
    out.sort(key=lambda r: -r[1])
    return out[:max_dets]


TEXT_LEN = 16  # OWLv2 text encoder context length (max_position_embeddings)
TEXT_PROMPT = "a photo of a {}"  # CLIP-style prompt, as in the OWL-ViT / Hugging Face examples


def text_threshold(strictness: float) -> float:
    """Score cut-off for text queries. Text-query scores are closer to probabilities than image-query
    scores (Hugging Face post-processing defaults to 0.1), so a fixed range is used:
    0.10 at strictness 0 (loose), 0.20 at 0.5, 0.30 at 1 (strict). Not yet tuned on real data."""
    return 0.10 + 0.20 * float(np.clip(strictness, 0.0, 1.0))


class OwlTextEncoder:
    """OWLv2 text tower via ONNX Runtime: "car" -> unit query embedding (same space as class_embeds).

    Tokenisation uses the CLIP tokenizer saved by models/export_owlv2.py (Hugging Face `tokenizers`).
    """

    def __init__(self, model_path: Path, tokenizer_path: Path):
        from tokenizers import Tokenizer  # imported here: only needed when text search is used

        self.session = make_session(model_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(TEXT_LEN)
        self.tokenizer.enable_padding(length=TEXT_LEN, pad_id=0, pad_token="!")  # OWL pads with "!" (id 0)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def encode(self, text: str) -> tuple[np.ndarray, float]:
        """Unit query embedding for a class name. Returns (embedding (D,), latency_ms; 0 when cached)."""
        key = text.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key], 0.0
        enc = self.tokenizer.encode(TEXT_PROMPT.format(key))
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        t0 = time.perf_counter()
        emb = np.asarray(self.session.run(None, feed)[0], dtype=np.float32).reshape(-1)
        latency = (time.perf_counter() - t0) * 1000
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        with self._lock:
            self._cache[key] = emb
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return emb, latency


class OwlModel:
    def __init__(self, model_path: Path, cache_size: int):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.cache_size = cache_size
        self._cache: OrderedDict[object, tuple[dict[str, np.ndarray], tuple[int, int]]] = OrderedDict()
        self._lock = threading.Lock()

    def embed(self, key: object, image_path: Path) -> tuple[dict[str, np.ndarray], tuple[int, int], float]:
        """Per-patch outputs for an image (cached). Boxes are xyxy in original pixels.

        Returns (patches, (h, w), latency_ms; 0 when cached).
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                patches, size = self._cache[key]
                return patches, size, 0.0
        rgb = load_rgb(image_path)
        pixels, side = preprocess(rgb)
        t0 = time.perf_counter()
        raw = output_dict(self.session, self.session.run(None, {self.input_name: pixels}))
        latency = (time.perf_counter() - t0) * 1000
        cx, cy, bw, bh = np.asarray(raw["pred_boxes"], dtype=np.float32)[0].T  # normalised to the padded square
        patches = {
            "class_embeds": np.asarray(raw["class_embeds"], dtype=np.float32)[0],
            "logit_shift": np.asarray(raw["logit_shift"], dtype=np.float32)[0].reshape(-1),
            "logit_scale": np.asarray(raw["logit_scale"], dtype=np.float32)[0].reshape(-1),
            "objectness": np.asarray(raw["objectness"], dtype=np.float32)[0].reshape(-1),
            "boxes": np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) * side,
        }
        size = rgb.shape[:2]
        with self._lock:
            self._cache[key] = (patches, size)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return patches, size, latency
