"""Segment Anything (SAM 2.1 / MobileSAM) via ONNX Runtime.

The model is split in two ONNX graphs, as in the SAM paper:
* encoder: heavy, run once per image; its outputs are cached.
* decoder: light, run per click / box with the cached features.

Encoder outputs are fed to decoder inputs of the same name, so exports from
`samexporter` (SAM 2.x: image_embed + high_res_feats_0/1) and SAM v1 /
MobileSAM (image_embeddings) both work. See models/README.md.
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from .onnx_utils import build_feed, make_session, output_dict
from .postprocess import mask_to_polygon

INPUT_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path: Path) -> np.ndarray:
    with PILImage.open(path) as img:
        return np.asarray(img.convert("RGB"))


class SamModel:
    def __init__(self, encoder_path: Path, decoder_path: Path, variant: str, cache_size: int):
        if variant not in ("sam1", "sam2"):
            raise ValueError(f"SAM_VARIANT must be 'sam1' or 'sam2', got {variant!r}")
        self.variant = variant
        self.encoder = make_session(encoder_path)
        self.decoder = make_session(decoder_path)
        enc_input = self.encoder.get_inputs()[0]
        self.encoder_input = enc_input.name
        self.encoder_rank = len(enc_input.shape)
        self.cache_size = cache_size
        self._cache: OrderedDict[object, tuple[dict[str, np.ndarray], tuple[int, int]]] = OrderedDict()
        self._lock = threading.Lock()

    # --- geometry -------------------------------------------------------
    def _scale(self, w: int, h: int) -> tuple[float, float, int, int]:
        """(sx, sy, resized_w, resized_h) mapping original pixels to model input."""
        if self.variant == "sam2":
            return INPUT_SIZE / w, INPUT_SIZE / h, INPUT_SIZE, INPUT_SIZE
        s = INPUT_SIZE / max(w, h)
        return s, s, int(round(w * s)), int(round(h * s))

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        _, _, nw, nh = self._scale(w, h)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.encoder_rank == 3:
            # Encoder graph does its own normalisation/padding (HWC, 0-255 input).
            return resized
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        canvas[:nh, :nw] = (resized / 255.0 - MEAN) / STD
        return canvas.transpose(2, 0, 1)[None]

    def _mask_to_original(self, mask: np.ndarray, w: int, h: int) -> np.ndarray:
        if mask.shape == (h, w):
            return mask
        if self.variant == "sam1":
            # Low-res mask covers the padded square: upscale, crop padding, resize.
            _, _, nw, nh = self._scale(w, h)
            mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)[:nh, :nw]
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- inference ------------------------------------------------------
    def embed(self, key: object, image_path: Path) -> tuple[dict[str, np.ndarray], tuple[int, int], float, bool]:
        """Run (or reuse) the encoder. Returns (features, (h, w), latency_ms, cached)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                feats, size = self._cache[key]
                return feats, size, 0.0, True
        rgb = load_rgb(image_path)
        t0 = time.perf_counter()
        outputs = self.encoder.run(None, {self.encoder_input: self._preprocess(rgb)})
        latency = (time.perf_counter() - t0) * 1000
        feats = output_dict(self.encoder, outputs)
        size = rgb.shape[:2]
        with self._lock:
            self._cache[key] = (feats, size)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return feats, size, latency, False

    def segment(
        self,
        key: object,
        image_path: Path,
        points: list[tuple[float, float, int]],
        box: tuple[float, float, float, float] | None,
    ) -> tuple[list[float], float, float, float]:
        """Prompt -> polygon. points: (x, y, label 1=fg/0=bg); box: (x1, y1, x2, y2).

        Returns (polygon, predicted IoU score, decoder_ms, encoder_ms).
        """
        feats, (h, w), embed_ms, _ = self.embed(key, image_path)
        sx, sy, _, _ = self._scale(w, h)

        coords = [[x * sx, y * sy] for x, y, _ in points]
        labels = [label for _, _, label in points]
        if box is not None:
            x1, y1, x2, y2 = box
            coords += [[x1 * sx, y1 * sy], [x2 * sx, y2 * sy]]
            labels += [2, 3]  # SAM convention: box corners
        else:
            coords.append([0.0, 0.0])
            labels.append(-1)  # SAM convention: padding point when there is no box

        values: dict[str, object] = dict(feats)
        values.update(
            point_coords=np.array([coords]),
            point_labels=np.array([labels]),
            mask_input=np.zeros((1, 1, 256, 256)),
            has_mask_input=np.zeros(1),
            orig_im_size=np.array([h, w]),
        )
        # Single-output encoders whose output name differs from the decoder input.
        dec_names = {i.name for i in self.decoder.get_inputs()}
        missing = dec_names - values.keys()
        if len(missing) == 1 and len(feats) == 1:
            values[missing.pop()] = next(iter(feats.values()))

        t0 = time.perf_counter()
        outputs = self.decoder.run(None, build_feed(self.decoder, values))
        decode_ms = (time.perf_counter() - t0) * 1000
        out = output_dict(self.decoder, outputs)
        masks = out.get("masks", outputs[0])
        scores = out.get("iou_predictions", outputs[1] if len(outputs) > 1 else np.ones((1, 1)))

        masks = np.asarray(masks).reshape(-1, *np.asarray(masks).shape[-2:])
        scores = np.asarray(scores).reshape(-1)
        best = int(scores.argmax()) if scores.size == masks.shape[0] else 0
        mask = self._mask_to_original(masks[best].astype(np.float32), w, h)
        polygon = mask_to_polygon(mask > 0)
        score = float(scores[best]) if scores.size else 0.0
        return polygon, score, decode_ms, embed_ms
