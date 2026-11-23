"""Lazy, process-wide model instances.

Models load on first use, so the API starts (and all non-AI features work)
even when weight files are absent, e.g. on a dev machine without models.
"""

import threading

from .. import config
from .sam import SamModel
from .yolo import YoloModel


class ModelUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_sam: SamModel | None = None
_yolo: YoloModel | None = None
_errors: dict[str, str] = {}


def _sam_files_present() -> bool:
    return config.SAM_ENCODER.is_file() and config.SAM_DECODER.is_file()


def get_sam() -> SamModel:
    global _sam
    with _lock:
        if _sam is None:
            if not _sam_files_present():
                raise ModelUnavailable(
                    f"SAM weights not found ({config.SAM_ENCODER.name}, {config.SAM_DECODER.name} in "
                    f"{config.MODELS_DIR}); see models/README.md"
                )
            try:
                _sam = SamModel(config.SAM_ENCODER, config.SAM_DECODER, config.SAM_VARIANT, config.EMBED_CACHE_SIZE)
                _errors.pop("sam", None)
            except Exception as exc:  # bad file, missing onnxruntime, wrong variant...
                _errors["sam"] = str(exc)
                raise ModelUnavailable(f"SAM failed to load: {exc}") from exc
        return _sam


def get_yolo() -> YoloModel:
    global _yolo
    with _lock:
        if _yolo is None:
            if not config.YOLO_MODEL.is_file():
                raise ModelUnavailable(
                    f"YOLO weights not found ({config.YOLO_MODEL.name} in {config.MODELS_DIR}); see models/README.md"
                )
            try:
                _yolo = YoloModel(config.YOLO_MODEL)
                _errors.pop("yolo", None)
            except Exception as exc:
                _errors["yolo"] = str(exc)
                raise ModelUnavailable(f"YOLO failed to load: {exc}") from exc
        return _yolo


def status() -> dict[str, dict]:
    sam_present = _sam_files_present()
    yolo_present = config.YOLO_MODEL.is_file()

    def detail(name: str, present: bool, loaded: bool, files: str) -> str:
        if name in _errors:
            return f"load error: {_errors[name]}"
        if loaded:
            return "loaded"
        return "ready (loads on first use)" if present else f"missing {files}"

    return {
        "sam": {
            "available": sam_present,
            "loaded": _sam is not None,
            "detail": detail("sam", sam_present, _sam is not None,
                             f"{config.SAM_ENCODER.name} + {config.SAM_DECODER.name}"),
        },
        "yolo": {
            "available": yolo_present,
            "loaded": _yolo is not None,
            "detail": detail("yolo", yolo_present, _yolo is not None, config.YOLO_MODEL.name),
        },
    }
