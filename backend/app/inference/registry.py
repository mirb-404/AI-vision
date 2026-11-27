"""Lazy, process-wide model instances.

Models load on first use, so the API starts (and all non-AI features work)
even when weight files are absent, e.g. on a dev machine without models.
"""

import threading

from .. import config
from .owl import OwlModel, OwlTextEncoder
from .sam import SamModel
from .yolo import YoloModel


class ModelUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_sam: SamModel | None = None
_yolo: YoloModel | None = None
_owl: OwlModel | None = None
_owl_text: OwlTextEncoder | None = None
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


def get_owl() -> OwlModel:
    global _owl
    with _lock:
        if _owl is None:
            if not config.OWL_MODEL.is_file():
                raise ModelUnavailable(
                    f"OWLv2 weights not found ({config.OWL_MODEL.name} in {config.MODELS_DIR}); see models/README.md"
                )
            try:
                _owl = OwlModel(config.OWL_MODEL, config.EMBED_CACHE_SIZE)
                _errors.pop("owl", None)
            except Exception as exc:
                _errors["owl"] = str(exc)
                raise ModelUnavailable(f"OWLv2 failed to load: {exc}") from exc
        return _owl


def _owl_text_files_present() -> bool:
    return config.OWL_TEXT_MODEL.is_file() and config.OWL_TOKENIZER.is_file()


def get_owl_text() -> OwlTextEncoder:
    global _owl_text
    with _lock:
        if _owl_text is None:
            if not _owl_text_files_present():
                raise ModelUnavailable(
                    f"OWLv2 text files not found ({config.OWL_TEXT_MODEL.name}, {config.OWL_TOKENIZER.name} in "
                    f"{config.MODELS_DIR}); see models/README.md"
                )
            try:
                _owl_text = OwlTextEncoder(config.OWL_TEXT_MODEL, config.OWL_TOKENIZER)
                _errors.pop("owl_text", None)
            except Exception as exc:  # missing `tokenizers`, bad file...
                _errors["owl_text"] = str(exc)
                raise ModelUnavailable(f"OWLv2 text encoder failed to load: {exc}") from exc
        return _owl_text


def status() -> dict[str, dict]:
    sam_present = _sam_files_present()
    yolo_present = config.YOLO_MODEL.is_file()
    owl_present = config.OWL_MODEL.is_file()
    # Find by name needs the image model too (the text only gives the query).
    text_present = owl_present and _owl_text_files_present()

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
        "owl": {
            "available": owl_present,
            "loaded": _owl is not None,
            "detail": detail("owl", owl_present, _owl is not None, config.OWL_MODEL.name),
        },
        "owl_text": {
            "available": text_present,
            "loaded": _owl_text is not None,
            "detail": detail("owl_text", text_present, _owl_text is not None,
                             f"{config.OWL_MODEL.name} + {config.OWL_TEXT_MODEL.name} + {config.OWL_TOKENIZER.name}"),
        },
    }
