"""Small helpers shared by the ONNX model wrappers."""

from typing import Any

import numpy as np

from .. import config

# ONNX type strings -> numpy dtypes. Inputs are cast to what the graph declares,
# so exports that differ only in dtype (e.g. int32 vs int64 sizes) both work.
_DTYPES: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def make_session(path):
    import onnxruntime as ort  # imported lazily: the API still runs without it

    available = set(ort.get_available_providers())
    providers = [p for p in config.ORT_PROVIDERS if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def build_feed(session, values: dict[str, Any]) -> dict[str, np.ndarray]:
    """Pick and cast the inputs the session declares. Fails loudly on gaps."""
    feed: dict[str, np.ndarray] = {}
    missing = []
    for inp in session.get_inputs():
        if inp.name not in values:
            missing.append(inp.name)
            continue
        feed[inp.name] = np.asarray(values[inp.name]).astype(_DTYPES.get(inp.type, np.float32))
    if missing:
        raise RuntimeError(
            f"model input(s) {missing} not provided; run models/inspect_onnx.py and check models/README.md"
        )
    return feed


def output_dict(session, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {o.name: arr for o, arr in zip(session.get_outputs(), outputs)}
