"""JSON-safe encoding for numpy arrays and arbitrary payloads. No pickle anywhere (RPent's socket RPC
unpickles arbitrary frames; DSPy's .pkl programs execute code on load)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from pydantic import BaseModel

ALLOWED_KINDS = frozenset("biuf")  # bool, int, uint, float (same whitelist as vla-eval numpy_codec)


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    if arr.dtype.kind not in ALLOWED_KINDS:
        raise TypeError(f"dtype {arr.dtype} not allowed on the wire")
    arr = np.ascontiguousarray(arr)
    return {"__nd__": True, "dtype": arr.dtype.str, "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii")}


def decode_array(obj: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(obj["dtype"])
    if dtype.kind not in ALLOWED_KINDS:
        raise TypeError(f"dtype {dtype} not allowed on the wire")
    return np.frombuffer(base64.b64decode(obj["b64"]), dtype=dtype).reshape(obj["shape"]).copy()


def encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    return obj


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            return decode_array(obj)
        if "__bytes__" in obj and len(obj) == 1:
            return base64.b64decode(obj["__bytes__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj
