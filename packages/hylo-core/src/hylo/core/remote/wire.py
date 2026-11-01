"""msgpack-numpy wire format (the one openpi uses), shared by the policy server (hylo-serve) and the openpi client.

msgpack frames in which numpy arrays are maps
{b"__ndarray__": True, b"data": raw bytes, b"dtype": dtype.str, b"shape": shape} (and numpy scalars
{b"__npgeneric__": True, b"data": item, b"dtype": dtype.str}). Written from the format, not copied; like
openpi and core.codec it never falls back to pickle, so object / void dtypes are refused.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

_REFUSED_KINDS = frozenset("OVc")  # object, void, complex: not needed and not safe/portable on the wire


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _REFUSED_KINDS:
        raise TypeError(f"dtype {obj.dtype} not allowed on the wire")
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {b"__ndarray__": True, b"data": arr.tobytes(), b"dtype": arr.dtype.str, b"shape": arr.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _hook(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in _REFUSED_KINDS:
            raise TypeError(f"dtype {dtype} not allowed on the wire")
        return np.frombuffer(obj[b"data"], dtype=dtype).reshape(obj[b"shape"]).copy()
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_hook)
