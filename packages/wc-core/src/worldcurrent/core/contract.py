"""Base class and field types for data contracts.

Rules:
- Data that crosses a boundary (process, file, version, plugin) is a frozen pydantic model with
  `extra="forbid"`, so typos fail loudly and every contract has a JSON Schema.
- Runtime objects that hold callables or live handles (EpisodeContext, SafetyContext) stay plain
  dataclasses; they are never serialized.
- numpy arrays use `NDArray`; free-form payloads that may contain arrays use `JsonAny`. Both
  round-trip through JSON via core.codec, so `model_dump(mode="json")` / `model_validate` is the
  whole wire format.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator, WithJsonSchema

from . import codec


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, dict) and value.get("__nd__"):
        return codec.decode_array(value)
    arr = np.asarray(value)
    if arr.dtype.kind not in codec.ALLOWED_KINDS:
        raise ValueError(f"array dtype {arr.dtype} is not allowed (bool/int/uint/float only)")
    return arr


NDArray = Annotated[
    np.ndarray,
    PlainValidator(_to_array),
    PlainSerializer(codec.encode_array, when_used="json"),
    WithJsonSchema({"type": "object", "description": "numpy array {__nd__, dtype, shape, b64}",
                    "required": ["__nd__", "dtype", "shape", "b64"]}),
]

JsonAny = Annotated[
    Any,
    PlainValidator(codec.decode),
    PlainSerializer(codec.encode, when_used="json"),
    WithJsonSchema({}),
]
