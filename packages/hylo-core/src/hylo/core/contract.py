"""Base class and field types for data contracts.

Rules:
- Data that crosses a boundary (process, file, version, plugin) is a frozen pydantic model with
  `extra="forbid"`, so typos fail loudly and every contract has a JSON Schema.
- Runtime objects that hold callables or live handles (EpisodeContext, SafetyContext) stay plain
  dataclasses; they are never serialized.
- numpy arrays use `NDArray`; free-form payloads that may contain arrays use `JsonAny`. Both
  round-trip through JSON via core.codec, so `model_dump(mode="json")` / `model_validate` is the
  whole wire format.
- Formats written before the rename to Hylo (2026-11-01) carry `wc.<name>/v<x>` schema ids and refer to
  `worldcurrent.*` modules. They are read as `hylo.<name>/v<x>` and `hylo.*` (`current_schema`,
  `current_module`), so recorded runs, artifacts, configs and profiles stay readable; nothing is rewritten on disk.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator, WithJsonSchema, model_validator

from . import codec

LEGACY_SCHEMA_PREFIX, SCHEMA_PREFIX = "wc.", "hylo."
LEGACY_MODULE_PREFIXES = {"worldcurrent.": "hylo.", "wc_examples": "hylo_examples"}


def current_schema(value: Any) -> Any:
    """A schema id as it is named now: "wc.critic/v0.1" -> "hylo.critic/v0.1" (anything else unchanged)."""
    if isinstance(value, str) and value.startswith(LEGACY_SCHEMA_PREFIX) and "/v" in value:
        return SCHEMA_PREFIX + value[len(LEGACY_SCHEMA_PREFIX):]
    return value


def current_module(ref: str) -> str:
    """A module or `module:attr` reference as it is named now: "worldcurrent.agents.x:Y" -> "hylo.agents.x:Y"."""
    for old, new in LEGACY_MODULE_PREFIXES.items():
        if ref.startswith(old):
            return new + ref[len(old):]
    return ref


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _current_schema_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("schema_version", "schema"):
                if key in data and current_schema(data[key]) != data[key]:
                    data = {**data, key: current_schema(data[key])}
        return data


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
