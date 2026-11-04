"""Memory contracts (schema from research/projects/memory-landscape.md §10).

They live in core so that agents (consumers) and memory backends (providers) depend only on core,
never on each other. Time and pose are first-class fields with real range semantics, not values
packed into an embedding; facts about the world are invalidated, never deleted (bi-temporal).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import Field, model_validator

from .contract import Contract, JsonAny


class MemoryKind(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    SKILL = "skill"


class Pose(Contract):
    frame_id: str = Field(min_length=1)
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None = None  # quaternion xyzw

    @model_validator(mode="after")
    def _finite(self) -> "Pose":
        if not all(math.isfinite(v) for v in (*self.position, *(self.orientation or ()))):
            raise ValueError("pose components must be finite")
        return self


class TimeSpan(Contract):
    start: float = Field(allow_inf_nan=False)
    end: float | None = Field(default=None, allow_inf_nan=False)
    clock: Literal["sim", "wall"] = "sim"

    @model_validator(mode="after")
    def _ordered(self) -> "TimeSpan":
        if self.end is not None and self.end < self.start:
            raise ValueError("event interval end precedes start")
        return self


class MemoryItem(Contract):
    kind: MemoryKind
    content: str
    key: str | None = None            # identity of the described thing; a new (kind, key) item
                                      # invalidates the previous one
    payload: dict[str, JsonAny] = Field(default_factory=dict)
    entities: tuple[str, ...] = ()
    scope: dict[str, str] = Field(default_factory=dict)   # task_id / robot_id / episode_id
    pose: Pose | None = None
    event_time: TimeSpan | None = None
    valid_from: float | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = ""                  # producer, e.g. "agent:testing.rule_landmark@0.1"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    def content_hash(self) -> str:
        # Scope, frame, orientation and event time distinguish evidence. Identical-looking
        # observations in different episodes (or repeated actions) are not duplicates.
        blob = json.dumps(self.model_dump(mode="json", exclude={"id", "valid_from"}), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class MemoryQuery(Contract):
    text: str | None = None                       # substring in v0.1; embeddings in v0.2
    kinds: frozenset[MemoryKind] | None = None
    key: str | None = None
    entities: tuple[str, ...] = ()                # match any
    scope: dict[str, str] | None = None           # all must match
    time_range: tuple[float, float] | None = None  # overlap with event_time
    as_of: float | None = None                    # valid time, in the selected clock domain
    clock: Literal["sim", "wall"] | None = None   # never compare sim and wall times implicitly
    near: tuple[Pose, float] | None = None        # (pose, radius); same frame_id required
    include_invalidated: bool = False
    k: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _ranges(self) -> "MemoryQuery":
        if self.time_range is not None:
            lo, hi = self.time_range
            if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
                raise ValueError("time_range must be finite and ordered")
        if self.as_of is not None and not math.isfinite(self.as_of):
            raise ValueError("as_of must be finite")
        if self.near is not None and (not math.isfinite(self.near[1]) or self.near[1] < 0):
            raise ValueError("radius must be finite and nonnegative")
        return self


class Hit(Contract):
    item: MemoryItem
    score: float
    details: dict[str, float] = Field(default_factory=dict)
    valid_to: float | None = None


WriteOp = Literal["ADD", "INVALIDATE", "SKIP_DUP"]


class WriteResult(Contract):
    events: tuple[tuple[WriteOp, str], ...]


@runtime_checkable
class MemoryStore(Protocol):
    def write(self, items: Sequence[MemoryItem]) -> WriteResult: ...
    def query(self, q: MemoryQuery) -> list[Hit]: ...
    def forget(self, q: MemoryQuery) -> int: ...          # soft-expire matching items
    def snapshot(self) -> str: ...                        # content hash, recorded in run manifests


class WorkingMemory:
    """Per-episode key/value scratchpad with a hard capacity (ReMEmbR's working memory is unbounded)."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._d: OrderedDict[str, Any] = OrderedDict()

    def put(self, key: str, value: Any) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def clear(self) -> None:
        self._d.clear()

    def items(self) -> list[tuple[str, Any]]:
        return list(self._d.items())


def now() -> float:
    return time.time()
