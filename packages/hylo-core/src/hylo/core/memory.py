"""Memory contracts (schema from research/projects/memory-landscape.md §10).

They live in core so that agents (consumers) and memory backends (providers) depend only on core,
never on each other. Time and pose are first-class fields with real range semantics, not values
packed into an embedding; facts about the world are invalidated, never deleted (bi-temporal).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import Field

from .contract import Contract, JsonAny


class MemoryKind(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    SKILL = "skill"


class Pose(Contract):
    frame_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None = None  # quaternion xyzw


class TimeSpan(Contract):
    start: float
    end: float | None = None
    clock: Literal["sim", "wall"] = "sim"


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
        blob = json.dumps({"k": self.kind.value, "c": self.content, "key": self.key,
                           "p": self.model_dump(mode="json")["payload"],
                           "pose": self.pose.position if self.pose else None}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class MemoryQuery(Contract):
    text: str | None = None                       # substring in v0.1; embeddings in v0.2
    kinds: frozenset[MemoryKind] | None = None
    key: str | None = None
    entities: tuple[str, ...] = ()                # match any
    scope: dict[str, str] | None = None           # all must match
    time_range: tuple[float, float] | None = None  # overlap with event_time
    as_of: float | None = None                    # only items valid at this (recorded) time
    near: tuple[Pose, float] | None = None        # (pose, radius); same frame_id required
    include_invalidated: bool = False
    k: int = Field(default=10, ge=1)


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
