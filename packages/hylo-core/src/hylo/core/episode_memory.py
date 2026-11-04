"""Observable episode-memory contracts. No simulator handles, evaluators or model dependencies."""
from __future__ import annotations

from typing import Literal, Protocol, Sequence

from pydantic import Field

from .contract import Contract, JsonAny
from .types import Observation


class EpisodeScope(Contract):
    task_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)


class MemoryEvent(Contract):
    """One observed event, not one model guess or one repeated image.

    Producers assign stable IDs and a total (step, order) ordering. A completed action is an
    observed transition, not an action request. ``_gap`` marks missing observation coverage.
    Evidence references identify agent-visible frames/receipts retained in the trace.
    """
    id: str = Field(min_length=1)
    step: int = Field(ge=0)
    order: int = Field(default=0, ge=0)
    time: float = Field(ge=0, allow_inf_nan=False)
    kind: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    value: JsonAny = None
    evidence: tuple[str, ...] = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)


class RecallQuery(Contract):
    name: str = Field(min_length=1)
    op: Literal["latest", "first", "count", "sequence", "nth", "unseen", "pending", "elapsed"]
    kind: str = Field(min_length=1)
    subject: str | None = None
    before: str | None = None  # event IDs, exclusive bounds, not semantic search approximations
    after: str | None = None
    as_of_step: int | None = Field(default=None, ge=0)
    at_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    index: int = Field(default=0, ge=0)
    universe: tuple[str, ...] = ()
    complete: bool = False    # explicit closed-world/observation-coverage claim from the adapter
    min_confidence: float = Field(default=0.8, ge=0, le=1)
    limit: int = Field(default=64, ge=1)


class RecallResult(Contract):
    name: str
    status: Literal["known", "partial", "unknown"]
    value: JsonAny = None
    evidence: tuple[str, ...] = ()
    reason: str = ""


class EpisodeMemory(Protocol):
    def reset(self, scope: EpisodeScope, *, resume: bool = False) -> None: ...
    def append(self, events: Sequence[MemoryEvent]) -> int: ...
    def recall(self, query: RecallQuery) -> RecallResult: ...
    def snapshot(self) -> str: ...


class EventExtractor(Protocol):
    name: str
    required_state: tuple[str, ...]

    def reset(self, scope: EpisodeScope) -> None: ...
    def observe(self, observation: Observation, step: int) -> Sequence[MemoryEvent]: ...
