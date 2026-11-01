"""Typed events and a synchronous event bus.

Every observable thing in an episode is an Event. The Trace is the persisted event stream, so
tracing, live dashboards and online critics all subscribe to the same bus.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic import Field

from .contract import Contract

TRACE_SCHEMA = "wc.trace/v0.1"

# Canonical event types. Components may emit namespaced custom types ("x.<vendor>.<name>").
EPISODE_START = "episode.start"
EPISODE_END = "episode.end"
OBSERVATION = "observation"
AGENT_CHUNK = "agent.chunk"
AGENT_DECISION = "agent.decision"          # reasoner / planner output
MEMORY_RETRIEVAL = "memory.retrieval"
MEMORY_WRITE = "memory.write"
SAFETY_REVIEW = "safety.review"
ACTION = "action"                          # the action actually sent to the env
STEP_RESULT = "step.result"
PRIVILEGED_STATE = "privileged.state"      # scorer/diagnosis only
ERROR = "error"
SCORE = "score"


class Event(Contract):
    type: str
    episode_id: str
    seq: int
    payload: dict[str, Any] = Field(default_factory=dict)   # encoded by the trace writer, not here
    source: str = "kernel"
    sim_time: float | None = None
    wall_time: float = Field(default_factory=time.time)


Subscriber = Callable[[Event], None]


class EventBus:
    """Fan-out with failure isolation: a failing non-critical subscriber is detached, not fatal.
    Critical subscribers (the trace writer) propagate their exceptions."""

    def __init__(self) -> None:
        self._subs: list[tuple[Subscriber, bool]] = []
        self.detached: list[tuple[Subscriber, str]] = []

    def subscribe(self, fn: Subscriber, *, critical: bool = False) -> None:
        self._subs.append((fn, critical))

    def publish(self, event: Event) -> None:
        for fn, critical in list(self._subs):
            try:
                fn(event)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                if critical:
                    raise
                self._subs.remove((fn, critical))
                self.detached.append((fn, repr(exc)))
