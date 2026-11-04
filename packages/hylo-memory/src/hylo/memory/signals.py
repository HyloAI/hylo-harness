"""Reference extraction for declared, agent-visible scalar/vector signals.

For controlled CPU tasks and adapters that already provide measured signals. This is NOT a
RGB detector and must not be bound to simulator-only fields on a visual benchmark.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Sequence

import numpy as np

from hylo.core.contract import Contract
from hylo.core.episode_memory import EpisodeScope, MemoryEvent
from hylo.core.types import Observation


class Signal(Contract):
    kind: str
    subject: str
    value_key: str
    visible_key: str | None = None
    mode: Literal["change", "rising", "falling"] = "change"


class SignalExtractor:
    def __init__(self, signals: Sequence[Signal]) -> None:
        self.signals = tuple(signals)
        identities = [(s.kind, s.subject) for s in signals]
        if len(set(identities)) != len(identities):
            raise ValueError("signal identities must be unique")
        digest = hashlib.sha256(json.dumps([s.model_dump() for s in signals], sort_keys=True).encode()).hexdigest()[:16]
        self.name = f"observable-signals/v1:{digest}"
        self.required_state = tuple(dict.fromkeys(k for s in signals for k in (s.value_key, s.visible_key) if k))
        self._previous = {}
        self._visible = {}

    def reset(self, scope: EpisodeScope) -> None:
        self._previous.clear()
        self._visible.clear()

    def observe(self, observation: Observation, step: int) -> list[MemoryEvent]:
        events = []
        for s in self.signals:
            key = (s.kind, s.subject)
            visible = s.visible_key is None or bool(np.asarray(observation.state[s.visible_key]).item())
            was_visible = self._visible.get(key)
            self._visible[key] = visible
            kind = s.kind
            if not visible:
                if s.mode == "change" or was_visible is False:
                    continue
                # A hidden transition stream cannot support exact counting or negative recall.
                kind, value = "_gap", None
            else:
                raw = np.asarray(observation.state[s.value_key])
                if not np.isfinite(raw).all():
                    raise ValueError("non-finite observed signal")
                value = raw.item() if raw.size == 1 else raw.tolist()
                previous = self._previous.get(key)
                self._previous[key] = value
                if s.mode == "change":
                    if value == previous:
                        continue
                else:
                    if raw.size != 1:
                        raise ValueError("edge signals must be scalar")
                    # An initially high level is not evidence that an edge occurred in view.
                    if previous is None and ((s.mode == "rising" and not bool(value)) or
                                             (s.mode == "falling" and bool(value))):
                        continue
                    if previous is None or was_visible is False:
                        kind = "_gap"
                    elif s.mode == "rising" and not (not bool(previous) and bool(value)):
                        continue
                    elif s.mode == "falling" and not (bool(previous) and not bool(value)):
                        continue
            events.append(MemoryEvent(id=f"{step}:{s.kind}:{s.subject}", step=step, order=len(events),
                                      time=observation.time, kind=kind, subject=s.subject, value=value,
                                      evidence=(f"observation:{step}:state:{s.value_key}",), source=self.name))
        return events
