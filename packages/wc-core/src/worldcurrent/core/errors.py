"""Error taxonomy (semantics follow inspect-robots errors.py and vla-eval EpisodeError.phase).

- AgentError: the agent failed; the episode is marked `error` and not scored.
- EnvironmentFault: the env/sim failed; infra problem, counted as infra_invalid.
- SafetyAbort: a guard refused an action; the episode halts and the event is scored as a safety event.
- IncompatibleComponents: bind-time contract mismatch; the run never starts.
The partial episode record travels on `exc.record` so nothing collected so far is lost.
"""

from __future__ import annotations

from typing import Any


class WorldcurrentError(Exception):
    phase: str = "unknown"

    def __init__(self, message: str, *, record: Any = None) -> None:
        super().__init__(message)
        self.record = record


class AgentError(WorldcurrentError):
    phase = "agent_act"


class AgentTimeout(AgentError):
    phase = "agent_timeout"


class EnvironmentFault(WorldcurrentError):
    phase = "env_step"


class SafetyAbort(WorldcurrentError):
    phase = "safety"


class IncompatibleComponents(WorldcurrentError):
    phase = "bind"


class ProtectedScopeViolation(WorldcurrentError):
    phase = "evolution"
