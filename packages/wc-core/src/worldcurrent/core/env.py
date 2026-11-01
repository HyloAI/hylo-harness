"""Environment contract. An Environment = Simulator + Task logic behind reset/step.

Optional capabilities are declared in EnvSpec.capabilities and exposed through separate
protocols, never by duck-typing, so the kernel can refuse a run that needs a missing capability.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .task import TaskInstance
from .types import Action, EnvSpec, Observation, StepResult


@runtime_checkable
class Environment(Protocol):
    def spec(self) -> EnvSpec:
        """Must be cheap: no simulator construction (ManiSkill builds obs space via a real reset)."""
        ...

    def reset(self, task: TaskInstance) -> Observation:
        """Initialise the episode for `task` (seed lives on the instance). The realised
        instruction is returned in Observation.instruction."""
        ...

    def step(self, action: Action) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class Snapshottable(Protocol):
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...


@runtime_checkable
class PrivilegedStateSource(Protocol):
    """Ground-truth state for Tasks/Scorers/diagnosis. The kernel never passes this to an Agent."""

    def privileged_state(self) -> dict[str, Any]: ...
