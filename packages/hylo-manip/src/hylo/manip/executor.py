"""How the toolkit reaches the robot. The toolkit never touches an environment: an agent gives it an `Executor`
(hylo-agents bridges it to the kernel, so every action still goes through the SafetyShield and into the trace)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from hylo.core.types import Observation


@runtime_checkable
class Executor(Protocol):
    def observation(self) -> Observation:
        """The latest observation (after the last executed step)."""
        ...

    def run(self, actions: np.ndarray) -> dict[str, Any]:
        """Execute (T, action_dim) joint-position targets, one per control step, and block until done or the episode
        ended. Returns {"steps": executed, "ended": bool, "success": bool | None, "reason": str | None}."""
        ...

    def status(self) -> dict[str, Any]:
        """{"step": int, "max_steps": int, "instruction": str | None, "ended": bool}."""
        ...
