"""SafetyShield: the single, non-bypassable checkpoint between any Agent and any Environment.

- Built by the kernel from the EnvSpec plus config; Agents never see or construct it.
- Mandatory guards (finite check, bounds clamp) are always installed; config can only add guards
  or tighten limits. RSI candidates cannot target the `safety.*` config scope (evolve/candidate.py).
- Extra guards are plugins ("guards" group): `safety: {guards: [{type: keep_out, params: {...}}]}`. They run
  between the mandatory finite check (first) and bounds clamp (last), so no plugin can let a non-finite or
  out-of-bounds action through. Built in: delta_limit, workspace, norm_limit, keep_out.
- Rotation-aware: per-dimension clamping is refused for quaternion / rot6d dims
  (inspect-robots approver.py makes the same choice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .errors import SafetyAbort
from .types import Action, BoxSpace, EnvSpec


@dataclass
class SafetyContext:
    space: BoxSpace
    clip_tolerance: float = 0.0
    state: Any = None              # latest observation.state (for state-dependent guards)
    last_action: np.ndarray | None = None
    step: int = 0


@dataclass(frozen=True)
class GuardVerdict:
    data: np.ndarray
    modified: bool
    reason: str = ""
    minor: bool = False      # a numerical clip within tolerance, not a safety intervention


class Guard(Protocol):
    name: str

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict: ...


class FiniteGuard:
    name = "finite"

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        if not np.all(np.isfinite(data)):
            raise SafetyAbort(f"non-finite action at step {ctx.step}")
        return GuardVerdict(data, False)


class BoundsGuard:
    name = "bounds"

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        sem = ctx.space.semantics
        if sem and sem.rotation_repr in ("quat_xyzw", "rot6d"):
            if not ctx.space.contains(data):
                raise SafetyAbort("out-of-bounds rotation action cannot be clamped per-dimension")
            return GuardVerdict(data, False)
        clipped = np.clip(data, ctx.space.low_array, ctx.space.high_array)
        modified = not np.array_equal(clipped, data)
        overshoot = float(np.max(np.abs(clipped - data))) if modified else 0.0
        return GuardVerdict(clipped, modified, f"clamped to action bounds (overshoot {overshoot:.3f})" if modified else "",
                            minor=modified and overshoot <= ctx.clip_tolerance)


class WorkspaceGuard:
    """Zeroes delta-position components that would push the controlled point further outside its workspace."""

    name = "workspace"

    def __init__(self, limit: Any) -> None:
        self.limit = limit
        self.low, self.high = np.asarray(limit.low, dtype=float), np.asarray(limit.high, dtype=float)

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        if not ctx.state or self.limit.state_key not in ctx.state:
            return GuardVerdict(data, False)
        pos = np.asarray(ctx.state[self.limit.state_key], dtype=float).reshape(-1)[: len(self.low)]
        out = data.copy()
        flat = out.reshape(-1)
        hit = []
        for axis, dim in enumerate(self.limit.action_dims):
            if (pos[axis] < self.low[axis] and flat[dim] < 0) or (pos[axis] > self.high[axis] and flat[dim] > 0):
                flat[dim] = 0.0
                hit.append(axis)
        return GuardVerdict(out, bool(hit), f"outside workspace on axes {hit}" if hit else "")


class DeltaLimitGuard:
    """Limits per-step change of the action signal (rate limit)."""

    name = "delta_limit"

    def __init__(self, max_delta: float | list[float]) -> None:
        self.max_delta = np.asarray(max_delta, dtype=np.float64)

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        if ctx.last_action is None:
            return GuardVerdict(data, False)
        lo, hi = ctx.last_action - self.max_delta, ctx.last_action + self.max_delta
        limited = np.clip(data, lo, hi)
        modified = not np.array_equal(limited, data)
        return GuardVerdict(limited, modified, "rate-limited" if modified else "")


@dataclass
class SafetyReview:
    proposed: np.ndarray
    approved: np.ndarray
    modified: bool                 # a safety intervention: counted, and the remaining plan is discarded
    reasons: list[str] = field(default_factory=list)
    clipped: bool = False          # changed only by a within-tolerance bounds clip (recorded, not an intervention)

    def to_payload(self) -> dict[str, Any]:
        return {"proposed": self.proposed, "approved": self.approved,
                "modified": self.modified, "clipped": self.clipped, "reasons": self.reasons}


class SafetyShield:
    MANDATORY = (FiniteGuard, BoundsGuard)

    def __init__(self, env_spec: EnvSpec, extra_guards: list[Guard] | None = None,
                 *, abort_on_modification: bool = False, clip_tolerance: float = 0.05) -> None:
        if env_spec.workspace is not None:          # environment-contributed guardrail, always on
            extra_guards = [WorkspaceGuard(env_spec.workspace), *(extra_guards or [])]
        # Policies routinely overshoot normalized bounds slightly (e.g. a gripper at 1.03); clipping that is
        # bookkeeping, not a safety event. On real robots every clip is an intervention.
        tol = 0.0 if env_spec.capabilities.real_robot else clip_tolerance
        self._ctx = SafetyContext(space=env_spec.action, clip_tolerance=tol)
        # Mandatory guards first and last: extra guards cannot push an action back out of bounds.
        self._guards: list[Guard] = [FiniteGuard(), *(extra_guards or []), BoundsGuard()]
        self.abort_on_modification = abort_on_modification or env_spec.capabilities.real_robot
        self.interventions = 0

    @property
    def guard_names(self) -> list[str]:
        return [g.name for g in self._guards]

    def reset(self) -> None:
        self._ctx.last_action = None
        self._ctx.step = 0
        self.interventions = 0

    def observe(self, state: Any) -> None:
        """Latest observation state, for state-dependent guards (called by the kernel every step)."""
        self._ctx.state = state

    def review(self, action: Action) -> tuple[Action, SafetyReview]:
        proposed = np.asarray(action.data, dtype=np.float64).reshape(self._ctx.space.shape)
        data, reasons, modified, clipped = proposed, [], False, False
        for guard in self._guards:
            verdict = guard.review(data, self._ctx)
            if verdict.modified:
                if verdict.minor:
                    clipped = True
                else:
                    modified = True
                reasons.append(f"{guard.name}: {verdict.reason}")
            data = verdict.data
        if modified:
            self.interventions += 1
            if self.abort_on_modification:
                raise SafetyAbort("; ".join(reasons))
        self._ctx.last_action = data
        self._ctx.step += 1
        return Action(data=data, meta=action.meta), SafetyReview(proposed, data, modified, reasons, clipped and not modified)


class NormLimitGuard:
    """Scales the selected action dimensions down so their Euclidean norm stays within `max_norm`
    (e.g. a speed limit on the translational part of an end-effector delta)."""

    name = "norm_limit"

    def __init__(self, max_norm: float, dims: list[int] | None = None) -> None:
        self.max_norm, self.dims = float(max_norm), dims

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        flat = data.reshape(-1)
        idx = np.arange(len(flat)) if self.dims is None else np.asarray(self.dims)
        n = float(np.linalg.norm(flat[idx]))
        if n <= self.max_norm:
            return GuardVerdict(data, False)
        out = data.copy().reshape(-1)
        out[idx] *= self.max_norm / n
        return GuardVerdict(out.reshape(data.shape), True, f"norm {n:.3f} > {self.max_norm}")


class KeepOutGuard:
    """A box the controlled point must not enter (a fixture, a person's side of the table). Delta-position
    components that move the point into the box, or deeper into it, are zeroed."""

    name = "keep_out"

    def __init__(self, state_key: str, low: list[float], high: list[float], action_dims: list[int], margin: float = 0.0) -> None:
        self.key, self.dims = state_key, list(action_dims)
        self.low, self.high = np.asarray(low, float) - margin, np.asarray(high, float) + margin

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        if not ctx.state or self.key not in ctx.state:
            return GuardVerdict(data, False)
        pos = np.asarray(ctx.state[self.key], float).reshape(-1)[: len(self.low)]
        out = data.copy()
        flat = out.reshape(-1)
        nxt = pos + np.array([flat[d] for d in self.dims])
        if not np.all((nxt >= self.low) & (nxt <= self.high)):
            return GuardVerdict(data, False)                       # the next point is outside the box
        hit = []
        for axis, d in enumerate(self.dims):                       # remove the motion that leads into the box
            inside_before = self.low[axis] <= pos[axis] <= self.high[axis]
            toward = (flat[d] > 0 and pos[axis] < self.high[axis]) or (flat[d] < 0 and pos[axis] > self.low[axis])
            if not inside_before or toward:
                flat[d] = 0.0
                hit.append(axis)
        return GuardVerdict(out, bool(hit), f"keep-out zone on axes {hit}" if hit else "")


def workspace_guard(state_key: str, low: list[float], high: list[float], action_dims: list[int]) -> WorkspaceGuard:
    from .types import WorkspaceLimit

    return WorkspaceGuard(WorkspaceLimit(state_key=state_key, low=tuple(low), high=tuple(high), action_dims=tuple(action_dims)))


def build_guards(cfg: dict[str, Any] | None) -> list[Guard]:
    """Extra guards from the safety config: `max_delta` (shorthand) and `guards: [{type, params}]` plugins."""
    from .registry import registry

    cfg = cfg or {}
    guards: list[Guard] = []
    if "max_delta" in cfg:
        guards.append(DeltaLimitGuard(cfg["max_delta"]))
    for spec in cfg.get("guards") or []:
        g = registry.build("guards", spec)
        if not callable(getattr(g, "review", None)) or not getattr(g, "name", None):
            raise TypeError(f"guard {spec.get('type')!r} must have a name and review(data, ctx)")
        guards.append(g)
    return guards
