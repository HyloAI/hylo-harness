"""Bind-time compatibility check between an Agent and an Environment.

Unlike vla-eval (mismatch -> logger.warning), errors here stop the run. Severity split follows
inspect-robots compat.py: control mode / rotation / dimensionality are errors; frame and gripper
conventions are warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import AgentSpec
from .errors import IncompatibleComponents
from .types import EnvSpec


@dataclass
class CompatReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_compatibility(agent: AgentSpec, env: EnvSpec) -> CompatReport:
    r = CompatReport()
    missing = [k for k in agent.required_state if k not in env.observation.state]
    if missing:
        r.errors.append(f"env does not provide state keys {missing}")
    missing_cams = [c for c in agent.required_cameras if c not in env.observation.cameras]
    if missing_cams:
        r.errors.append(f"env does not provide cameras {missing_cams}")
    n = 1
    for d in env.action.shape:
        n *= d
    if agent.action_dim is not None and agent.action_dim != n:
        r.errors.append(f"action dim: agent {agent.action_dim} != env {n}")
    a, e = agent.action_semantics, env.action.semantics
    if a and e:
        if a.control_mode != e.control_mode:
            r.errors.append(f"control_mode: agent {a.control_mode} != env {e.control_mode}")
        if a.rotation_repr != e.rotation_repr:
            r.errors.append(f"rotation_repr: agent {a.rotation_repr} != env {e.rotation_repr}")
        if a.frame != e.frame:
            r.warnings.append(f"frame: agent {a.frame} != env {e.frame}")
        if a.gripper != e.gripper:
            r.warnings.append(f"gripper: agent {a.gripper} != env {e.gripper}")
    elif e and not a:
        r.warnings.append("agent declares no action semantics; cannot verify control mode")
    return r


def require_compatible(agent: AgentSpec, env: EnvSpec) -> CompatReport:
    report = check_compatibility(agent, env)
    if not report.ok:
        raise IncompatibleComponents("; ".join(report.errors))
    return report
