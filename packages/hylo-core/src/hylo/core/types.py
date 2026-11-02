"""Core data contracts shared by every layer (design/interfaces/environment.md, agent.md).

- Observations and actions are typed, not free-form dicts (vla-eval's Dict[str, Any] ends up with
  inconsistent keys across benchmarks).
- Action semantics live on the action space (inspect-robots ActionSemantics); compatibility is
  checked at bind time and a mismatch is an error.
- `success` is reported separately from `terminated` / `truncated`.
"""

from __future__ import annotations

from functools import cached_property
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .contract import Contract, JsonAny, NDArray
from .robot import RobotSpec

ControlMode = Literal[
    "joint_pos", "joint_delta", "joint_vel",
    "eef_delta_pos", "eef_delta_pose", "eef_abs_pose",
    "planar_vel",
]


class ActionSemantics(Contract):
    control_mode: ControlMode
    frame: str = "base"
    rotation_repr: Literal["axis_angle", "quat_xyzw", "rot6d"] | None = None
    gripper: str | None = None              # vocabulary follows vla-eval DimSpec.format
    dim_labels: tuple[str, ...] = ()
    max_step: tuple[float | None, ...] | None = None


class BoxSpace(Contract):
    shape: tuple[int, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    semantics: ActionSemantics | None = None
    dtype: Literal["float32", "float64"] = "float64"

    @model_validator(mode="after")
    def _check(self) -> "BoxSpace":
        n = int(np.prod(self.shape))
        if len(self.low) != n or len(self.high) != n:
            raise ValueError("low/high must have one entry per element")
        if any(lo > hi for lo, hi in zip(self.low, self.high)):
            raise ValueError("low must be <= high")
        if self.semantics and self.semantics.dim_labels and len(self.semantics.dim_labels) != n:
            raise ValueError("dim_labels must have one entry per element")
        return self

    @cached_property
    def low_array(self) -> np.ndarray:
        return np.asarray(self.low, dtype=self.dtype).reshape(self.shape)

    @cached_property
    def high_array(self) -> np.ndarray:
        return np.asarray(self.high, dtype=self.dtype).reshape(self.shape)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low_array) and np.all(x <= self.high_array))


class ObservationSpec(Contract):
    state: dict[str, tuple[int, ...]]                         # key -> shape
    cameras: dict[str, tuple[int, int, int]] = Field(default_factory=dict)  # name -> (H, W, C) uint8
    units: dict[str, str] = Field(default_factory=dict)
    has_instruction: bool = True


class EnvCapabilities(Contract):
    seedable: bool = True
    snapshot: bool = False          # snapshot()/restore()
    render: bool = False
    privileged_state: bool = False  # privileged_state(); never exposed to the Agent
    batch_size: int = Field(default=1, ge=1)
    self_paced: bool = False        # env enforces its own wall-clock pacing (real robots)
    real_robot: bool = False


class WorkspaceLimit(Contract):
    """Box the controlled point must stay in. `action_dims[i]` is the action dimension that moves along axis i
    of `state_key` (delta-position control); a move that pushes further outside the box is zeroed."""

    state_key: str
    low: tuple[float, ...]
    high: tuple[float, ...]
    action_dims: tuple[int, ...]


class EnvSpec(Contract):
    name: str
    version: str
    observation: ObservationSpec
    action: BoxSpace
    control_hz: float = Field(gt=0)
    capabilities: EnvCapabilities = Field(default_factory=EnvCapabilities)
    workspace: WorkspaceLimit | None = None     # contributed by the environment, enforced by the SafetyShield
    robot: RobotSpec | None = None              # manipulation environments: arms, hands, cameras (core.robot)


class Observation(Contract):
    state: dict[str, NDArray]
    images: dict[str, NDArray] = Field(default_factory=dict)
    instruction: str | None = None
    time: float = 0.0                      # simulation (or robot) time in seconds
    extra: dict[str, JsonAny] = Field(default_factory=dict)


class Action(Contract):
    data: NDArray
    meta: dict[str, JsonAny] = Field(default_factory=dict)


class ActionChunk(Contract):
    """T actions, of which the executor runs `execution_steps` before replanning
    (LeRobot ChunkPolicySpec: prediction steps != execution steps)."""

    actions: NDArray                       # shape (T, *action_shape)
    execution_steps: int | None = Field(default=None, ge=1)
    provenance: dict[str, JsonAny] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "ActionChunk":
        if self.actions.ndim < 2 or self.actions.shape[0] < 1:
            raise ValueError("ActionChunk.actions must have shape (T>=1, *action_shape)")
        return self

    @classmethod
    def single(cls, action: np.ndarray, **provenance: object) -> "ActionChunk":
        return cls(actions=np.asarray(action, dtype=np.float64)[None], provenance=provenance)


class StepResult(Contract):
    observation: Observation
    reward: float | None
    terminated: bool
    truncated: bool
    success: bool | None = None            # task predicate, evaluated by the Task, not the Agent
    info: dict[str, JsonAny] = Field(default_factory=dict)
