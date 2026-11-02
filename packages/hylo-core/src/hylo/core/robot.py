"""Robot description contract (hylo.robot/v0.1): what a manipulation environment tells tool-using agents about its
robot, so the same perception, kinematics and planning tools (hylo-manip) work on every simulator and robot.

An environment that supports tool use puts a `RobotSpec` in `EnvSpec.robot` (so it also travels over the remote
protocols) and follows the observation layout below. Nothing here changes how the kernel runs an episode: actions are
still joint-position targets reviewed by the SafetyShield, one per control step.

Frames and units: metres and radians; the environment's world frame with +z up; poses are 4x4 homogeneous matrices
given as nested lists. Cameras use the OpenCV optical frame (x right, y down, z forward).

The tool centre point (TCP) is defined the same way for every hand, so a grasp means the same on every robot:

```text
  origin  where a held object's centre sits (between the pads of a parallel gripper; among the fingers of a
          dexterous hand, a few centimetres in front of the palm)
  +x      approach: the direction the hand moves to reach into an object
  +y      closing axis: the pads close along it (dexterous: the thumb-to-fingers direction)
  +z      x cross y
```

Observation layout of a manipulation environment (keys of `Observation`):

```text
  state["q.<arm>"]           measured joint positions of that arm, in ArmSpec.joints order
  state["hand.<arm>"]        measured hand state in command units (parallel: one value, 0 closed .. 1 open), not the
                             last command: pads that stop short of closed tell the tools something is held
  state["base"]              [x, y, yaw] of a mobile base (only when RobotSpec.mobile_base)
  images["<cam>"]            RGB, (H, W, 3) uint8
  images["<cam>.depth"]      depth in metres along the optical axis, (H, W) float32; 0 or non-finite: no return
  images["<cam>.seg"]        instance ids, (H, W) int32, only when the environment was configured to give
                             segmentation (a privileged observation: it is part of the measured config)
  extra["cameras"]["<cam>"]  {"K": 3x3, "T_world_cam": 4x4}: intrinsics and the current camera pose
  extra["segments"]          {"<id>": "<label>"} for the instance ids above

```

Action layout: the concatenation, in RobotSpec.arms order, of each arm's joint targets followed by its hand command
(ArmSpec.action_slice / hand_slice say where each sits).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .contract import Contract

ROBOT_SCHEMA = "hylo.robot/v0.1"
Mat = list[list[float]]
IDENTITY: Mat = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class HandSpec(Contract):
    kind: Literal["parallel", "dexterous"]
    tcp_in_ee: Mat = Field(default_factory=lambda: [r[:] for r in IDENTITY])  # the TCP in the arm's ee frame
    open: tuple[float, ...]            # command: fully open (parallel: (1.0,))
    closed: tuple[float, ...]          # command: fully closed (parallel: (0.0,))
    max_width: float = 0.0             # parallel: distance between the pads at command 1.0, metres
    joints: tuple[str, ...] = ()       # finger joints in the model (for forward kinematics of the fingers)
    presets: dict[str, tuple[float, ...]] = Field(default_factory=dict)  # more named commands, e.g. "point"


class ArmSpec(Contract):
    name: str                          # "left", "right", "arm"
    model: str                         # MJCF or URDF file the kinematics tools load (absolute path, or a
    #                                    "pkg://<distribution>/<path>" resource of an installed package)
    joints: tuple[str, ...]            # joint names in the model, in command order
    ee_frame: str                      # body (MJCF) or link (URDF) the hand is attached to
    base_in_world: Mat = Field(default_factory=lambda: [r[:] for r in IDENTITY])  # the model's root, now
    lower: tuple[float, ...]           # joint limits used by IK and planning (may be tighter than the model's)
    upper: tuple[float, ...]
    home: tuple[float, ...]            # a safe posture with the hand out of the cameras' view of the workspace
    hand: HandSpec
    action_slice: tuple[int, int]      # [start, stop) of this arm's joint targets in the action vector
    hand_slice: tuple[int, int]        # [start, stop) of its hand command

    @model_validator(mode="after")
    def _check(self) -> "ArmSpec":
        n = len(self.joints)
        if not (len(self.lower) == len(self.upper) == len(self.home) == n):
            raise ValueError(f"arm {self.name}: lower, upper and home need {n} values (one per joint)")
        if self.action_slice[1] - self.action_slice[0] != n:
            raise ValueError(f"arm {self.name}: action_slice must span {n} joints")
        if self.hand_slice[1] - self.hand_slice[0] != len(self.hand.open):
            raise ValueError(f"arm {self.name}: hand_slice must span the hand command ({len(self.hand.open)})")
        return self


class CameraSpec(Contract):
    name: str
    height: int
    width: int
    depth: bool = True
    mounted_on: str | None = None      # arm name for a wrist camera (its pose moves with the arm)


class RobotSpec(Contract):
    schema_version: Literal["hylo.robot/v0.1"] = ROBOT_SCHEMA
    arms: tuple[ArmSpec, ...]
    cameras: tuple[CameraSpec, ...] = ()
    mobile_base: bool = False
    table_height: float | None = None  # a hint for the tools (they still measure the table from depth)
    step_units: str = "control steps"  # what the episode's max_steps counts
    notes: str = ""                    # free text for agents: workspace extent, conventions, what is hard

    def arm(self, name: str) -> ArmSpec:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"no arm {name!r}; arms: {[a.name for a in self.arms]}")

    @property
    def action_dim(self) -> int:
        return max(max(a.action_slice[1], a.hand_slice[1]) for a in self.arms)


__all__ = ["ArmSpec", "CameraSpec", "HandSpec", "IDENTITY", "ROBOT_SCHEMA", "RobotSpec"]
