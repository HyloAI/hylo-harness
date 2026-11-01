"""Test double: a deterministic 2-D point-mass environment ("testing.point_reach") for unit and conformance tests.

Not an evaluation environment: examples, protocols and campaigns use the real simulators of worldcurrent-bench
(MuJoCo, LIBERO). `wc plugins` lists testing.* plugins only with --all.

Task params (TaskSpec.params):
    goal:               [x, y] or "random" (seeded)
    goal_visible_steps: int | None   goal shown only for the first N steps (working memory)
    sense_radius:       float | None goal shown only when within this radius (exploration / memory)
    push:               {"step": k, "velocity": [vx, vy]}  external perturbation (recovery)
"""

from __future__ import annotations

import json

import numpy as np

from worldcurrent.core.task import TaskInstance, TaskSpec
from worldcurrent.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                          ObservationSpec, StepResult)

ARENA = 1.0
SUCCESS_RADIUS = 0.05


class PointReachEnv:
    def __init__(self, dt: float = 0.05, max_speed: float = 0.5, lag: float = 0.5) -> None:
        self.dt, self.max_speed, self.lag = dt, max_speed, lag
        self._task: TaskInstance | None = None

    def spec(self) -> EnvSpec:
        return EnvSpec(
            name="testing.point_reach", version="0.1",
            observation=ObservationSpec(
                state={"pos": (2,), "vel": (2,), "goal": (2,), "goal_visible": (1,)},
                units={"pos": "m", "vel": "m/s", "goal": "m", "goal_visible": "bool"}),
            action=BoxSpace(shape=(2,), low=(-1.0, -1.0), high=(1.0, 1.0),
                            semantics=ActionSemantics(control_mode="planar_vel", frame="world",
                                                      dim_labels=("vx", "vy"))),
            control_hz=1.0 / self.dt,
            capabilities=EnvCapabilities(seedable=True, snapshot=True, privileged_state=True),
        )

    def reset(self, task: TaskInstance) -> Observation:
        self._task = task
        p = task.spec.params
        self._rng = np.random.default_rng(task.seed)
        self.pos = self._rng.uniform(-0.8 * ARENA, 0.8 * ARENA, size=2)
        goal = p.get("goal", "random")
        self.goal = self._rng.uniform(-0.8 * ARENA, 0.8 * ARENA, size=2) if goal == "random" \
            else np.asarray(goal, dtype=np.float64)
        self.vel = np.zeros(2)
        self.t = 0
        return self._observe()

    def _goal_visible(self) -> bool:
        p = self._task.spec.params  # type: ignore[union-attr]
        n = p.get("goal_visible_steps")
        if n is not None and self.t >= n:
            return False
        r = p.get("sense_radius")
        if r is not None and np.linalg.norm(self.goal - self.pos) > r:
            return False
        return True

    def _observe(self) -> Observation:
        visible = self._goal_visible()
        return Observation(
            state={"pos": self.pos.copy(), "vel": self.vel.copy(),
                   "goal": self.goal.copy() if visible else np.zeros(2),
                   "goal_visible": np.array([1.0 if visible else 0.0])},
            instruction=self._task.spec.instruction,  # type: ignore[union-attr]
            time=self.t * self.dt,
        )

    def step(self, action: Action) -> StepResult:
        cmd = np.asarray(action.data, dtype=np.float64) * self.max_speed
        self.vel = self.vel + self.lag * (cmd - self.vel)
        push = self._task.spec.params.get("push")  # type: ignore[union-attr]
        if push and self.t == push["step"]:
            self.vel = self.vel + np.asarray(push["velocity"], dtype=np.float64)
        self.pos = np.clip(self.pos + self.vel * self.dt, -ARENA, ARENA)
        self.t += 1
        dist = float(np.linalg.norm(self.goal - self.pos))
        success = dist < SUCCESS_RADIUS
        return StepResult(observation=self._observe(), reward=-dist, terminated=success,
                          truncated=False, success=success, info={"distance": dist})

    def privileged_state(self) -> dict:
        return {"pos": self.pos.copy(), "goal": self.goal.copy(),
                "distance": float(np.linalg.norm(self.goal - self.pos))}

    def snapshot(self) -> bytes:
        return json.dumps({"pos": self.pos.tolist(), "vel": self.vel.tolist(),
                           "goal": self.goal.tolist(), "t": self.t}).encode()

    def restore(self, blob: bytes) -> None:
        s = json.loads(blob)
        self.pos, self.vel, self.goal = (np.asarray(s[k]) for k in ("pos", "vel", "goal"))
        self.t = s["t"]

    def close(self) -> None:
        pass


def reach_tasks() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(id="testing/reach_fixed", version="1", instruction="move to the target at (0.5, 0.5)",
                 params={"goal": [0.5, 0.5]}, max_steps=120, tags=("reach",)),
        TaskSpec(id="testing/reach_random", version="1", instruction="move to the target",
                 params={"goal": "random"}, max_steps=120, tags=("reach",)),
        TaskSpec(id="testing/reach_push", version="1", instruction="move to the target despite disturbances",
                 params={"goal": "random", "push": {"step": 15, "velocity": [0.6, -0.6]}},
                 max_steps=120, tags=("reach", "recovery")),
        TaskSpec(id="testing/reach_remembered", version="1",
                 instruction="the target is shown briefly; move to where it was",
                 params={"goal": "random", "goal_visible_steps": 3}, max_steps=120, tags=("memory", "working")),
        TaskSpec(id="testing/revisit_landmark", version="1",
                 instruction="find the landmark and stop on it",
                 params={"goal": [-0.6, 0.4], "sense_radius": 0.25}, max_steps=300,
                 tags=("memory", "spatial", "episodic")),
    )
