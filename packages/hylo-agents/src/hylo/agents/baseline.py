"""Baseline agents. They implement the Agent protocol directly (no reasoner, no memory)."""

from __future__ import annotations

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.types import ActionChunk, ActionSemantics, Observation


class PControllerAgent(AgentBase):
    """Test double ("testing.p_controller"): proportional control toward state['goal'] in testing.point_reach."""

    def __init__(self, kp: float = 2.0, kd: float = 0.0, horizon: int = 1, execution_steps: int | None = None) -> None:
        self.kp, self.kd, self.horizon, self.execution_steps = kp, kd, horizon, execution_steps

    def spec(self) -> AgentSpec:
        return AgentSpec(name="testing.p_controller", version="0.1", required_state=("pos", "vel", "goal"),
                         action_semantics=ActionSemantics(control_mode="planar_vel", frame="world"),
                         action_dim=2, components={"policy": f"p_controller(kp={self.kp},kd={self.kd})@0.1"})

    def act(self, observation: Observation) -> ActionChunk:
        s = observation.state
        a = np.clip(self.kp * (s["goal"] - s["pos"]) - self.kd * s["vel"], -1.0, 1.0)
        return ActionChunk(actions=np.repeat(a[None], self.horizon, axis=0),
                           execution_steps=self.execution_steps, provenance={"model": "p_controller"})


class RandomAgent(AgentBase):
    def __init__(self, dim: int = 2, seed: int = 0) -> None:
        self.dim, self._rng = dim, np.random.default_rng(seed)

    def spec(self) -> AgentSpec:
        return AgentSpec(name="baseline.random", version="0.1", action_dim=self.dim)

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk.single(self._rng.uniform(-1, 1, self.dim), model="random")


class PlanarIKAgent(AgentBase):
    """Analytic 2-link IK + joint-space P control for mujoco.planar_reach."""

    def __init__(self, kp: float = 0.5, l1: float = 0.3, l2: float = 0.25, max_step: float = 0.05) -> None:
        self.kp, self.l1, self.l2, self.max_step = kp, l1, l2, max_step

    def spec(self) -> AgentSpec:
        return AgentSpec(name="baseline.planar_ik", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"),
                         action_dim=2, components={"policy": f"planar_ik(kp={self.kp})@0.1"})

    def _ik(self, x: float, y: float) -> np.ndarray:
        c2 = np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        return np.array([q1, q2])

    def act(self, observation: Observation) -> ActionChunk:
        q = observation.state["qpos"]
        err = self._ik(*observation.state["target_pos"]) - q
        err = (err + np.pi) % (2 * np.pi) - np.pi
        return ActionChunk.single(np.clip(self.kp * err / self.max_step, -1.0, 1.0), model="planar_ik")


class NearestObjectAgent(AgentBase):
    """No language, no memory: drives the fingertip to the object nearest to it in the *current*
    observation (mujoco.table_reach). Once objects are occluded it has nothing to go on."""

    def __init__(self, kp: float = 0.3) -> None:
        from .composable import PlanarIKGoalPolicy

        self.policy = PlanarIKGoalPolicy(kp=kp)

    def spec(self) -> AgentSpec:
        return AgentSpec(name="baseline.nearest_object", version="0.1",
                         required_state=("qpos", "tip_pos", "obj_pos", "objects_visible"),
                         action_semantics=self.policy.semantics, action_dim=2,
                         components={"policy": self.policy.name})

    def act(self, observation: Observation) -> ActionChunk:
        s = observation.state
        if s["objects_visible"][0] < 0.5:
            return ActionChunk.single(np.zeros(2), model="hold")
        goal = s["obj_pos"][int(np.argmin(np.linalg.norm(s["obj_pos"] - s["tip_pos"], axis=1)))]
        return self.policy.act(observation, goal)
