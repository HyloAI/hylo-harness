"""ComposableAgent: Reasoner (slow, event-triggered) + goal-conditioned Policy (every step) + Memory.

This is the reference composition, not a kernel concept. An end-to-end VLA implements Agent
directly; a VLA can also be the `policy` here, with an LLM/VLM as the `reasoner`. Components
declare which observation keys they read, so the same agent works on any environment.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from hylo.core.agent import AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.memory import MemoryStore, WorkingMemory
from hylo.core.registry import registry
from hylo.core.types import ActionChunk, ActionSemantics, Observation, StepResult

from .reasoner import LandmarkReasoner, Reasoner, ReasonerContext


class GoalPolicy(Protocol):
    name: str
    semantics: ActionSemantics
    action_dim: int
    position_key: str                   # observation.state key holding the controlled point (task space)
    required_state: tuple[str, ...]

    def act(self, obs: Observation, subgoal: np.ndarray) -> ActionChunk: ...


class PGoalPolicy:
    """Planar velocity P-controller toward a task-space subgoal (testing.point_reach)."""

    semantics = ActionSemantics(control_mode="planar_vel", frame="world")
    action_dim = 2
    position_key = "pos"
    required_state = ("pos", "vel")

    def __init__(self, kp: float = 3.0, kd: float = 0.0) -> None:
        self.kp, self.kd = kp, kd
        self.name = f"p_goal(kp={kp},kd={kd})@0.1"

    def act(self, obs: Observation, subgoal: np.ndarray) -> ActionChunk:
        s = obs.state
        a = np.clip(self.kp * (subgoal - s["pos"]) - self.kd * s["vel"], -1.0, 1.0)
        return ActionChunk.single(a, model=self.name)


class PlanarIKGoalPolicy:
    """Analytic 2-link IK toward a task-space subgoal, joint-delta output (mujoco planar arms)."""

    semantics = ActionSemantics(control_mode="joint_delta", frame="base")
    action_dim = 2
    position_key = "tip_pos"
    required_state = ("qpos", "tip_pos")

    def __init__(self, kp: float = 0.3, l1: float = 0.3, l2: float = 0.25, max_step: float = 0.05) -> None:
        self.kp, self.l1, self.l2, self.max_step = kp, l1, l2, max_step
        self.name = f"planar_ik_goal(kp={kp})@0.1"

    def ik(self, x: float, y: float) -> np.ndarray:
        c2 = np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        return np.array([q1, q2])

    def act(self, obs: Observation, subgoal: np.ndarray) -> ActionChunk:
        err = self.ik(*subgoal) - obs.state["qpos"]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        return ActionChunk.single(np.clip(self.kp * err / self.max_step, -1.0, 1.0), model=self.name)


class ComposableAgent:
    def __init__(self, policy: GoalPolicy, reasoner: Reasoner, memory: MemoryStore | None = None,
                 replan_every: int = 10, reach_tol: float = 0.06) -> None:
        self.policy, self.reasoner, self.memory = policy, reasoner, memory
        self.replan_every, self.reach_tol = replan_every, reach_tol
        self.working = WorkingMemory()

    def spec(self) -> AgentSpec:
        return AgentSpec(
            name="composable", version="0.2",
            required_state=tuple(dict.fromkeys(self.policy.required_state + self.reasoner.required_state)),
            action_semantics=self.policy.semantics, action_dim=self.policy.action_dim,
            components={"policy": self.policy.name, "reasoner": self.reasoner.name,
                        "memory": type(self.memory).__name__ if self.memory else "none"})

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.rctx = ReasonerContext(task_id=ctx.task.spec.id, episode_id=ctx.episode_id,
                                    working=self.working, memory=self.memory, emit=ctx.emit)
        self.reasoner.begin(self.rctx)
        self.subgoal: np.ndarray | None = None
        self._since = 0
        self._last_visible: bool | None = None
        self._last_emitted: tuple | None = None

    def _trigger(self, obs: Observation) -> str | None:
        visible = bool(obs.state[self.reasoner.visibility_key][0] > 0.5)
        changed = self._last_visible is not None and visible != self._last_visible
        self._last_visible = visible
        if self.subgoal is None:
            return "episode_start"
        if changed:
            return "perception_change"
        if np.linalg.norm(self.subgoal - obs.state[self.policy.position_key]) < self.reach_tol:
            return "subgoal_reached"
        if self._since >= self.replan_every:
            return "periodic"
        return None

    def act(self, observation: Observation) -> ActionChunk:
        trigger = self._trigger(observation)
        if trigger:
            d = self.reasoner.deliberate(observation, trigger, self.rctx)
            out = np.asarray(d.output, dtype=float)
            # Emit when what the agent believes changes: target, reason, or where the belief comes from.
            key = (d.kind, tuple(np.round(out, 6)), d.rationale, d.inputs.get("object_source"))
            if key != self._last_emitted or d.inputs.get("new_choice"):
                self.ctx.emit("agent.decision", {**d.model_dump(), "output": out})
                self._last_emitted = key
            self.subgoal = out
            self._since = 0
        self._since += 1
        return self.policy.act(observation, self.subgoal)  # type: ignore[arg-type]

    def on_step(self, result: StepResult) -> None:
        pass

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        self.reasoner.end(outcome.success, self.rctx)


DEFAULT_MEMORY = {"type": "sqlite", "params": {"path": ":memory:"}}


def make_landmark_agent(kp: float = 3.0, memory: dict | None = DEFAULT_MEMORY, replan_every: int = 10) -> ComposableAgent:
    """Factory for configs. The memory backend is resolved through the plugin registry, so this
    package never imports a memory implementation (memory=None disables long-term memory)."""
    store = registry.build("memory", memory) if memory else None
    return ComposableAgent(PGoalPolicy(kp=kp), LandmarkReasoner(), store, replan_every=replan_every)


def make_object_reach_agent(kp: float = 0.3, chooser: dict | None = None, memory: dict | None = DEFAULT_MEMORY,
                            replan_every: int = 10) -> ComposableAgent:
    """Language-conditioned object reaching: an ObjectReasoner picks the instructed object with a
    pluggable chooser (rule-based or LLM, resolved from the `reasoners` plugin group), remembers
    objects in spatial memory while they are visible, and an IK policy drives the arm to it."""
    from .object_reasoner import ObjectReasoner

    chooser_obj = registry.build("choosers", chooser or {"type": "chooser.rule_color"})
    store = registry.build("memory", memory) if memory else None
    return ComposableAgent(PlanarIKGoalPolicy(kp=kp), ObjectReasoner(chooser_obj), store, replan_every=replan_every)
