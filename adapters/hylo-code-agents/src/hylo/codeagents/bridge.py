"""The toolkit's Executor on top of the kernel (RFC-0006 part 4): a program moves the robot only through the kernel.

Two threads meet here. The kernel thread calls the agent (`act`, `on_step`, `on_episode_end`); the program thread
(a model's turns, or a frozen solve(robot)) calls the Executor (`observation`, `run`, `status`) and `finish`:

    program: robot.run(plan) -> bridge.run(actions) ---- posts a chunk, blocks ------------------+
    kernel:  act(obs) blocks until a chunk is posted, returns it -> SafetyShield -> env.step x T |
             act(obs) again (the queue is empty): the chunk is done -> wakes the program <-------+
    program: finish() -> a chunk marked END_EPISODE -> the kernel ends the episode ("agent_done")

While the program computes (or the model thinks) the kernel is blocked in act(): simulation time does not pass, and
a robot runner holds. A chunk cut short (the shield modified an action and the kernel dropped the rest) wakes the
program with `interrupted: true`. When the episode ends for any reason (task terminated, max_steps, an error, the
operator), on_episode_end wakes the program: every later run() returns at once with `ended: true`. Needs the
kernel's sync timing (realtime mode asks for chunks before the last one finished).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import numpy as np

from hylo.core.agent import END_EPISODE, AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.task import TaskInstance
from hylo.core.types import ActionChunk, ActionSemantics, EnvSpec, Observation, StepResult


class EpisodeBridge:
    """One episode's handoff between the kernel thread and the program thread. Implements hylo.manip.Executor."""

    def __init__(self, env_spec: EnvSpec, task: TaskInstance, *, expose_success: bool = False,
                 act_wait_s: float = 1800.0) -> None:
        self.env_spec, self.task = env_spec, task
        self.action_dim = int(np.prod(env_spec.action.shape))
        self.expose_success, self.act_wait_s = expose_success, act_wait_s
        self.provenance: dict[str, Any] = {}            # set by the program side (turn number, tool)
        self._cv = threading.Condition()
        self._obs: Observation | None = None
        self._step = 0
        self._success: bool | None = None
        self._last: np.ndarray | None = None
        self._request: ActionChunk | None = None
        self._inflight: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._kernel_waiting = False
        self._ended = False
        self._cancelled = False
        self.outcome: dict[str, Any] | None = None
        self.chunks = 0

    # ------------------------------------------------------------------ kernel side
    def act(self, observation: Observation) -> ActionChunk:
        with self._cv:
            self._obs = observation
            if self._inflight is not None:               # asked again: the queue ran empty, the chunk is done
                self._complete(ended=False)
            self._kernel_waiting = True
            self._cv.notify_all()
            deadline = time.monotonic() + self.act_wait_s
            while self._request is None and not self._cancelled:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._cancelled = True
                    break
                self._cv.wait(min(left, 1.0))
            self._kernel_waiting = False
            if self._request is None:                    # cancelled or no action in time: end the episode
                why = "agent_timeout" if not self._ended else "cancelled"
                chunk = self._end_chunk(f"{why}: no action from the program within {self.act_wait_s:.0f} s"
                                        if why == "agent_timeout" else why)
            else:
                chunk, self._request = self._request, None
            if not chunk.provenance.get(END_EPISODE):
                self._inflight = {"start": self._step, "n": int(chunk.actions.shape[0])}
            self.chunks += 1
            return chunk

    def on_step(self, result: StepResult) -> None:
        with self._cv:
            self._step += 1
            self._obs = result.observation
            self._success = result.success

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        with self._cv:
            self._ended = True
            self.outcome = {"status": outcome.status, "success": outcome.success, "steps": outcome.steps,
                            "reason": outcome.termination_reason}
            if self._inflight is not None:
                self._complete(ended=True)
            self._cv.notify_all()

    def cancel(self) -> None:
        """Unblock both sides for good (shutdown): the kernel gets an END_EPISODE chunk, the program `ended`."""
        with self._cv:
            self._cancelled = True
            self._cv.notify_all()

    def _complete(self, *, ended: bool) -> None:
        assert self._inflight is not None
        steps = self._step - self._inflight["start"]
        interrupted = not ended and steps < self._inflight["n"]
        reason = None
        if ended and self.outcome is not None:
            reason = self.outcome["reason"]
        elif interrupted:
            reason = "interrupted: the safety shield modified an action and the rest of the motion was dropped"
        self._result = {"steps": steps, "ended": ended, "interrupted": interrupted,
                        "success": self._success if self.expose_success else None, "reason": reason}
        self._inflight = None
        self._cv.notify_all()

    def _end_chunk(self, reason: str | bool) -> ActionChunk:
        hold = self._last if self._last is not None else np.zeros(self.action_dim)
        return ActionChunk(actions=np.asarray(hold, dtype=float).reshape(1, -1),
                           provenance={**self.provenance, END_EPISODE: reason or True})

    # ------------------------------------------------------------------ program side (Executor)
    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def step(self) -> int:
        return self._step

    def wait_ready(self, timeout_s: float = 60.0) -> bool:
        """Block until the kernel has the first observation (it is waiting in act) or the episode ended."""
        with self._cv:
            return self._cv.wait_for(lambda: self._obs is not None or self._ended or self._cancelled, timeout_s)

    def observation(self) -> Observation:
        with self._cv:
            self._cv.wait_for(lambda: self._obs is not None or self._ended or self._cancelled)
            if self._obs is None:
                raise RuntimeError("the episode ended before its first observation")
            return self._obs

    def status(self) -> dict[str, Any]:
        with self._cv:                                   # the program may start before act() stored the first
            self._cv.wait_for(lambda: self._obs is not None or self._ended or self._cancelled, 60.0)   # observation
            obs = self._obs
        return {"step": self._step, "max_steps": self.task.spec.max_steps,
                "instruction": (obs.instruction if obs is not None else None) or self.task.spec.instruction,
                "ended": self._ended}

    def run(self, actions: np.ndarray) -> dict[str, Any]:
        a = np.asarray(actions, dtype=float)
        if a.ndim == 1:
            a = a[None]
        if a.ndim != 2 or a.shape[1] != self.action_dim or a.shape[0] < 1:
            raise ValueError(f"run() needs (T, {self.action_dim}) joint targets, got shape {tuple(np.shape(actions))}")
        if not np.all(np.isfinite(a)):
            raise ValueError("run() got non-finite joint targets")
        with self._cv:
            if self._ended or self._cancelled:
                return self._ended_result()
            self._result = None
            self._request = ActionChunk(actions=a, provenance={**self.provenance, "tool": "run", "steps": int(len(a))})
            self._last = a[-1].copy()
            self._cv.notify_all()
            self._cv.wait_for(lambda: self._result is not None or self._ended or self._cancelled)
            res = self._result if self._result is not None else self._ended_result()
            self._result = None
            return dict(res)

    def _ended_result(self) -> dict[str, Any]:
        o = self.outcome or {}
        return {"steps": 0, "ended": True, "interrupted": False,
                "success": o.get("success") if self.expose_success else None, "reason": o.get("reason", "cancelled")}

    def finish(self, reason: str = "") -> dict[str, Any]:
        """End the episode (END_EPISODE) and block until the kernel has ended it; returns the outcome record."""
        with self._cv:
            if not self._ended and not self._cancelled:
                self._request = self._end_chunk(reason or True)
                self._cv.notify_all()
            self._cv.wait_for(lambda: self._ended or self._cancelled)
            return dict(self.outcome or {"status": "failure", "success": None, "steps": self._step, "reason": "cancelled"})


class TimedExecutor:
    """The Executor a program's toolkit gets: the bridge, with the sandbox clock paused while the robot moves."""

    def __init__(self, bridge: Any, sandbox: Any) -> None:
        self._bridge, self._sandbox = bridge, sandbox          # private: programs reach the robot only via run()

    def observation(self) -> Observation:
        return self._bridge.observation()

    def status(self) -> dict[str, Any]:
        return self._bridge.status()

    def run(self, actions: np.ndarray) -> dict[str, Any]:
        with self._sandbox.paused():
            return self._bridge.run(actions)


class LiveExecutor:
    """An Executor that forwards to whichever episode is live (research sessions keep one toolkit across resets)."""

    def __init__(self) -> None:
        self.bridge: EpisodeBridge | None = None

    def _b(self) -> EpisodeBridge:
        if self.bridge is None:
            raise RuntimeError("no live episode: call reset()")
        return self.bridge

    def observation(self) -> Observation:
        return self._b().observation()

    def status(self) -> dict[str, Any]:
        return self._b().status()

    def run(self, actions: np.ndarray) -> dict[str, Any]:
        return self._b().run(actions)


def code_agent_spec(name: str, components: Mapping[str, str] | None = None) -> AgentSpec:
    return AgentSpec(name=name, version="0.1", action_semantics=ActionSemantics(control_mode="joint_pos", frame="base"),
                     components=dict(components or {}))


class BridgeAgent:
    """An Agent that is nothing but a bridge: the program runs elsewhere (a research session's model loop)."""

    def __init__(self, bridge: EpisodeBridge, name: str = "code.research", components: Mapping[str, str] | None = None) -> None:
        self.bridge, self.name, self.components = bridge, name, dict(components or {})

    def spec(self) -> AgentSpec:
        return code_agent_spec(self.name, self.components)

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def act(self, observation: Observation) -> ActionChunk:
        return self.bridge.act(observation)

    def on_step(self, result: StepResult) -> None:
        self.bridge.on_step(result)

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        self.bridge.on_episode_end(outcome)


__all__ = ["BridgeAgent", "EpisodeBridge", "LiveExecutor", "TimedExecutor", "code_agent_spec"]
