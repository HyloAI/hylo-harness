"""Execution kernel: the one episode loop, used for evaluation, learning rollouts and deployment.

    env.reset -> agent.reset -> loop[ request a chunk (when the queue runs low) -> merge it when it arrives
                                      -> SafetyShield.review -> env.step -> agent.on_step ]
              -> episode.end -> agent.on_episode_end

`Timing` decides how policy time relates to robot time:
- "sync" (default): the world waits for every policy call; a chunk is requested when the queue is empty and runs
  `execution_steps` actions. Deterministic; what benchmarks have always measured.
- "realtime": the robot keeps moving while the policy thinks (LeRobot async inference, openpi's runtime,
  vla-eval's live runner): a chunk is requested when at most `refill_steps` actions remain, a chunk computed from
  the observation of step s that arrives at step t has its first t - s actions dropped (aligned merge), an empty
  queue executes a hold action, and a watchdog aborts after `max_hold_steps` consecutive holds. On the "virtual"
  clock the policy latency is measured (or fixed by `latency_s`) and the steps it would have taken run before the
  chunk is merged: deterministic, so benchmarks can measure deployment behaviour. On the "wall" clock the policy
  runs in a worker thread and the loop ticks at the control rate: the workcell next to a robot.

Both modes share the SafetyShield, the trace events, error classification and the operator channel (stop, latched
e-stop, reset confirmation, outcome verdicts), so what the gate approved in simulation is what the robot runs.

This is "driven mode". In "served mode" (an external harness owns the loop) the same Agent + SafetyShield + trace
are wrapped by a protocol adapter; see docs/execution-lifecycle.md.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
from pydantic import Field

from . import events as ev
from .agent import Agent, EpisodeContext, EpisodeOutcome
from .compat import CompatReport, require_compatible
from .contract import Contract
from .env import Environment, PrivilegedStateSource
from .errors import AgentError, AgentTimeout, EnvironmentFault, SafetyAbort, WorldcurrentError
from .events import Event, EventBus
from .safety import SafetyShield, build_guards
from .task import TaskInstance
from .trace import TraceStore
from .types import Action, EnvSpec, Observation

ABSOLUTE_MODES = ("joint_pos", "eef_abs_pose")


class Timing(Contract):
    """How policy time relates to robot time. Part of what an experiment measures."""

    mode: Literal["sync", "realtime"] = "sync"
    clock: Literal["virtual", "wall"] = "virtual"   # realtime: simulated latency, or a real control loop
    control_hz: float | None = None                 # default: the environment's
    refill_steps: int = Field(default=2, ge=0)      # request the next chunk when this many queued actions remain
    adaptive_refill: bool = True                    # ... or earlier: when the queue covers only the recent latency + 1 step
    latency_s: float | None = None                  # virtual clock: fixed inference latency instead of the measured one
    max_hold_steps: int = Field(default=100, ge=1)  # watchdog
    require_reset_confirm: bool = False             # wait for the operator's "reset_done" before an episode starts
    verdict_timeout_s: float = 0.0                  # wait this long for the operator's verdict when the env cannot judge


def hold_action(env: Any, spec: EnvSpec, last: np.ndarray | None) -> np.ndarray:
    """What to execute when no action is queued: the env's own `hold_action(last)`, the last target for absolute
    control, or zero motion (gripper kept) for delta / velocity control."""
    custom = getattr(env, "hold_action", None)
    if callable(custom):
        return np.asarray(custom(last), dtype=float)
    shape = spec.action.shape
    sem = spec.action.semantics
    if sem is not None and sem.control_mode in ABSOLUTE_MODES:
        if last is None:
            raise SafetyAbort("no target to hold for absolute control (the environment must provide hold_action)")
        return last.copy()
    a = np.zeros(shape)
    if last is not None and sem is not None and sem.gripper:
        a.reshape(-1)[-1] = last.reshape(-1)[-1]                   # keep the gripper where it is
    return a


class OperatorChannel:
    """Thread-safe operator input: stop, estop (latched), clear_fault, reset_done, verdict."""

    def __init__(self) -> None:
        self._q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.fault: str | None = None
        self.stop = threading.Event()
        self.reset_done = threading.Event()
        self.verdicts: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.log: list[dict[str, Any]] = []

    def post(self, event: Mapping[str, Any]) -> None:
        e = {"t": time.time(), **event}
        self.log.append(e)
        kind = e.get("type")
        if kind == "estop":
            self.fault = f"e-stop by {e.get('by', 'operator')}"
            self.stop.set()
        elif kind == "stop":
            self.stop.set()
        elif kind == "clear_fault":
            self.fault = None
        elif kind == "reset_done":
            self.reset_done.set()
        elif kind == "verdict":
            if e.get("verdict") not in ("success", "failure", "partial"):
                raise ValueError("verdict must be success, failure or partial")
            self.verdicts.put(e)
        else:
            raise ValueError(f"unknown operator event {kind!r}")
        self._q.put(e)

    def drain(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


@dataclass(frozen=True)
class EpisodeRecord:
    outcome: EpisodeOutcome
    safety_interventions: int
    events: int
    compat: CompatReport
    holds: int = 0
    chunks: int = 0
    verdict: dict[str, Any] | None = None
    status: str = ""                                # as written to episode.end ("unlabeled" when nobody judged)


def obs_payload(obs: Observation, with_images: bool = True) -> dict[str, Any]:
    return {"state": dict(obs.state), "images": dict(obs.images) if with_images else {},
            "instruction": obs.instruction, "time": obs.time}


@dataclass
class _Pending:
    request_step: int
    arrival_step: int | None = None                 # virtual clock / sync
    future: Future | None = None                    # wall clock
    chunk: Any = None
    latency_s: float = 0.0


class EpisodeRunner:
    def __init__(self, env: Environment, agent: Agent, *, trace: TraceStore | None = None,
                 bus: EventBus | None = None, safety: Mapping[str, Any] | None = None,
                 act_timeout_s: float | None = None, record_privileged: bool = True, image_every: int = 1,
                 timing: Timing | None = None, operator: OperatorChannel | None = None) -> None:
        self.env, self.agent = env, agent
        self.env_spec = env.spec()
        self.compat = require_compatible(agent.spec(), self.env_spec)
        safety = dict(safety or {})
        self.shield = SafetyShield(self.env_spec, build_guards(safety), clip_tolerance=float(safety.get("clip_tolerance", 0.05)))
        self.image_every = image_every
        self.bus = bus or EventBus()
        self.trace = trace
        if trace is not None:
            self.bus.subscribe(trace.write, critical=True)
        self.act_timeout_s = act_timeout_s
        self.record_privileged = record_privileged and isinstance(env, PrivilegedStateSource) \
            and self.env_spec.capabilities.privileged_state
        self.timing = timing or Timing()
        self.realtime = self.timing.mode == "realtime"
        self.hz = float(self.timing.control_hz or self.env_spec.control_hz)
        self.operator = operator or OperatorChannel()
        self._seq, self._lock = 0, threading.Lock()
        self._steps: list[Any] = []                 # wall clock: step results for the agent, delivered in its thread
        wall = self.realtime and self.timing.clock == "wall"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wc-policy") if wall else None

    def _emit(self, episode_id: str, type_: str, payload: Mapping[str, Any], *,
              source: str = "kernel", sim_time: float | None = None) -> None:
        with self._lock:                                             # a policy thread emits agent events too
            self._seq += 1
            self.bus.publish(Event(type=type_, episode_id=episode_id, seq=self._seq,
                                   payload=payload, source=source, sim_time=sim_time))

    def run(self, task: TaskInstance, *, episode_id: str | None = None) -> EpisodeRecord:
        if self.operator.fault:
            raise SafetyAbort(f"latched fault: {self.operator.fault} (clear it first)")
        eid = episode_id or f"{task.spec.id.replace('/', '.')}-s{task.seed}-{uuid.uuid4().hex[:8]}"
        self._seq = 0
        self.operator.stop.clear()
        while not self.operator.verdicts.empty():                   # a verdict belongs to the episode it follows
            self.operator.verdicts.get_nowait()
        self.shield.reset()
        aspec = self.agent.spec()
        start = {
            "task_id": task.spec.id, "task_version": task.spec.version, "task_digest": task.spec.digest(),
            "seed": task.seed, "split": task.split, "episode_index": task.episode_index,
            "env": f"{self.env_spec.name}@{self.env_spec.version}",
            "agent": f"{aspec.name}@{aspec.version}", "agent_components": dict(aspec.components),
            "guards": self.shield.guard_names, "compat_warnings": self.compat.warnings,
        }
        if self.realtime:
            start.update(runtime=self.timing.model_dump(mode="json"), control_hz=self.hz)
        self._emit(eid, ev.EPISODE_START, start)
        steps, holds, consecutive, chunks, success = 0, 0, 0, 0, None
        status, reason, last, pending = "failure", "max_steps", None, None
        try:
            self._wait_reset(eid)
            try:
                obs = self.env.reset(task)
            except Exception as exc:
                raise EnvironmentFault(f"reset failed: {exc!r}") from exc
            self._emit(eid, ev.OBSERVATION, obs_payload(obs), sim_time=obs.time)
            self.agent.reset(EpisodeContext(episode_id=eid, task=task, env_spec=self.env_spec, seed=task.seed,
                                            emit=lambda t, p: self._emit(eid, t, p, source="agent")))
            q: list[np.ndarray] = []
            recent: list[float] = []                                 # recent policy latencies (adaptive refill)
            tick_t = time.perf_counter()
            while steps < task.spec.max_steps:
                for e in self.operator.drain():
                    self._emit(eid, f"operator.{e['type']}", e, source="operator")
                if self.operator.stop.is_set():
                    raise SafetyAbort(self.operator.fault or "stopped by the operator")
                # 1. ask for the next chunk when the queue runs low (sync: when it is empty)
                refill = 0
                if self.realtime:
                    refill = self.timing.refill_steps
                    if self.timing.adaptive_refill and recent:
                        refill = max(refill, math.ceil(max(recent[-5:]) * self.hz) + 1)
                if pending is None and len(q) <= refill:
                    pending = self._request(obs, steps)
                # 2. merge a chunk that has arrived, aligned to the steps that passed meanwhile
                if pending is not None and self._arrived(pending, steps):
                    q, chunks = self._merge(eid, pending, steps, obs), chunks + 1
                    recent.append(pending.latency_s)
                    pending = None
                # 3. execute one action, or hold
                if q:
                    data, consecutive = q.pop(0), 0
                else:
                    data = hold_action(self.env, self.env_spec, last)
                    holds, consecutive = holds + 1, consecutive + 1
                    self._emit(eid, "x.workcell.hold", {"step": steps, "consecutive": consecutive}, source="workcell")
                    if consecutive > self.timing.max_hold_steps:
                        raise SafetyAbort(f"watchdog: no new actions for {consecutive} steps")
                self.shield.observe(obs.state)
                approved, review = self.shield.review(Action(data=np.asarray(data, dtype=float) if self.realtime else data))
                self._emit(eid, ev.SAFETY_REVIEW, review.to_payload(), sim_time=obs.time)
                if review.modified:
                    q = []                                           # the plan no longer matches what will happen
                self._emit(eid, ev.ACTION, {"data": approved.data}, sim_time=obs.time)
                try:
                    result = self.env.step(approved)
                except Exception as exc:
                    raise EnvironmentFault(f"step failed: {exc!r}") from exc
                steps += 1
                last = np.asarray(approved.data, dtype=float)
                obs = result.observation
                self._emit(eid, ev.STEP_RESULT, {
                    "step": steps, "reward": result.reward, "terminated": result.terminated,
                    "truncated": result.truncated, "success": result.success, "info": dict(result.info),
                }, sim_time=obs.time)
                keep = self.image_every > 0 and steps % self.image_every == 0
                self._emit(eid, ev.OBSERVATION, obs_payload(obs, keep), sim_time=obs.time)
                if self.record_privileged:
                    self._emit(eid, ev.PRIVILEGED_STATE, self.env.privileged_state(),  # type: ignore[attr-defined]
                               source="env", sim_time=obs.time)
                self._on_step(result)
                success = result.success
                if result.terminated or result.truncated:
                    reason = "sim_error" if result.info.get("sim_error") else (
                        "terminated" if result.terminated else "truncated")
                    break
                if self._pool is not None:                           # wall clock: pace the control loop
                    tick_t += 1.0 / self.hz
                    time.sleep(max(0.0, tick_t - time.perf_counter()))
            if reason == "sim_error":
                status = "infra_invalid"
            elif success is None and self.realtime:
                status = "unlabeled"                                 # the env cannot judge: the operator may
            else:
                status = "success" if success else "failure"
        except SafetyAbort as exc:
            status, reason = ("error" if "watchdog" in str(exc) else "safety_abort"), str(exc)
            self._emit(eid, ev.ERROR, {"phase": exc.phase, "message": str(exc)})
            stop = getattr(self.env, "stop", None)                   # robots: brake / hold now
            if callable(stop):
                stop()
        except EnvironmentFault as exc:
            status, reason = "infra_invalid", str(exc)
            self._emit(eid, ev.ERROR, {"phase": exc.phase, "message": str(exc)})
        except WorldcurrentError as exc:
            status, reason = "error", str(exc)
            self._emit(eid, ev.ERROR, {"phase": exc.phase, "message": str(exc)})
        self._flush_steps(pending)
        verdict = self._verdict(eid) if status == "unlabeled" else None
        if verdict is not None:
            status, success = ("success" if verdict["verdict"] == "success" else "failure"), verdict["verdict"] == "success"
        outcome = EpisodeOutcome(episode_id=eid, task=task, status=status if status != "unlabeled" else "failure",
                                 success=success if status in ("success", "failure") else None,
                                 steps=steps, termination_reason=reason)
        end = {"status": status, "success": outcome.success, "steps": steps, "termination_reason": reason,
               "safety_interventions": self.shield.interventions}
        if self.realtime:
            end.update(holds=holds, chunks=chunks, verdict=verdict)
        self._emit(eid, ev.EPISODE_END, end)
        try:
            self.agent.on_episode_end(outcome)
        finally:
            if self.trace is not None:
                self.trace.close_episode(eid)
        return EpisodeRecord(outcome, self.shield.interventions, self._seq, self.compat, holds, chunks, verdict, status)

    # -- operator -------------------------------------------------------------------------------------
    def _wait_reset(self, eid: str) -> None:
        if not (self.realtime and self.timing.require_reset_confirm):
            return
        self._emit(eid, "operator.awaiting_reset", {}, source="workcell")
        while not self.operator.reset_done.wait(0.2):
            if self.operator.stop.is_set():
                raise SafetyAbort("stopped while waiting for the reset")
        self.operator.reset_done.clear()

    def _verdict(self, eid: str) -> dict[str, Any] | None:
        """The operator's outcome for an episode the environment could not judge (real robots)."""
        if self.timing.verdict_timeout_s > 0:
            self._emit(eid, "operator.awaiting_verdict", {"timeout_s": self.timing.verdict_timeout_s}, source="workcell")
        deadline = time.monotonic() + self.timing.verdict_timeout_s
        while True:
            try:
                v = self.operator.verdicts.get(timeout=max(0.0, min(0.2, deadline - time.monotonic())))
                self._emit(eid, "operator.verdict", v, source="operator")
                return v
            except queue.Empty:
                if time.monotonic() >= deadline:
                    return None

    # -- agent calls ----------------------------------------------------------------------------------
    # On the wall clock the policy runs in a worker thread. The agent is not thread-safe, so step results are
    # buffered and handed to it, in order, right before its next policy call (inside the worker).
    def _on_step(self, result) -> None:
        if self._pool is not None:
            self._steps.append(result)
        else:
            self.agent.on_step(result)

    def _act(self, obs):
        while self._steps:
            self.agent.on_step(self._steps.pop(0))
        return self.agent.act(obs)

    def _flush_steps(self, pending: _Pending | None) -> None:
        if pending is not None and pending.future is not None:
            try:
                pending.future.result(timeout=60)
            except Exception:  # noqa: BLE001 - the episode is over; a late failure changes nothing
                pass
        while self._steps:
            self.agent.on_step(self._steps.pop(0))

    def _request(self, obs: Observation, step: int) -> _Pending:
        t0 = time.perf_counter()
        if self._pool is not None:
            fut = self._pool.submit(self._act, obs)
            p = _Pending(request_step=step, future=fut)
            fut.add_done_callback(lambda f: setattr(p, "latency_s", time.perf_counter() - t0))
            return p
        try:
            chunk = self.agent.act(obs)
        except Exception as exc:
            raise AgentError(f"agent.act failed: {exc!r}") from exc
        latency = time.perf_counter() - t0
        if self.act_timeout_s is not None and latency > self.act_timeout_s:
            raise AgentTimeout(f"agent.act took {latency:.3f}s > {self.act_timeout_s}s")
        if not self.realtime:                                        # the world waited
            return _Pending(request_step=step, arrival_step=step, chunk=chunk, latency_s=latency)
        lat = self.timing.latency_s if self.timing.latency_s is not None else latency
        return _Pending(request_step=step, arrival_step=step + math.ceil(lat * self.hz - 1e-9), chunk=chunk, latency_s=lat)

    def _arrived(self, p: _Pending, step: int) -> bool:
        if p.future is not None:
            if not p.future.done():
                return False
            try:
                p.chunk = p.future.result()
            except Exception as exc:
                raise AgentError(f"agent.act failed: {exc!r}") from exc
            return True
        return step >= (p.arrival_step or 0)

    def _merge(self, eid: str, p: _Pending, step: int, obs: Observation) -> list[np.ndarray]:
        chunk = p.chunk
        if not self.realtime:
            n = chunk.execution_steps or chunk.actions.shape[0]
            self._emit(eid, ev.AGENT_CHUNK, {"actions": chunk.actions, "execution_steps": chunk.execution_steps,
                                             "latency_s": p.latency_s, "provenance": dict(chunk.provenance)},
                       sim_time=obs.time)
            return [np.asarray(a) for a in chunk.actions[:n]]
        actions = np.asarray(chunk.actions, dtype=float)
        horizon = len(actions)
        execute = min(int(chunk.execution_steps or horizon), horizon)
        offset = step - p.request_step                                 # steps that passed while it was computed
        keep = list(actions[offset:max(execute, offset)]) if offset < horizon else []
        self._emit(eid, ev.AGENT_CHUNK, {"actions": actions, "execution_steps": execute, "latency_s": p.latency_s,
                                         "provenance": {**dict(chunk.provenance), "request_step": p.request_step,
                                                        "merge_step": step, "dropped": min(offset, horizon),
                                                        "queued": len(keep)}},
                   sim_time=obs.time)
        return keep


__all__ = ["ABSOLUTE_MODES", "EpisodeRecord", "EpisodeRunner", "OperatorChannel", "Timing", "hold_action", "obs_payload"]
