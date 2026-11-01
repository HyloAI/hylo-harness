"""Shadow mode (RFC-0004 W4; roadmap v0.3): evaluate a candidate on real hardware without letting it act.

ShadowAgent executes the active agent and hands the same observation to a shadow (the candidate), whose chunk is
never executed. Each decision emits an `x.shadow` event: how far the shadow's executed prefix would have been from
what the robot actually did (mean / max absolute difference over arm dims, gripper agreement), and the shadow's
latency. The shadow also receives the real step results, so it follows the episode the active agent produces.

With `background=True` the shadow runs in a worker thread and its result is emitted at the next decision, so the
control loop never waits for it (on a robot the shadow must not add latency). The shadow's own events are prefixed
`x.shadow.` so nothing it emits is mistaken for the active agent's.

    agent: {type: shadow, params: {active: {type: ...}, shadow: {type: ...}, background: true}}
"""

from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.agent import EpisodeContext
from hylo.core.registry import registry
from hylo.core.types import ActionChunk, Observation


def divergence(active: np.ndarray, shadow: np.ndarray, steps: int, gripper_dim: int | None = -1) -> dict[str, float]:
    a, b = np.asarray(active, float), np.asarray(shadow, float)
    n = max(1, min(steps, len(a), len(b)))
    a, b = a[:n], b[:n]
    arm = slice(0, a.shape[1] - 1) if gripper_dim is not None and a.shape[1] > 1 else slice(None)
    d = np.abs(a[:, arm] - b[:, arm])
    out = {"steps": n, "mean_abs": round(float(d.mean()), 5), "max_abs": round(float(d.max()), 5)}
    if gripper_dim is not None and a.shape[1] > 1:
        out["gripper_agree"] = round(float(np.mean(np.sign(a[:, gripper_dim]) == np.sign(b[:, gripper_dim]))), 4)
    return out


class ShadowAgent:
    def __init__(self, active: dict[str, Any] | Any, shadow: dict[str, Any] | Any, background: bool = False,
                 gripper_dim: int | None = -1) -> None:
        build = lambda a: registry.build("agents", a) if isinstance(a, dict) else a  # noqa: E731
        self.active, self.shadow = build(active), build(shadow)
        self.background, self.gripper_dim = background, gripper_dim
        self._pool = ThreadPoolExecutor(max_workers=1) if background else None
        self._pending: list[tuple[int, int, np.ndarray, int, float, Future]] = []
        self.chunk_index = 0

    def spec(self):
        base = self.active.spec()
        sh = self.shadow.spec()
        return base.model_copy(update={"components": {**base.components, "shadow": f"{sh.name}@{sh.version}"}})

    def reset(self, ctx: EpisodeContext) -> None:
        self._flush(wait=True)
        self.ctx, self.chunk_index, self.steps = ctx, 0, 0
        self.active.reset(ctx)
        self.shadow.reset(dataclasses.replace(ctx, emit=lambda t, p: ctx.emit(f"x.shadow.{t}", p)))

    def _emit(self, idx: int, step: int, active: np.ndarray, exec_steps: int, t0: float, fut: Future) -> None:
        try:
            chunk = fut.result()
            payload = {"chunk_index": idx, "step": step, "latency_s": round(time.perf_counter() - t0, 3),
                       **divergence(active, chunk.actions, exec_steps, self.gripper_dim)}
        except Exception as exc:  # noqa: BLE001 - a failing shadow is recorded, never fatal to the episode
            payload = {"chunk_index": idx, "step": step, "error": f"{type(exc).__name__}: {exc}"[:300]}
        self.ctx.emit("x.shadow", payload)

    def _flush(self, wait: bool = False) -> None:
        keep = []
        for item in self._pending:
            if wait or item[-1].done():
                self._emit(*item)
            else:
                keep.append(item)
        self._pending = keep

    def act(self, observation: Observation) -> ActionChunk:
        self._flush()
        chunk = self.active.act(observation)
        exec_steps = int(chunk.execution_steps or len(chunk.actions))
        t0 = time.perf_counter()
        if self._pool is not None:
            fut = self._pool.submit(self.shadow.act, observation)
        else:
            fut = Future()
            try:
                fut.set_result(self.shadow.act(observation))
            except Exception as exc:  # noqa: BLE001
                fut.set_exception(exc)
        self._pending.append((self.chunk_index, self.steps, np.asarray(chunk.actions), exec_steps, t0, fut))
        if self._pool is None:
            self._flush()
        self.chunk_index += 1
        return chunk

    def on_step(self, result) -> None:
        self.steps += 1
        self.active.on_step(result)
        self.shadow.on_step(result)

    def on_episode_end(self, outcome) -> None:
        self._flush(wait=True)
        self.active.on_episode_end(outcome)
        self.shadow.on_episode_end(outcome)

    def close(self) -> None:
        for a in (self.active, self.shadow):
            fn = getattr(a, "close", None)
            if callable(fn):
                fn()
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def shadow_report(run_dir: str | Path) -> dict[str, Any]:
    """Aggregate a shadow run: how often and how far the candidate would have acted differently, per task."""
    from hylo.core.trace import TraceStore

    store = TraceStore(run_dir)
    log = json.loads((Path(run_dir) / "evallog.json").read_text())
    per_task: dict[str, list[dict[str, Any]]] = {}
    errors = 0
    for s in log["samples"]:
        for e in store.read(s["episode_id"]).of_type("x.shadow"):
            p = e["payload"]
            if "error" in p:
                errors += 1
            else:
                per_task.setdefault(s["task_id"], []).append(p)
    rows = [p for ps in per_task.values() for p in ps]
    if not rows:
        return {"decisions": 0, "errors": errors}

    def agg(ps):
        out = {"decisions": len(ps), "mean_abs": round(float(np.mean([p["mean_abs"] for p in ps])), 5),
               "p95_max_abs": round(float(np.percentile([p["max_abs"] for p in ps], 95)), 5)}
        if "gripper_agree" in ps[0]:
            out["gripper_agree"] = round(float(np.mean([p["gripper_agree"] for p in ps])), 4)
        return out

    return {**agg(rows), "errors": errors, "latency_p50_s": float(np.median([p["latency_s"] for p in rows])),
            "per_task": {t: agg(ps) for t, ps in sorted(per_task.items())}}
