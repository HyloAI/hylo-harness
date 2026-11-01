"""Conformance kit: contract checks every plugin must pass (design/plugin-system.md §6).

    from worldcurrent.testing.conformance import check_environment, check_agent, check_memory_store, check_scorer
    report = check_environment(lambda: MyEnv(), task)          # -> ConformanceReport
    assert report.ok, report.summary()

Also available as `wc conformance env|agent|memory <plugin> [--params JSON]`.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
from pydantic import Field

from ..core.agent import Agent
from ..core.contract import Contract
from ..core.env import Environment, Snapshottable
from ..core.task import TaskInstance, TaskSpec
from ..core.types import Action, ActionChunk, EnvSpec, Observation, StepResult


class CheckResult(Contract):
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


class ConformanceReport(Contract):
    subject: str
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed or c.skipped for c in self.checks)

    def summary(self) -> str:
        lines = [f"{self.subject}: {'PASS' if self.ok else 'FAIL'}"]
        for c in self.checks:
            tag = "skip" if c.skipped else ("ok" if c.passed else "FAIL")
            lines.append(f"  [{tag:>4}] {c.name}" + (f": {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def _run(report: ConformanceReport, name: str, fn: Callable[[], str | None], skip: bool = False) -> None:
    if skip:
        report.checks.append(CheckResult(name=name, passed=True, skipped=True))
        return
    try:
        detail = fn() or ""
        report.checks.append(CheckResult(name=name, passed=True, detail=detail))
    except Exception as exc:  # noqa: BLE001 - a failed check is data
        report.checks.append(CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"[:400]))


def _same(a: Observation, b: Observation) -> bool:
    return (set(a.state) == set(b.state) and all(np.array_equal(a.state[k], b.state[k]) for k in a.state)
            and a.instruction == b.instruction)


def check_environment(env_factory: Callable[[], Environment], task: TaskInstance | TaskSpec, *, n_steps: int = 5,
                      spec_budget_s: float = 2.0) -> ConformanceReport:
    task = task if isinstance(task, TaskInstance) else TaskInstance(spec=task, seed=0)
    env = env_factory()
    rep = ConformanceReport(subject=f"environment {type(env).__name__}")
    spec_holder: dict[str, EnvSpec] = {}
    try:
        def spec_cheap() -> str:
            t = time.perf_counter()
            spec_holder["spec"] = env.spec()
            dt = time.perf_counter() - t
            assert isinstance(spec_holder["spec"], EnvSpec), "spec() must return EnvSpec"
            assert dt < spec_budget_s, f"spec() took {dt:.2f}s (must not build the simulator)"
            return f"{dt * 1000:.1f} ms"
        _run(rep, "spec() is cheap and typed", spec_cheap)
        spec = spec_holder.get("spec")

        def spec_roundtrip() -> None:
            assert EnvSpec.model_validate_json(spec.model_dump_json()) == spec  # type: ignore[union-attr]
        _run(rep, "spec survives JSON (wire format)", spec_roundtrip, skip=spec is None)

        first: dict[str, Observation] = {}

        def reset_obs() -> str:
            obs = env.reset(task)
            assert isinstance(obs, Observation)
            missing = [k for k in spec.observation.state if k not in obs.state]  # type: ignore[union-attr]
            assert not missing, f"state keys missing from observation: {missing}"
            for k, shape in spec.observation.state.items():  # type: ignore[union-attr]
                assert tuple(np.shape(obs.state[k])) == tuple(shape), f"{k}: shape {np.shape(obs.state[k])} != {shape}"
            first["obs"] = obs
            return f"instruction={obs.instruction!r}"
        _run(rep, "reset returns an Observation matching the spec", reset_obs, skip=spec is None)

        def seeded() -> None:
            again = env.reset(task)
            assert _same(first["obs"], again), "same TaskInstance (seed) gave a different initial observation"
        _run(rep, "reset is deterministic in the seed", seeded,
             skip=spec is None or not spec.capabilities.seedable or "obs" not in first)

        def stepping() -> str:
            env.reset(task)
            low, high = spec.action.low_array, spec.action.high_array  # type: ignore[union-attr]
            zero = np.clip(np.zeros(spec.action.shape), low, high)  # type: ignore[union-attr]
            for i in range(n_steps):
                r = env.step(Action(data=zero))
                assert isinstance(r, StepResult)
                assert isinstance(r.terminated, bool) and isinstance(r.truncated, bool)
                assert r.success in (None, True, False)
                if r.success:
                    assert r.terminated or not r.truncated, "success must not be reported as a truncation"
                if r.terminated or r.truncated:
                    return f"ended after {i + 1} steps"
                assert StepResult.model_validate_json(r.model_dump_json()).terminated == r.terminated
            return f"{n_steps} steps"
        _run(rep, "step returns StepResult with separate success/terminated/truncated", stepping, skip=spec is None)

        def snapshot() -> None:
            env.reset(task)
            zero = np.clip(np.zeros(spec.action.shape), spec.action.low_array, spec.action.high_array)  # type: ignore[union-attr]
            env.step(Action(data=zero))
            blob = env.snapshot()  # type: ignore[attr-defined]
            a = [env.step(Action(data=zero)).observation for _ in range(3)]
            env.restore(blob)  # type: ignore[attr-defined]
            b = [env.step(Action(data=zero)).observation for _ in range(3)]
            assert all(_same(x, y) for x, y in zip(a, b)), "restore(snapshot()) did not reproduce the trajectory"
        _run(rep, "snapshot/restore reproduces the trajectory", snapshot,
             skip=spec is None or not spec.capabilities.snapshot or not isinstance(env, Snapshottable))
    finally:
        _run(rep, "close() works", lambda: env.close())
    return rep


def check_agent(agent: Agent, env: Environment, task: TaskInstance | TaskSpec, *, deterministic: bool = True) -> ConformanceReport:
    from ..core.kernel import EpisodeRunner

    task = task if isinstance(task, TaskInstance) else TaskInstance(spec=task, seed=0)
    rep = ConformanceReport(subject=f"agent {type(agent).__name__}")
    spec = env.spec()

    def compatible() -> str:
        EpisodeRunner(env, agent)          # raises IncompatibleComponents on contract mismatch
        return agent.spec().name
    _run(rep, "spec is compatible with the environment", compatible)

    def chunk_shape() -> None:
        obs = env.reset(task)
        from ..core.agent import EpisodeContext
        agent.reset(EpisodeContext(episode_id="conformance", task=task, env_spec=spec, emit=lambda t, p: None, seed=task.seed))
        c = agent.act(obs)
        assert isinstance(c, ActionChunk)
        assert tuple(c.actions.shape[1:]) == tuple(spec.action.shape), f"chunk {c.actions.shape} vs action {spec.action.shape}"
        assert np.all(np.isfinite(c.actions)), "non-finite actions"
    _run(rep, "act returns finite chunks of the env's action shape", chunk_shape)

    def reset_clean() -> None:
        runner = EpisodeRunner(env, agent)
        a = runner.run(task).outcome
        b = runner.run(task).outcome
        assert (a.status, a.steps) == (b.status, b.steps), f"episode differs after reset: {a.status}/{a.steps} vs {b.status}/{b.steps}"
    _run(rep, "reset leaves no state behind (same seed, same episode)", reset_clean, skip=not deterministic)
    return rep


def check_memory_store(store_factory: Callable[[], Any]) -> ConformanceReport:
    from ..core.memory import MemoryItem, MemoryKind, MemoryQuery, Pose

    rep = ConformanceReport(subject="memory store")
    item = lambda x, **kw: MemoryItem(kind=MemoryKind.SPATIAL, key="cup", content=f"cup at {x}",  # noqa: E731
                                      pose=Pose(frame_id="map", position=(x, 0.0, 0.0)), **kw)

    def dedup() -> None:
        s = store_factory()
        s.write([item(0.0)])
        assert s.write([item(0.0)]).events[0][0] == "SKIP_DUP"
    _run(rep, "duplicate writes are skipped", dedup)

    def invalidate() -> None:
        s = store_factory()
        s.write([item(0.0, valid_from=1.0)])
        s.write([item(1.0, valid_from=2.0)])
        now = s.query(MemoryQuery(key="cup"))
        then = s.query(MemoryQuery(key="cup", as_of=1.5))
        assert len(now) == 1 and now[0].item.pose.position[0] == 1.0
        assert len(then) == 1 and then[0].item.pose.position[0] == 0.0
    _run(rep, "a new fact invalidates the old one; as_of sees history", invalidate)

    def radius() -> None:
        s = store_factory()
        s.write([item(0.0), MemoryItem(kind=MemoryKind.SPATIAL, key="far", content="far",
                                       pose=Pose(frame_id="map", position=(0.3, 0.3, 0.0)))])
        hits = s.query(MemoryQuery(near=(Pose(frame_id="map", position=(0.0, 0.0, 0.0)), 0.35)))
        assert {h.item.key for h in hits} == {"cup"}, "radius query must use true distance"
        assert s.query(MemoryQuery(near=(Pose(frame_id="odom", position=(0.0, 0.0, 0.0)), 1.0))) == [], "frames must match"
    _run(rep, "radius queries use real distance within one frame", radius)

    def forget() -> None:
        s = store_factory()
        s.write([item(0.0)])
        assert s.forget(MemoryQuery(key="cup")) == 1 and s.query(MemoryQuery(key="cup")) == []
    _run(rep, "forget soft-expires", forget)

    def snapshot() -> None:
        s = store_factory()
        h0 = s.snapshot()
        s.write([item(0.0)])
        assert s.snapshot() != h0, "snapshot hash must change with content"
    _run(rep, "snapshot hash tracks content", snapshot)
    return rep


def check_scorer(scorer: Any, trace: Any, task: TaskInstance) -> ConformanceReport:
    rep = ConformanceReport(subject=f"scorer {getattr(scorer, 'name', type(scorer).__name__)}")

    def from_trace() -> str:
        a, b = scorer.score(trace, task), scorer.score(trace, task)
        assert a == b, "scoring the same trace twice must give the same score"
        return f"value={a.value}"
    _run(rep, "scores from the trace alone, deterministically", from_trace)
    return rep


def describe(report: ConformanceReport) -> str:
    return json.dumps(report.model_dump(), indent=2)


def check_guard(guard: Any, *, action_dim: int = 2, trials: int = 50, seed: int = 0) -> ConformanceReport:
    """A guard plugin: named, returns a finite action of the same shape, never enlarges what it was given beyond the
    action box, works without state, and is deterministic for the same input."""
    from ..core.safety import SafetyContext
    from ..core.types import BoxSpace

    rep = ConformanceReport(subject=f"guard {getattr(guard, 'name', type(guard).__name__)}")
    space = BoxSpace(shape=(action_dim,), low=(-1.0,) * action_dim, high=(1.0,) * action_dim)
    rng = np.random.default_rng(seed)

    def named() -> str:
        if not isinstance(getattr(guard, "name", None), str) or not guard.name:
            raise AssertionError("guard.name must be a non-empty string")
        return guard.name

    def shapes() -> str:
        for _ in range(trials):
            a = rng.uniform(-1, 1, action_dim)
            ctx = SafetyContext(space=space, state={"pos": rng.uniform(-1, 1, action_dim)}, last_action=rng.uniform(-1, 1, action_dim))
            v = guard.review(a.copy(), ctx)
            if v.data.shape != a.shape or not np.all(np.isfinite(v.data)):
                raise AssertionError(f"bad output {v.data!r}")
            if not v.modified and not np.array_equal(v.data, a):
                raise AssertionError("changed the action without reporting modified=True")
        return f"{trials} random actions"

    def stateless() -> None:
        guard.review(np.zeros(action_dim), SafetyContext(space=space))

    def deterministic() -> None:
        a = rng.uniform(-1, 1, action_dim)
        ctx = SafetyContext(space=space, state={"pos": np.zeros(action_dim)}, last_action=np.zeros(action_dim))
        if not np.array_equal(guard.review(a.copy(), ctx).data, guard.review(a.copy(), ctx).data):
            raise AssertionError("same input, different output")

    _run(rep, "has a name", named)
    _run(rep, "returns finite actions of the same shape and reports changes", shapes)
    _run(rep, "works without state or a previous action", stateless)
    _run(rep, "is deterministic", deterministic)
    return rep


def check_policy(policy: Any, env: Environment, task: TaskInstance | TaskSpec) -> ConformanceReport:
    """A policy plugin (core.policy.ChunkPolicy): its spec fits the environment, `sample` returns finite (N, T, A)
    chunks of the action shape, the same seeds give the same chunks (randomness comes from the seeds only), and
    `default_execution_steps` is within the chunk."""
    from ..core.agent import EpisodeContext
    from ..core.compat import check_compatibility

    task = task if isinstance(task, TaskInstance) else TaskInstance(spec=task, seed=0)
    rep = ConformanceReport(subject=f"policy {type(policy).__name__}")
    spec = env.spec()
    obs = env.reset(task)
    out: dict[str, Any] = {}

    def compatible() -> str:
        r = check_compatibility(policy.spec(), spec)
        assert r.ok, "; ".join(r.errors)
        return policy.spec().name

    def shapes() -> str:
        policy.reset(EpisodeContext(episode_id="conformance", task=task, env_spec=spec, emit=lambda t, p: None, seed=task.seed))
        c = np.asarray(policy.sample(obs, [1, 2, 3], 1.0))
        assert c.ndim == 3 and c.shape[0] == 3, f"expected (3, T, A) for 3 seeds, got {c.shape}"
        assert tuple(c.shape[2:]) == tuple(spec.action.shape), f"action dim {c.shape[2:]} vs {spec.action.shape}"
        assert np.all(np.isfinite(c)), "non-finite actions"
        assert np.asarray(policy.sample(obs, None, 1.0)).shape[0] == 1, "seeds=None must give one chunk"
        out["chunks"] = c
        return f"chunk {c.shape[1]} x {c.shape[2]}"

    def seeded() -> str:
        a, b = np.asarray(policy.sample(obs, [7, 8], 1.0)), np.asarray(policy.sample(obs, [7, 8], 1.0))
        assert np.allclose(a, b), "the same seeds gave different chunks: draw noise from the seeds only"
        stochastic = not np.allclose(a[0], a[1])
        return "stochastic (seeded)" if stochastic else "deterministic"

    def execution_steps() -> None:
        k = int(policy.default_execution_steps())
        assert 1 <= k <= out["chunks"].shape[1], f"default_execution_steps {k} outside the chunk"

    _run(rep, "spec is compatible with the environment", compatible)
    _run(rep, "sample returns finite (N, T, A) chunks of the action shape", shapes)
    _run(rep, "the same seeds give the same chunks", seeded, skip="chunks" not in out)
    _run(rep, "default_execution_steps is within the chunk", execution_steps, skip="chunks" not in out)
    return rep
