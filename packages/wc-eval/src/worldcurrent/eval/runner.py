"""EvaluationRunner: runs a benchmark split through the kernel, then scores from the trace.

Sequential in v0.1. Sharding / resume follow the vla-eval SQLite work-queue pattern later.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from worldcurrent.core.agent import Agent
from worldcurrent.core.env import Environment
from worldcurrent.core.kernel import EpisodeRunner, Timing
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.trace import TraceStore
from .benchmark import Benchmark
from .scoring import DEFAULT_SCORERS, Scorer, mean_ignoring_none

EVALLOG_VERSION = "wc.evallog/v0.1"


@dataclass
class EvalResult:
    run_dir: Path
    log: dict[str, Any]

    def per_key(self, metric: str = "success") -> dict[str, float | None]:
        return {s["key"]: s["scores"][metric]["value"] for s in self.log["samples"]}

    @property
    def metrics(self) -> dict[str, Any]:
        return self.log["results"]


def score_run(run_dir: Path, log_samples: list[dict[str, Any]], tasks: Mapping[str, Any],
              scorers: Sequence[Scorer]) -> list[dict[str, Any]]:
    store = TraceStore(run_dir)
    for s in log_samples:
        trace = store.read(s["episode_id"])
        s["scores"] = {sc.name: vars_score(sc.score(trace, tasks[s["key"]])) for sc in scorers}
    return log_samples


def vars_score(score: Any) -> dict[str, Any]:
    return score.model_dump(mode="json")


def _close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _run_shard(shard: list[TaskInstance], run_dir: Path, env: Environment, agent: Agent,
               safety: Mapping[str, Any] | None, record_privileged: bool,
               progress: Callable[[dict[str, Any]], None] | None, image_every: int = 1,
               timing: Timing | None = None) -> list[dict[str, Any]]:
    store = TraceStore(run_dir)
    out = []
    try:
        runner = EpisodeRunner(env, agent, trace=store, safety=safety, record_privileged=record_privileged,
                               image_every=image_every, timing=timing)
        for inst in shard:
            rec = runner.run(inst)
            sample = {"key": inst.key, "task_id": inst.spec.id, "seed": inst.seed,
                      "episode_id": rec.outcome.episode_id, "status": rec.outcome.status,
                      "steps": rec.outcome.steps, "trace": f"episodes/{rec.outcome.episode_id}.jsonl"}
            out.append(sample)
            if progress:
                progress(sample)
    finally:
        _close(env)
        store.close()
    return out


def evaluate(benchmark: Benchmark, split: str, *, env_factory: Callable[[], Environment],
             agent: Agent | None = None, agent_factory: Callable[[], Agent] | None = None,
             run_root: str | Path, safety: Mapping[str, Any] | None = None,
             scorers: Sequence[Scorer] = DEFAULT_SCORERS, manifest: Mapping[str, Any] | None = None,
             run_id: str | None = None, record_privileged: bool = True, workers: int = 1,
             progress: Callable[[dict[str, Any]], None] | None = None, image_every: int = 1,
             timing: Timing | None = None) -> EvalResult:
    """Run every TaskInstance of `split`. With workers > 1 each worker gets its own environment and
    agent (from the factories) and a disjoint shard of episodes; episode results do not depend on
    which worker ran them."""
    if agent is None and agent_factory is None:
        raise ValueError("pass an agent or an agent_factory")
    run_id = run_id or f"{benchmark.id.replace('/', '.')}-{split}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(run_root) / run_id
    instances = benchmark.instances(split)
    if workers <= 1 or len(instances) <= 1:
        a = agent if agent is not None else agent_factory()  # type: ignore[misc]
        try:
            aspec = a.spec()
            samples = _run_shard(instances, run_dir, env_factory(), a, safety, record_privileged, progress, image_every,
                                 timing)
        finally:
            if agent is None:
                _close(a)
    else:
        import queue
        from concurrent.futures import ThreadPoolExecutor

        factory = agent_factory
        if factory is None:
            raise ValueError("parallel evaluation needs an agent_factory (one agent per worker)")
        # Dynamic work queue (vla-eval style): idle workers pull the next episode, so long tasks do not leave
        # the rest of the pool idle. Instances stay grouped by task so a worker's environment rarely rebuilds.
        work: queue.Queue = queue.Queue()
        for inst in instances:
            work.put(inst)
        specs: list[Any] = []
        lock = threading.Lock()

        def worker(_: int) -> list[dict[str, Any]]:
            a = factory()
            env = env_factory()
            store = TraceStore(run_dir)
            out: list[dict[str, Any]] = []
            try:
                with lock:
                    specs.append(a.spec())
                runner = EpisodeRunner(env, a, trace=store, safety=safety, record_privileged=record_privileged,
                                       image_every=image_every, timing=timing)
                while True:
                    try:
                        inst = work.get_nowait()
                    except queue.Empty:
                        return out
                    rec = runner.run(inst)
                    sample = {"key": inst.key, "task_id": inst.spec.id, "seed": inst.seed,
                              "episode_id": rec.outcome.episode_id, "status": rec.outcome.status,
                              "steps": rec.outcome.steps, "trace": f"episodes/{rec.outcome.episode_id}.jsonl"}
                    out.append(sample)
                    if progress:
                        with lock:
                            progress(sample)
            finally:
                _close(env)
                _close(a)
                store.close()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results_by_worker = list(pool.map(worker, range(min(workers, len(instances)))))
        order = {inst.key: i for i, inst in enumerate(instances)}
        samples = sorted((s for r in results_by_worker for s in r), key=lambda s: order[s["key"]])
        aspec = specs[0] if specs else None
    tasks = {i.key: i for i in instances}
    score_run(run_dir, samples, tasks, scorers)
    results = {f"{sc.name}_mean": mean_ignoring_none(s["scores"][sc.name]["value"] for s in samples)
               for sc in scorers}
    results["n_episodes"] = len(samples)
    results["n_scored"] = sum(1 for s in samples if s["scores"]["success"]["value"] is not None) \
        if any(sc.name == "success" for sc in scorers) else len(samples)
    log = {
        "version": EVALLOG_VERSION, "run_id": run_id,
        "benchmark": {"id": benchmark.id, "version": benchmark.version, "digest": benchmark.digest(),
                      "split": split, "requires_privileged": benchmark.requires_privileged},
        "agent": aspec.model_dump(mode="json"),
        "results": results, "samples": samples,
    }
    (run_dir / "evallog.json").write_text(json.dumps(log, indent=2, default=str))
    (run_dir / "manifest.json").write_text(json.dumps(dict(manifest or {"run_id": run_id}), indent=2, default=str))
    return EvalResult(run_dir, log)
