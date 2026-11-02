"""Helpers shared by the evolution examples (not an example itself).

Everything here goes through public APIs: the plugin registry, `run_experiment`, the evaluation runner and learner
jobs. MuJoCo environments are referenced by registry name only (`mujoco.planar_reach`, from hylo-env-mujoco).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]          # the harness repository
AREA = "evolve"
EXPERT = {"type": "baseline.planar_ik", "params": {"kp": 0.3}}


def parse_args(doc: str, slug: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="smaller sizes, under a minute")
    ap.add_argument("--out", default=None, help=f"output directory (default runs/examples/{AREA}/{slug})")
    ap.add_argument("--workers", type=int, default=24, help="worker processes for large evaluations")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.out = Path(a.out).resolve() if a.out else ROOT / "runs" / "examples" / AREA / slug
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def require(group: str, name: str, hint: str = "install hylo-env-mujoco (hylo-bench) into this venv") -> None:
    from hylo.core.registry import registry

    try:
        registry.reference(group, name)
    except KeyError:
        sys.exit(f"{name!r} is not registered: {hint}")


def table(rows: list[list[Any]], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def write_result(out: Path, result: dict[str, Any]) -> None:
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {out / 'result.json'}")


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def reach_config(agent: dict[str, Any], name: str, *, workers: int = 1, **bench_params: Any):
    from hylo.core.config import ExperimentConfig

    return ExperimentConfig(name=name, benchmark={"type": "mujoco/reach-v0", "params": dict(bench_params)},
                            env={"type": "mujoco.planar_reach"}, agent=agent, workers=workers,
                            tags=("examples", AREA))


def flow_agent(path: str | Path, noise_scale: float = 1.0, **steering: Any) -> dict[str, Any]:
    """The reference flow-matching policy, run by the `steered` agent (seeded noise: replayable, paired)."""
    return {"type": "steered", "params": {"sampler": {"type": "policy.flow", "params": {"path": str(path)}},
                                          "steering": {"noise_scale": noise_scale, **steering}}}


def train_flow(out: Path, *, epochs: int = 300, log=print) -> Path:
    """Record IK-expert demonstrations on the development splits (search, search_val) and train a flow policy.
    A finished job is not re-run (the artifact is sealed with the job's digest)."""
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.learn.jobs import run_job

    demos = out / "demos"
    if not (demos / ".done").exists():                      # recorded once; an interrupted recording starts over
        shutil.rmtree(demos, ignore_errors=True)
        cfg = reach_config(EXPERT, f"{AREA}-demos", protocol="rsi")
        for split in ("search", "search_val"):
            run_experiment(cfg, split=split, run_root=demos, label=f"expert-{split}")
        (demos / ".done").touch()
    t = time.time()
    m = run_job(TrainJob(learner="policy.flow", inputs=[str(demos)], out=str(out / "flow"),
                         params={"chunk": 8, "epochs": epochs, "integration_steps": 10}))
    log(f"flow policy: {m.metrics['episodes']} demonstrations, {m.metrics['samples']} samples, digest {m.digest} "
        f"({time.time() - t:.0f}s)")
    return out / "flow"


# ---------------------------------------------------------------- large evaluations in worker processes

def _eval_chunk(job: tuple[dict[str, Any], list[int], str]) -> list[dict[str, Any]]:
    """Run `agent` on MuJoCo reach for the given seeds (every seed on both tasks); outcomes only, traces in a tmp dir."""
    import tempfile

    from hylo.core.benchmark import Benchmark
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry
    from hylo.eval.experiment import build_agent, env_factory
    from hylo.eval.runner import evaluate

    cfg_d, seeds, tmp_root = job
    cfg = ExperimentConfig.model_validate(cfg_d)
    base = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    bench = Benchmark(id=base.id, version=base.version, tasks=base.tasks, splits={"pool": tuple(seeds)})
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        r = evaluate(bench, "pool", env_factory=env_factory(cfg), agent=build_agent(cfg), run_root=tmp,
                     safety=cfg.safety, record_privileged=False, timing=cfg.timing)
    return [{"key": s["key"], "task_id": s["task_id"], "status": s["status"], "steps": s["steps"],
             "success": s["scores"]["success"]["value"]} for s in r.log["samples"]]


def pool_outcomes(cfg, seeds: Iterable[int], *, workers: int, tmp_root: Path, chunk: int = 25) -> list[dict[str, Any]]:
    """Outcomes of `cfg`'s agent on many seeds, sharded over worker processes (the runner's own workers are
    threads, so they do not speed up CPU-bound numpy agents)."""
    seeds = list(seeds)
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg.model_dump(mode="json"), seeds[i:i + chunk], str(tmp_root)) for i in range(0, len(seeds), chunk)]
    ctx = None
    if sys.platform.startswith("linux"):
        import multiprocessing as mp

        ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs), os.cpu_count() or 1)), mp_context=ctx) as ex:
        return [row for rows in ex.map(_eval_chunk, jobs) for row in rows]
