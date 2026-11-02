"""Helpers shared by the agent examples: output folders, MuJoCo runs, demonstrations, learners, tables.

Not an example itself. Environments and benchmarks are referenced by registry name only
(`mujoco.planar_reach`, `mujoco/reach-v0`, ...), so nothing here imports a simulator package.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable

# A shared machine: cap BLAS threads before numpy loads (small numpy models gain nothing from 64 BLAS threads).
BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _v in BLAS_VARS:
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]            # the harness repository
AREA = Path(__file__).resolve().parent.name
EXPERT = {"type": "baseline.planar_ik", "params": {"kp": 0.3}}
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


# ---------------------------------------------------------------- command line and output

def parse_args(doc: str | None, slug: str, extra: Callable[[argparse.ArgumentParser], None] | None = None):
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="small run (< 60 s) for smoke tests")
    ap.add_argument("--out", default=None, help=f"output directory (default runs/examples/{AREA}/{slug})")
    ap.add_argument("--workers", type=int, default=16, help="parallel processes / episode workers (default 16)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.out = Path(a.out) if a.out else ROOT / "runs" / "examples" / AREA / slug
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def log(msg: str) -> None:
    print(msg, flush=True)


def write_result(out: Path, payload: dict[str, Any]) -> Path:
    path = out / "result.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def table(rows: list[dict[str, Any]], cols: list[str], title: str | None = None) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    width = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(cols)]
    lines = ([title] if title else []) + ["  ".join(c.ljust(w) for c, w in zip(cols, width)),
                                          "  ".join("-" * w for w in width)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, width)) for row in cells]
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties."""
    def ranks(x):
        x = np.asarray(x, float)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r

    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- rendering (VLM examples)

def ensure_render_env() -> bool:
    """Re-execute once with OSMesa offscreen rendering if it is installed in user space (scripts/install-osmesa-user.sh).
    LD_LIBRARY_PATH is read at process start only. Returns True when MuJoCo can render."""
    if os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
        return True
    if not (USER_MESA / "libOSMesa.so.8").exists():
        return False
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
           "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return True                                                    # not reached


# ---------------------------------------------------------------- runs

def reach_bench(rsi_states: int) -> dict[str, Any]:
    """MuJoCo planar reach under the RFC-0002 protocol: search / search_val (development: demonstrations, training),
    gate, heldout (evaluation only). 40 / 20 / 20 / 20 percent of `rsi_states` init states, two tasks each."""
    return {"type": "mujoco/reach-v0", "params": {"protocol": "rsi", "rsi_states": rsi_states}}


def config(name: str, agent: dict[str, Any], bench: dict[str, Any], *, env: str | dict = "mujoco.planar_reach",
           workers: int = 1, **kw: Any):
    """An experiment config; `env` is a registry name or a full {type, params} spec."""
    from hylo.core.config import ExperimentConfig

    return ExperimentConfig(name=name, benchmark=bench, env={"type": env} if isinstance(env, str) else env,
                            agent=agent, workers=workers, **kw)


def run(name: str, agent: dict[str, Any], bench: dict[str, Any], split: str, root: Path, *,
        env: str | dict = "mujoco.planar_reach", workers: int = 1, **kw: Any):
    from hylo.eval.experiment import run_experiment

    return run_experiment(config(name, agent, bench, env=env, workers=workers, **kw), split=split,
                          run_root=root, label=f"{name}-{split}")


def slugify(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.]+", "-", name).strip("-").lower()


def _run_shard(args: tuple) -> tuple[str, str, float]:
    """One shard of one agent's split, in a worker process: the benchmark restricted to some seeds."""
    name, agent, bench, split, seeds, root, env, kw = args
    from hylo.core.config import build_manifest
    from hylo.core.registry import registry
    from hylo.eval.experiment import build_agent, env_factory
    from hylo.eval.runner import evaluate

    cfg = config(name, agent, bench, env=env, **kw)
    full = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    sub = full.model_copy(update={"splits": {split: tuple(seeds)}})
    t = time.perf_counter()
    run_id = f"{slugify(name)}-{split}-{seeds[0]}"
    probe = build_agent(cfg)
    manifest = build_manifest(cfg, run_id=run_id, extra={
        "benchmark": {"id": full.id, "version": full.version, "digest": full.digest(), "split": split,
                      "shard_seeds": list(seeds)}, "agent_spec": probe.spec().model_dump(mode="json")})
    r = evaluate(sub, split, env_factory=env_factory(cfg), agent=probe, run_root=Path(root), safety=cfg.safety,
                 manifest=manifest, run_id=run_id, record_privileged=cfg.trace.record_privileged,
                 image_every=cfg.trace.image_every, timing=cfg.timing)
    close = getattr(probe, "close", None)
    if callable(close):
        close()
    return name, str(r.run_dir), time.perf_counter() - t


def run_many(jobs: dict[str, dict[str, Any]], bench: dict[str, Any], split: str, root: Path, *, procs: int = 16,
             env: str | dict = "mujoco.planar_reach", shards: int | None = None, **kw: Any) -> dict[str, list[Path]]:
    """Evaluate several agents on one split with a pool of `procs` processes. Each agent's seeds are cut into
    `shards` (default `procs`) run directories; numpy policies hold the GIL, so processes, not threads, give the
    speed-up. Episode results do not depend on the sharding (agents seed their noise from the task key)."""
    from hylo.core.registry import registry

    full = registry.build("benchmarks", {k: v for k, v in bench.items() if k != "split"})
    seeds = list(full.splits[split])
    n = max(1, min(len(seeds), shards or procs))
    parts = [seeds[i::n] for i in range(n)]
    args = [(name, agent, bench, split, sorted(p), str(Path(root) / slugify(name)), env, kw)
            for name, agent in jobs.items() for p in parts if p]
    out: dict[str, list[Path]] = {name: [] for name in jobs}
    for name in jobs:                                  # a rerun replaces the previous runs of the same name
        shutil.rmtree(Path(root) / slugify(name), ignore_errors=True)
    if procs <= 1:
        for name, d, _ in map(_run_shard, args):
            out[name].append(Path(d))
        return out
    with pool_executor(procs) as pool:
        for name, d, _ in pool.map(_run_shard, args):
            out[name].append(Path(d))
    return out


@contextmanager
def pool_executor(procs: int):
    """Fresh ("spawn") worker processes with one BLAS thread each: `procs` processes x several BLAS threads would
    oversubscribe a shared machine. Workers start while the pool is used, so the variables are set for that time."""
    import multiprocessing

    saved = {v: os.environ.get(v) for v in BLAS_VARS}
    os.environ.update({v: "1" for v in BLAS_VARS})
    try:
        with ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn")) as pool:
            yield pool
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def evallog(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "evallog.json").read_text())


def _dirs(run_dirs: Path | list[Path]) -> list[Path]:
    return [Path(run_dirs)] if isinstance(run_dirs, (str, Path)) else [Path(d) for d in run_dirs]


def samples(run_dirs: Path | list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """(run directory, evallog sample) of every episode of one or more (sharded) runs, in task / seed order."""
    rows = [(d, s) for d in _dirs(run_dirs) for s in evallog(d)["samples"]]
    return sorted(rows, key=lambda r: (r[1]["task_id"], r[1]["seed"]))


def episode_actions(run_dirs: Path | list[Path]) -> list[dict[str, Any]]:
    """Per episode: status, steps, executed actions (T, A) and the number of policy queries (agent.chunk events)."""
    from hylo.core.trace import TraceStore

    out = []
    for d, s in samples(run_dirs):
        store = TraceStore(d)
        tr = store.read(s["episode_id"])
        acts = [np.asarray(e["payload"]["data"], float).reshape(-1) for e in tr.of_type("action")]
        lat = [float(e["payload"].get("latency_s") or 0.0) for e in tr.of_type("agent.chunk")]
        out.append({"key": s["key"], "task_id": s["task_id"], "status": s["status"], "steps": s["steps"],
                    "actions": np.stack(acts) if acts else np.zeros((0, 2)), "queries": len(lat),
                    "latency_s": lat})
    return out


def summarize(run_dir: Path | list[Path]) -> dict[str, Any]:
    """Success with a 95% Wilson interval, mean steps, policy queries per episode, mean policy latency, and
    smoothness: mean |a_t - a_(t-1)| over every executed step (action units; lower is smoother)."""
    eps = episode_actions(run_dir)
    n = len(eps)
    k = sum(e["status"] == "success" for e in eps)
    lo, hi = wilson(k, n)
    jerk = [np.abs(np.diff(e["actions"], axis=0)).sum(1).mean() for e in eps if len(e["actions"]) > 1]
    lat = [v for e in eps for v in e["latency_s"]]
    return {"episodes": n, "success": k / n if n else None, "ci95": [round(lo, 3), round(hi, 3)],
            "steps": float(np.mean([e["steps"] for e in eps])) if n else None,
            "queries_per_ep": float(np.mean([e["queries"] for e in eps])) if n else None,
            "policy_ms": 1000 * float(np.mean(lat)) if lat else None,
            "jerk": float(np.mean(jerk)) if jerk else None}


def per_key(run_dirs: Path | list[Path]) -> dict[str, bool]:
    return {s["key"]: s["status"] == "success" for _, s in samples(run_dirs)}


def paired(candidate: dict[str, bool], baseline: dict[str, bool]) -> dict[str, Any]:
    """Same-seed comparison: episodes only the candidate solved, only the baseline solved, and the two-sided exact
    McNemar p-value (hylo.eval.scoring gives the one-sided test the gate uses)."""
    from hylo.eval.scoring import one_sided_exact_mcnemar

    keys = [k for k in candidate if k in baseline]
    w = sum(candidate[k] and not baseline[k] for k in keys)
    lo = sum(baseline[k] and not candidate[k] for k in keys)
    p = min(1.0, 2 * one_sided_exact_mcnemar(max(w, lo), min(w, lo))) if w + lo else 1.0
    return {"pairs": len(keys), "wins": w, "losses": lo, "p_two_sided": round(p, 4), "text": f"+{w}/-{lo} p={p:.2f}"}


# ---------------------------------------------------------------- demonstrations and learners

def demonstrations(out: Path, bench: dict[str, Any], *, workers: int = 16, explore: bool = False) -> dict[str, Path]:
    """Expert (IK controller) runs on the development splits, plus random-action runs for world models if asked.
    Reused when they already exist under OUT (delete the folder to regenerate)."""
    roots = {"demos": out / "data" / "demos", **({"explore": out / "data" / "explore"} if explore else {})}
    stamp = out / "data" / "bench.json"
    if stamp.exists() and json.loads(stamp.read_text()) == bench and all(p.exists() for p in roots.values()):
        return roots
    for p in roots.values():
        shutil.rmtree(p, ignore_errors=True)
    for split in ("search", "search_val"):
        run("expert", EXPERT, bench, split, roots["demos"], workers=workers)
        if explore:
            run("random", {"type": "baseline.random", "params": {"dim": 2}}, bench, split, roots["explore"],
                workers=workers)
    stamp.write_text(json.dumps(bench))
    return roots


def train(method: str, inputs: list[Path], out: Path, params: dict[str, Any] | None = None,
          splits: tuple[str, ...] = ("search", "search_val")) -> tuple[Path, dict]:
    """Seal a reference-policy (or world-model) artifact with the learner of that name; skipped when an artifact
    from the identical job already exists."""
    from hylo.core.learning import TrainJob
    from hylo.learn.jobs import run_job

    learner = method if "." in method else f"policy.{method}"
    t = time.perf_counter()
    job = TrainJob(learner=learner, inputs=[str(p) for p in inputs], out=str(out), params=params or {}, splits=splits)
    man = run_job(job)
    seconds = round(time.perf_counter() - t, 1)
    stamp = Path(out).parent / f"{Path(out).name}.train.json"      # training time of the run that built it
    if stamp.exists() and json.loads(stamp.read_text()).get("job") == job.digest():
        seconds = json.loads(stamp.read_text())["seconds"]
    else:
        stamp.write_text(json.dumps({"job": job.digest(), "seconds": seconds}))
    return out, {"digest": man.digest, "seconds": seconds, **{
        k: v for k, v in (man.metrics or {}).items() if isinstance(v, (int, float))}}


def policy_ref(method: str, path: Path, **params: Any) -> dict[str, Any]:
    return {"type": f"policy.{method}", "params": {"path": str(path), **params}}


def _train_job(args: tuple) -> tuple[str, Path, dict]:
    key, method, inputs, out, params = args
    path, info = train(method, inputs, out, params)
    return key, path, info


def train_many(jobs: dict[str, tuple[str, list[Path], Path, dict[str, Any]]], procs: int = 16) -> dict[str, tuple[Path, dict]]:
    """Several learner jobs in parallel processes: {key: (method, inputs, out, params)} -> {key: (artifact, info)}."""
    args = [(k, *v) for k, v in jobs.items()]
    if procs <= 1 or len(args) == 1:
        return {k: (p, i) for k, p, i in map(_train_job, args)}
    with pool_executor(min(procs, len(args))) as pool:
        return {k: (p, i) for k, p, i in pool.map(_train_job, args)}
