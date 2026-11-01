"""Build a frozen ActionBank from worldcurrent traces, with keys from the policy's own visual tokens (RFC-0003 M1).

    python -m worldcurrent.policies.lerobot.memory_build --runs runs/<search runs> ... --out runs/memory/<name> \
        [--repo-id lerobot/smolvla_libero] [--workers 8] [--task-prefix libero/libero_10/]

Only chunks of simulator-verified successful episodes from development splits are stored (search / search_val;
gate and held-out never). Replays of identical episodes (common random numbers across runs) are stored once.
Writes bank.npz / bank.json plus build_report.json with nearest-neighbour similarity statistics, which is what
`memory_steer.sim_floor` should be calibrated against, and seals the directory as an `action_memory` artifact
(RFC-0004). `ActionMemoryLearner` is the same builder behind the learner interface ("lerobot.action_memory").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_SAMPLER = None


def _init(repo_id: str, threads: int) -> None:
    global _SAMPLER
    from .experience_agent import PriorLeRobotSampler

    _SAMPLER = PriorLeRobotSampler(repo_id=repo_id, num_threads=threads)


def _episode_segments(job: tuple) -> list[dict[str, Any]]:
    from worldcurrent.agents.memory_steer import verified_decision_points
    from worldcurrent.core.trace import TraceStore
    from worldcurrent.core.types import Observation

    run_dir, episode_id, splits, store = job
    out = []
    run_id = json.loads((Path(run_dir) / "manifest.json").read_text())["run_id"]
    for pt in verified_decision_points(TraceStore(run_dir).read(episode_id), splits=splits):
        o = pt["observation"]
        obs = Observation(state={k: np.asarray(v) for k, v in o["state"].items()},
                          images={k: np.asarray(v, dtype=np.uint8) for k, v in o["images"].items()},
                          instruction=o.get("instruction"))
        seg = {**{k: v for k, v in pt.items() if k not in ("observation", "provenance")}, "run_id": run_id}
        if store == "noise":                  # the exact initial noise that produced this successful chunk
            from .experience_agent import seed_noise

            prov = pt.get("provenance") or {}
            seeds, sel = prov.get("seeds") or [], int(prov.get("selected", 0) or 0)
            if not seeds or sel >= len(seeds) or float(prov.get("t0", 1.0)) < 1.0:
                continue                      # unseeded or memory-steered chunks: their noise is not recoverable
            cfg = _SAMPLER.cfg
            seg["actions"] = seed_noise(seeds[sel], (int(cfg.chunk_size), int(getattr(cfg, "max_action_dim", 32))),
                                        float(prov.get("noise_scale", 1.0) or 1.0)).astype(np.float32)
        seg["key"] = _SAMPLER.retrieval_key(obs).astype(np.float32)
        out.append(seg)
    return out


def _nn_stats(keys: np.ndarray, tasks: np.ndarray, episodes: np.ndarray, n: int = 400, seed: int = 0) -> dict[str, Any]:
    """Nearest-neighbour cosine similarity from other episodes: same task vs other tasks."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    k = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    same, other = [], []
    for i in idx:
        sims = k @ k[i]
        sims[episodes == episodes[i]] = -np.inf
        st, ot = sims[tasks == tasks[i]], sims[tasks != tasks[i]]
        if np.isfinite(st).any():
            same.append(float(st.max()))
        if np.isfinite(ot).any():
            other.append(float(ot.max()))
    pct = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 5) for p in (5, 25, 50, 75, 95)} if v else {}  # noqa: E731
    return {"nn_same_task_other_episode": pct(same), "nn_other_task": pct(other), "n_probe": len(idx)}


def build(runs: list[str], out: str, *, repo_id: str = "lerobot/smolvla_libero", workers: int = 8, threads: int = 4,
          splits: tuple[str, ...] = ("search", "search_val"), task_prefix: str | None = None,
          capacity: int = 50_000, store: str = "actions") -> dict[str, Any]:
    from worldcurrent.core.artifact import Provenance, development_runs, write_manifest
    from worldcurrent.core.experience import ActionSegment, EpisodeRef
    from worldcurrent.memory.action_bank import ActionBank

    sources = development_runs(runs, splits)              # gate / held-out runs are never read
    run_dirs = [rd for rd, _ in sources]
    jobs = []
    for rd in run_dirs:
        for s in json.loads((rd / "evallog.json").read_text())["samples"]:
            if s["status"] == "success" and (not task_prefix or s["task_id"].startswith(task_prefix)):
                jobs.append((str(rd), s["episode_id"], splits, store))
    if not jobs:
        raise SystemExit("no verified development-split successes in these runs")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(repo_id, threads)) as pool:
        results = list(pool.map(_episode_segments, jobs, chunksize=1))
    seen, segs = set(), []
    for pts in results:
        for p in pts:
            sig = hashlib.sha256(np.asarray(p["actions"], dtype=np.float32).round(5).tobytes()
                                 + f"{p['task_id']}|{p['step']}".encode()).hexdigest()
            if sig in seen:
                continue                                 # the same episode replayed in another run
            seen.add(sig)
            segs.append(ActionSegment(key=p["key"], actions=p["actions"], task_id=p["task_id"], step=p["step"],
                                      progress=p["progress"],
                                      source=EpisodeRef(run_id=p["run_id"], episode_id=p["episode_id"], split=p["split"])))
    bank = ActionBank(capacity=capacity, frozen=False)
    bank.add(segs)
    digest = bank.save(out)
    keys = np.stack([np.asarray(s.key, dtype=np.float32) for s in segs])
    report = {"digest": digest, "segments": len(bank), "episodes": len({s.source.episode_id for s in segs}),
              "decision_points_read": sum(len(r) for r in results), "duplicates_dropped": sum(len(r) for r in results) - len(segs),
              "tasks": sorted({s.task_id for s in segs}), "splits": list(splits), "repo_id": repo_id,
              "key_dim": int(keys.shape[1]), "runs": [str(r) for r in run_dirs],
              **_nn_stats(keys, np.array([s.task_id for s in segs]), np.array([s.source.episode_id for s in segs]))}
    (Path(out) / "build_report.json").write_text(json.dumps(report, indent=2))
    report["store"] = store
    report["artifact"] = write_manifest(out, kind="noise_policy" if store == "noise" else "action_memory", role="behaviour", metrics={
        k: report[k] for k in ("digest", "segments", "episodes", "key_dim")},
        provenance=Provenance(builder=ActionMemoryLearner.name, sources=[r for _, r in sources],
                              params={"repo_id": repo_id, "task_prefix": task_prefix, "capacity": capacity, "store": store})).digest
    return report


class ActionMemoryLearner:
    """Learner interface (core.learning): job.params are `build` keyword arguments."""

    name = "worldcurrent.policies.lerobot.memory_build@0.1"

    def fit(self, job):
        from worldcurrent.core.artifact import read_manifest

        build(list(job.inputs), job.out, splits=tuple(job.splits), **dict(job.params))
        return read_manifest(job.out)


class NoisePriorLearner(ActionMemoryLearner):
    """Learner "lerobot.noise_prior": the same builder storing each successful chunk's initial flow noise."""

    name = "worldcurrent.policies.lerobot.memory_build[noise]@0.1"

    def fit(self, job):
        from worldcurrent.core.artifact import read_manifest

        build(list(job.inputs), job.out, splits=tuple(job.splits), **{**dict(job.params), "store": "noise"})
        return read_manifest(job.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="lerobot/smolvla_libero")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["search", "search_val"])
    ap.add_argument("--task-prefix")
    ap.add_argument("--store", choices=["actions", "noise"], default="actions",
                    help="what a segment keeps: the executed chunk (action memory) or its initial flow noise (noise prior)")
    a = ap.parse_args()
    from worldcurrent.core.artifact import SplitLeak, check_splits

    try:
        check_splits(a.splits)
    except SplitLeak as exc:
        ap.error(str(exc))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rep = build(a.runs, a.out, repo_id=a.repo_id, workers=a.workers, threads=a.threads, splits=tuple(a.splits),
                task_prefix=a.task_prefix, store=a.store)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("runs", "tasks")}, indent=2))


if __name__ == "__main__":
    main()
