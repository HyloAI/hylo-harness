"""First episode through the kernel: one episode by hand, then a whole benchmark split as an experiment.

What it shows
    1. The kernel loop by hand: an environment, an agent and a benchmark built from the plugin registry by name,
       one `EpisodeRunner.run()` writing an event-sourced trace, and what that trace contains.
    2. The same thing as an experiment: an `ExperimentConfig` run through `run_experiment` (what `hylo run CONFIG`
       does), its metrics, its config digest and run manifest, and `hylo trace` on the result.
    3. Reproducibility: the experiment run a second time gives the same status and step count on every episode.
    4. A figure drawn from the traces only (no simulator): the fingertip path of every episode.

Command (from the harness repository root)
    uv run python examples/core/01_first_episode.py [--quick] [--out DIR]

Typical runtime: about 4 s (--quick: about 2 s) on a 128-core CPU, no GPU.
Requires: hylo-env-mujoco (hylo-bench) for `mujoco.planar_reach`, matplotlib for the figure (optional).
Output: <out>/result.json, <out>/tip_paths.png, the run directories and traces under <out>.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hylo.core.config import ExperimentConfig
from hylo.core.kernel import EpisodeRunner
from hylo.core.registry import registry
from hylo.core.trace import TraceStore
from hylo.eval.experiment import run_experiment

SLUG = "first_episode"
ROOT = Path(__file__).resolve().parents[2]          # the harness repository


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def require(group: str, name: str) -> None:
    try:
        registry.reference(group, name)
    except KeyError:
        sys.exit(f"{name!r} is not registered: install hylo-env-mujoco (hylo-bench) into this venv")


def config(n_val: int, workers: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="core-first-episode",
        benchmark={"type": "mujoco/reach-v0", "split": "validation", "params": {"validation": n_val}},
        env={"type": "mujoco.planar_reach"},
        agent={"type": "baseline.planar_ik", "params": {"kp": 0.3}},
        seed=0, workers=workers, tags=("examples", "core"))


def episode_by_hand(out: Path) -> dict:
    """The kernel loop with no experiment machinery around it."""
    env = registry.build("envs", {"type": "mujoco.planar_reach"})
    agent = registry.build("agents", {"type": "baseline.planar_ik", "params": {"kp": 0.3}})
    bench = registry.build("benchmarks", {"type": "mujoco/reach-v0"})
    task = bench.instances("validation")[0]                     # planar_reach_fixed, seed 10000
    shutil.rmtree(out / "by_hand", ignore_errors=True)          # a fixed episode id: start from an empty trace dir
    store = TraceStore(out / "by_hand")
    t0 = time.perf_counter()
    rec = EpisodeRunner(env, agent, trace=store).run(task, episode_id="by-hand-0")
    wall = time.perf_counter() - t0
    store.close()
    tr = store.read("by-hand-0")
    dist = tr.last("step.result")["payload"]["info"]["distance"]
    return {"task": task.spec.id, "seed": task.seed, "status": rec.outcome.status, "steps": rec.outcome.steps,
            "final_distance_m": round(float(dist), 5), "safety_interventions": rec.safety_interventions,
            "events": rec.events, "events_by_type": dict(Counter(e["type"] for e in tr.events)),
            "guards": tr.first("episode.start")["payload"]["guards"], "wall_s": round(wall, 4)}


def plot_paths(run_dir: Path, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    store = TraceStore(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    log = json.loads((run_dir / "evallog.json").read_text())
    for ax, task in zip(axes, ("mujoco/planar_reach_fixed", "mujoco/planar_reach_random")):
        ax.add_patch(plt.Circle((0, 0), 0.55, fill=False, ls=":", lw=0.8, color="0.6"))
        for s in (s for s in log["samples"] if s["task_id"] == task):
            tr = store.read(s["episode_id"])
            tip = np.array([e["payload"]["state"]["tip_pos"] for e in tr.of_type("observation")])
            goal = tr.first("observation")["payload"]["state"]["target_pos"]
            ok = s["status"] == "success"
            ax.plot(tip[:, 0], tip[:, 1], lw=0.9, alpha=0.75, color="#2a6fdb" if ok else "#c8442d")
            ax.plot(*tip[0], "o", ms=2.5, color="0.35")
            ax.plot(*goal, "x", ms=5, mew=1.4, color="#c8442d")
        ax.set_title(task.split("/")[1], fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("y (m)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("x (m)", fontsize=8)
    fig.suptitle("Fingertip paths, drawn from the traces only (x = target, dot = start)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    a = ap.parse_args()
    require("envs", "mujoco.planar_reach")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_val, workers = (5, 4) if a.quick else (50, 8)

    print("== 1. one episode through the kernel")
    hand = episode_by_hand(out)
    table([[hand["task"], hand["seed"], hand["status"], hand["steps"], hand["final_distance_m"], hand["events"],
            hand["wall_s"]]], ["task", "seed", "status", "steps", "final dist (m)", "trace events", "wall (s)"])
    print("  guards installed by the kernel:", " -> ".join(hand["guards"]))
    print("  events per type:", json.dumps(hand["events_by_type"]))

    print(f"\n== 2. experiment: {n_val} seeds x 2 tasks, {workers} workers")
    cfg = config(n_val, workers)
    t0 = time.perf_counter()
    res = run_experiment(cfg, run_root=out / "runs")
    wall = time.perf_counter() - t0
    m = res.metrics
    by_task: dict[str, list[dict]] = {}
    for s in res.log["samples"]:
        by_task.setdefault(s["task_id"], []).append(s)
    rows = []
    for task, ss in by_task.items():
        steps = [s["steps"] for s in ss]
        rows.append([task, len(ss), f"{np.mean([s['status'] == 'success' for s in ss]):.2f}",
                     f"{np.mean(steps):.1f}", f"{min(steps)}-{max(steps)}"])
    table(rows, ["task", "episodes", "success", "mean steps", "range"])
    manifest = json.loads((res.run_dir / "manifest.json").read_text())
    print(f"  success_mean={m['success_mean']:.3f}  steps_mean={m['steps_mean']:.2f}  "
          f"safety_interventions_mean={m['safety_interventions_mean']}  wall={wall:.2f}s")
    print(f"  config digest {cfg.digest()} (manifest: {manifest['config_digest']}), trace schema {manifest['trace_schema']}")

    first = res.log["samples"][0]["episode_id"]
    cli = subprocess.run([sys.executable, "-m", "hylo.core.cli", "trace", str(res.run_dir), first],
                         capture_output=True, text=True, check=True)
    print(f"\n  $ hylo trace <run_dir> {first}\n" + "\n".join("    " + ln for ln in cli.stdout.splitlines()[:3]))

    print("\n== 3. reproducibility: the same config again")
    res2 = run_experiment(cfg, run_root=out / "runs")
    key = lambda r: [(s["key"], s["status"], s["steps"]) for s in r.log["samples"]]  # noqa: E731
    same = key(res) == key(res2)
    max_dev = 0.0                                               # executed actions, episode by episode
    for s1, s2 in zip(res.log["samples"], res2.log["samples"]):
        acts = [np.array([e["payload"]["data"] for e in TraceStore(r.run_dir).read(s["episode_id"]).of_type("action")])
                for r, s in ((res, s1), (res2, s2))]
        max_dev = max(max_dev, float(np.max(np.abs(acts[0] - acts[1]))) if acts[0].shape == acts[1].shape else np.inf)
    print(f"  identical status and steps on all {len(res.log['samples'])} episodes: {same}; "
          f"max |action difference| = {max_dev:g}")
    same = same and max_dev == 0.0

    fig = plot_paths(res.run_dir, out / "tip_paths.png")
    result = {
        "example": SLUG, "quick": a.quick, "by_hand": hand,
        "experiment": {"config_digest": cfg.digest(), "run_dir": str(res.run_dir), "wall_s": round(wall, 3),
                       "metrics": m, "per_task": {r[0]: {"episodes": r[1], "success": float(r[2]),
                                                         "mean_steps": float(r[3])} for r in rows}},
        "reproducible": same, "max_action_difference_between_runs": max_dev, "figure": str(out / "tip_paths.png") if fig else None,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {out / 'result.json'}" + (f"\nfigure: {out / 'tip_paths.png'}" if fig else ""))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
