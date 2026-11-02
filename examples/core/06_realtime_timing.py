"""Timing and the operator channel: what policy latency does to an episode, and a latched e-stop.

What it shows
    1. The kernel's two timing modes on MuJoCo planar reach, with an agent that plans 10-step action chunks:
           sync       the world waits for every policy call (what benchmarks usually measure)
           realtime   the robot keeps moving while the policy thinks: a chunk computed from the observation at
                      step s that arrives at step t has its first t - s actions dropped, and an empty queue
                      executes a hold action; on the virtual clock the latency is fixed (`timing.latency_s`),
                      so the run is deterministic
       swept over policy latencies from 0 to 0.8 s (0 to 16 control steps at 20 Hz). Measured: success, steps,
       hold steps, dropped actions and watchdog aborts. The chunk horizon is 10 steps, so a latency of more than
       10 steps leaves nothing to execute.
    2. The operator channel: an e-stop posted at step 10 (from a trace-event subscriber, as a workcell UI would)
       ends the episode before another action is sent; the fault is latched, so the next episode is refused until
       `clear_fault`.

Command (from the harness repository root)
    uv run python examples/core/06_realtime_timing.py [--quick] [--out DIR]

Typical runtime: about 40 s (--quick: about 2 s) on a 128-core CPU, no GPU.
Requires: hylo-env-mujoco (hylo-bench), matplotlib for the figure (optional).
Output: <out>/result.json, <out>/latency_sweep.png, one run directory per timing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.config import ExperimentConfig
from hylo.core.errors import SafetyAbort
from hylo.core.kernel import EpisodeRunner, OperatorChannel
from hylo.core.registry import registry
from hylo.core.trace import TraceStore
from hylo.core.types import ActionChunk, ActionSemantics, Observation
from hylo.eval.experiment import run_experiment

SLUG = "realtime_timing"
ROOT = Path(__file__).resolve().parents[2]
L1, L2, MAX_STEP = 0.3, 0.25, 0.05


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


class IKChunkAgent(AgentBase):
    """Analytic IK, then a P-controller rolled out on a kinematic model for `horizon` steps: an action chunk."""

    def __init__(self, kp: float = 0.3, horizon: int = 10) -> None:
        self.kp, self.horizon = kp, horizon

    def spec(self) -> AgentSpec:
        return AgentSpec(name="examples.ik_chunk", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"), action_dim=2)

    def act(self, observation: Observation) -> ActionChunk:
        x, y = observation.state["target_pos"]
        c2 = np.clip((x * x + y * y - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
        q2 = np.arccos(c2)
        goal = np.array([np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)), q2])
        q, chunk = np.array(observation.state["qpos"], dtype=float), []
        for _ in range(self.horizon):
            err = (goal - q + np.pi) % (2 * np.pi) - np.pi
            a = np.clip(self.kp * err / MAX_STEP, -1.0, 1.0)
            chunk.append(a)
            q = q + a * MAX_STEP
        return ActionChunk(actions=np.array(chunk), execution_steps=self.horizon, provenance={"model": "ik_chunk"})


registry.register("agents", "examples.ik_chunk")(IKChunkAgent)


def measure(run_dir: Path, log: dict) -> dict:
    store = TraceStore(run_dir)
    holds, dropped, chunks, aborts = [], [], [], 0
    for s in log["samples"]:
        tr = store.read(s["episode_id"])
        end = tr.last("episode.end")["payload"]
        holds.append(end.get("holds", 0))
        chunks.append(end.get("chunks", sum(1 for _ in tr.of_type("agent.chunk"))))
        dropped.append(sum(e["payload"]["provenance"].get("dropped", 0) for e in tr.of_type("agent.chunk")))
        aborts += end["status"] == "error"
    # success_mean is None when no episode could be scored (every one ended in a watchdog abort): count it as 0
    return {"episodes": len(log["samples"]), "success": log["results"]["success_mean"] or 0.0,
            "mean_steps": log["results"]["steps_mean"], "holds_per_episode": float(np.mean(holds)),
            "dropped_per_episode": float(np.mean(dropped)), "chunks_per_episode": float(np.mean(chunks)),
            "watchdog_aborts": aborts}


def plot(rows: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rt = [(v["latency_s"], v) for k, v in rows.items() if k != "sync"]
    lat = np.array([x for x, _ in rt]) * 20
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax[0].plot(lat, [v["success"] for _, v in rt], "o-", color="#2a6fdb", lw=1.4, ms=4, label="realtime")
    ax[0].axhline(rows["sync"]["success"], color="0.5", ls="--", lw=1, label="sync")
    ax[0].set_ylabel("success rate", fontsize=8)
    ax[0].set_ylim(-0.05, 1.05)
    ax[1].plot(lat, [v["mean_steps"] for _, v in rt], "o-", color="#2a6fdb", lw=1.4, ms=4, label="steps")
    ax[1].plot(lat, [v["holds_per_episode"] for _, v in rt], "s-", color="#c8442d", lw=1.2, ms=3.5, label="hold steps")
    ax[1].plot(lat, [v["dropped_per_episode"] for _, v in rt], "^-", color="0.45", lw=1.2, ms=3.5, label="dropped actions")
    ax[1].axhline(rows["sync"]["mean_steps"], color="0.5", ls="--", lw=1)
    ax[1].set_ylabel("per episode", fontsize=8)
    for a in ax:
        a.axvline(10, color="0.75", lw=0.8, ls=":")
        a.set_xlabel("policy latency (control steps at 20 Hz)", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(lw=0.3, alpha=0.5)
        a.legend(fontsize=7)
    fig.suptitle("Realtime timing, 10-step chunks: dotted line = chunk horizon", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return True


def estop_demo(out: Path) -> dict:
    env = registry.build("envs", {"type": "mujoco.planar_reach"})
    task = registry.build("benchmarks", {"type": "mujoco/reach-v0"}).instances("validation")[10]   # a random target
    op = OperatorChannel()
    shutil.rmtree(out / "estop", ignore_errors=True)            # fixed episode ids: start from an empty trace dir
    store = TraceStore(out / "estop")
    runner = EpisodeRunner(env, IKChunkAgent(), trace=store, operator=op)

    pressed = []

    def ui(event) -> None:                                   # what a workcell UI subscribed to the event bus would do
        if event.type == "step.result" and event.payload["step"] == 10 and not pressed:
            pressed.append(event.episode_id)
            op.post({"type": "estop", "by": "example"})
    runner.bus.subscribe(ui)
    first = runner.run(task, episode_id="estop-1")
    actions_after = sum(1 for e in store.read("estop-1").of_type("action")) - 10
    try:
        runner.run(task, episode_id="estop-2")
        latched = "not refused"
    except SafetyAbort as exc:
        latched = f"refused: {exc}"
    op.post({"type": "clear_fault", "by": "example"})
    again = runner.run(task, episode_id="estop-3")
    store.close()
    env.close()
    return {"first": {"status": first.outcome.status, "steps": first.outcome.steps,
                      "reason": first.outcome.termination_reason, "actions_after_estop": actions_after},
            "while_latched": latched, "after_clear_fault": {"status": again.outcome.status, "steps": again.outcome.steps}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes and latencies (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    a = ap.parse_args()
    try:
        registry.reference("envs", "mujoco.planar_reach")
    except KeyError:
        sys.exit("'mujoco.planar_reach' is not registered: install hylo-env-mujoco (hylo-bench) into this venv")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_val, workers = (5, 4) if a.quick else (50, 8)
    latencies = (0.0, 0.2, 0.4, 0.6) if a.quick else tuple(round(0.05 * k, 2) for k in range(0, 17))

    print(f"== 1. sync vs realtime over policy latency ({n_val} seeds x 2 tasks each, chunk horizon 10)")
    timings = {"sync": {"mode": "sync"}}
    timings.update({f"realtime {lat:.2f} s": {"mode": "realtime", "clock": "virtual", "latency_s": lat,
                                               "max_hold_steps": 40} for lat in latencies})
    rows, t0 = {}, time.perf_counter()
    for name, timing in timings.items():
        cfg = ExperimentConfig(name="core-timing", benchmark={"type": "mujoco/reach-v0", "params": {"validation": n_val}},
                               env={"type": "mujoco.planar_reach"}, agent={"type": "examples.ik_chunk", "params": {}},
                               timing=timing, workers=workers)
        res = run_experiment(cfg, run_root=out / "runs", label=name.replace(" ", "-"))
        rows[name] = {"timing": timing, "latency_s": timing.get("latency_s"), "config_digest": cfg.digest(),
                      **measure(res.run_dir, res.log)}
    wall = time.perf_counter() - t0
    table([[n, "-" if r["latency_s"] is None else f"{r['latency_s'] * 20:.0f}", f"{r['success']:.2f}",
            f"{r['mean_steps']:.1f}", f"{r['chunks_per_episode']:.1f}", f"{r['holds_per_episode']:.1f}",
            f"{r['dropped_per_episode']:.1f}", r["watchdog_aborts"]] for n, r in rows.items()],
          ["timing", "latency (steps)", "success", "steps", "chunks", "holds", "dropped", "watchdog aborts"])
    print(f"  ({wall:.1f} s; holds, dropped and chunks per episode)")

    print("\n== 2. e-stop through the operator channel")
    es = estop_demo(out)
    table([["e-stop posted after step 10", f"{es['first']['status']} at step {es['first']['steps']}; "
                                            f"actions sent after the e-stop: {es['first']['actions_after_estop']}"],
           ["next episode, fault latched", es["while_latched"]],
           ["after clear_fault", f"{es['after_clear_fault']['status']} in {es['after_clear_fault']['steps']} steps"]],
          ["operator input", "measured"])

    fig = plot(rows, out / "latency_sweep.png")
    result = {"example": SLUG, "quick": a.quick, "episodes_per_timing": 2 * n_val, "timings": rows, "estop": es,
              "figure": str(out / "latency_sweep.png") if fig else None}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {out / 'result.json'}" + (f"\nfigure: {out / 'latency_sweep.png'}" if fig else ""))
    ok = es["first"]["status"] == "safety_abort" and es["first"]["actions_after_estop"] == 0 \
        and es["while_latched"].startswith("refused") and es["after_clear_fault"]["status"] == "success"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
