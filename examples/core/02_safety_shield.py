"""SafetyShield: an unsafe action stream with and without limits, and why the shield cannot be bypassed.

What it shows
    An aggressive, noisy controller on MuJoCo planar reach: proportional IK control that saturates far from the target, Gaussian
    noise and occasional spikes up to 5x the action bounds, never clipped by the agent itself. The same agent and
    the same seeds run under these safety configs:
        mandatory   `safety: {}`: only the guards the kernel always installs (finite check, bounds clamp)
        rate limit  `safety: {max_delta: 0.25}`: the per-step change of the action is limited
        norm+rate   guard plugins `norm_limit` (|a| <= 1.0) then `delta_limit` (0.25), in that order
    (plus `max_delta` with `norm_limit` in the wrong order: the shorthand always runs first, so the norm scaling
    after it can re-introduce jumps larger than the rate limit)
    and a well-behaved agent (baseline.planar_ik) runs under the rate limit as a control. From the traces it
    measures interventions, the largest proposed and executed action, the largest step-to-step change of the
    executed action and the peak joint speed and acceleration the simulator reached.

    Then four attempts to get around the shield, each measured:
        a guard plugin that multiplies every action by 10      -> the bounds clamp still runs last
        an agent that emits NaN                                -> the episode aborts; the env never sees it
        guard order from config                                -> `finite` is always first, `bounds` always last
        the same stream on a real robot (capability flag)      -> the first intervention aborts the episode

Command (from the harness repository root)
    uv run python examples/core/02_safety_shield.py [--quick] [--out DIR]

Typical runtime: about 12 s (--quick: about 3 s) on a 128-core CPU, no GPU.
Requires: hylo-env-mujoco (hylo-bench), matplotlib for the figure (optional).
Output: <out>/result.json, <out>/shield_effect.png, one run directory per safety config.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.config import ExperimentConfig
from hylo.core.errors import SafetyAbort
from hylo.core.kernel import EpisodeRunner
from hylo.core.registry import registry
from hylo.core.safety import GuardVerdict, SafetyShield, build_guards
from hylo.core.trace import TraceStore
from hylo.core.types import Action, ActionChunk, ActionSemantics, Observation
from hylo.eval.experiment import run_experiment

SLUG = "safety_shield"
ROOT = Path(__file__).resolve().parents[2]
L1, L2, MAX_STEP = 0.3, 0.25, 0.05                  # mujoco.planar_reach geometry and joint step at |a| = 1


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


# ------------------------------------------------------------------ an unsafe agent and a hostile guard

class JitteryIKAgent(AgentBase):
    """Analytic IK + P control with a high gain, noise and spikes; it does not clip its own output."""

    def __init__(self, kp: float = 0.08, noise: float = 0.3, spike_p: float = 0.05, spike: float = 5.0,
                 nan_at_step: int | None = None) -> None:
        self.kp, self.noise, self.spike_p, self.spike, self.nan_at_step = kp, noise, spike_p, spike, nan_at_step

    def spec(self) -> AgentSpec:
        return AgentSpec(name="examples.jittery_ik", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"), action_dim=2)

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.rng, self.t = np.random.default_rng(ctx.seed), 0           # seeded: the same stream every run

    def act(self, observation: Observation) -> ActionChunk:
        x, y = observation.state["target_pos"]
        c2 = np.clip((x * x + y * y - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
        q2 = np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        err = (np.array([q1, q2]) - observation.state["qpos"] + np.pi) % (2 * np.pi) - np.pi
        a = self.kp * err / MAX_STEP + self.rng.normal(0.0, self.noise, 2)
        if self.rng.random() < self.spike_p:
            a = a + self.rng.choice([-1.0, 1.0], 2) * self.spike
        if self.nan_at_step is not None and self.t == self.nan_at_step:
            a = np.array([np.nan, 0.0])
        self.t += 1
        return ActionChunk.single(a, model="jittery_ik")


class AmplifyGuard:
    """A hostile guard plugin: tries to push every action out of bounds."""

    name = "amplify_x10"

    def review(self, data, ctx) -> GuardVerdict:
        return GuardVerdict(data * 10.0, True, "multiplied by 10")


registry.register("agents", "examples.jittery_ik")(JitteryIKAgent)
registry.register("guards", "examples.amplify_x10")(AmplifyGuard)


# ------------------------------------------------------------------ measuring a run from its traces

def measure(run_dir: Path, log: dict, hz: float = 20.0) -> dict:
    store = TraceStore(run_dir)
    prop, appr, dact, qvel, acc, modified, steps, interventions = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, []
    for s in log["samples"]:
        tr = store.read(s["episode_id"])
        rev = list(tr.of_type("safety.review"))
        acts = np.array([e["payload"]["data"] for e in tr.of_type("action")])
        q = np.array([e["payload"]["state"]["qvel"] for e in tr.of_type("observation")])
        prop = max(prop, max(float(np.max(np.abs(r["payload"]["proposed"]))) for r in rev))
        appr = max(appr, float(np.max(np.abs(acts))))
        if len(acts) > 1:
            dact = max(dact, float(np.max(np.abs(np.diff(acts, axis=0)))))
        qvel = max(qvel, float(np.max(np.abs(q))))
        acc = max(acc, float(np.max(np.abs(np.diff(q, axis=0)))) * hz)       # rad/s^2 over one control step
        modified += sum(r["payload"]["modified"] for r in rev)
        steps += len(rev)
        interventions.append(tr.last("episode.end")["payload"]["safety_interventions"])
    return {"episodes": len(log["samples"]), "success": log["results"]["success_mean"],
            "mean_steps": log["results"]["steps_mean"], "interventions_per_episode": float(np.mean(interventions)),
            "modified_step_fraction": modified / max(steps, 1), "max_abs_proposed": prop, "max_abs_executed": appr,
            "max_step_change_executed": dact, "peak_joint_speed_rad_s": qvel,
            "peak_joint_accel_rad_s2": acc}


def plot(runs: dict[str, Path], logs: dict[str, dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.6), sharex=True)
    for col, name in enumerate(("mandatory", "rate limit")):
        tr = TraceStore(runs[name]).read(logs[name]["samples"][-1]["episode_id"])
        rev = list(tr.of_type("safety.review"))
        p = np.array([r["payload"]["proposed"][0] for r in rev])
        a = np.array([r["payload"]["approved"][0] for r in rev])
        q = np.array([e["payload"]["state"]["qvel"][0] for e in tr.of_type("observation")])[1:]
        ax = axes[0, col]
        ax.plot(p, lw=0.8, color="0.6", label="proposed by the agent")
        ax.plot(a, lw=1.3, color="#2a6fdb", label="executed (after the shield)")
        ax.axhspan(-1, 1, color="#2a6fdb", alpha=0.06)
        ax.set_ylim(-6.5, 6.5)
        ax.set_title(f"{name}: joint 1 action", fontsize=9)
        axes[1, col].plot(q, lw=1.1, color="#c8442d")
        axes[1, col].set_title(f"{name}: joint 1 speed (rad/s)", fontsize=9)
        axes[1, col].set_xlabel("control step", fontsize=8)
        for x in (ax, axes[1, col]):
            x.tick_params(labelsize=7)
            x.grid(lw=0.3, alpha=0.5)
    lim = max(abs(np.array(axes[1, 0].get_ylim())).max(), abs(np.array(axes[1, 1].get_ylim())).max())
    for ax in axes[1]:
        ax.set_ylim(-lim, lim)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Same agent, same seed: mandatory guards only vs a 0.25 rate limit", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ bypass attempts

def bypass_attempts(env, task, recorded_proposals: np.ndarray, trace_dir: Path) -> dict:
    out: dict = {}
    shutil.rmtree(trace_dir, ignore_errors=True)                # fixed episode ids: start from an empty trace dir
    rec_store = TraceStore(trace_dir)
    # 1. a hostile guard plugin between the mandatory guards
    runner = EpisodeRunner(env, JitteryIKAgent(), trace=rec_store,
                           safety={"guards": [{"type": "examples.amplify_x10"}]})
    rec = runner.run(task, episode_id="amplify")
    acts = np.array([e["payload"]["data"] for e in rec_store.read("amplify").of_type("action")])
    out["hostile_guard"] = {"guards": runner.shield.guard_names, "status": rec.outcome.status,
                            "max_abs_executed": float(np.max(np.abs(acts)))}
    # 2. NaN from the agent
    rec = EpisodeRunner(env, JitteryIKAgent(nan_at_step=10), trace=rec_store).run(task, episode_id="nan")
    tr = rec_store.read("nan")
    acts = np.array([e["payload"]["data"] for e in tr.of_type("action")])
    out["nan_action"] = {"status": rec.outcome.status, "steps_executed": rec.outcome.steps,
                         "reason": rec.outcome.termination_reason, "all_executed_finite": bool(np.all(np.isfinite(acts)))}
    # 3. guard order, whatever the config lists
    orders = {}
    for name, cfg in (("{}", {}), ("max_delta", {"max_delta": 0.25}),
                      ("norm_limit + max_delta", {"max_delta": 0.25, "guards": [{"type": "norm_limit", "params": {"max_norm": 1.0}}]})):
        orders[name] = SafetyShield(env.spec(), build_guards(cfg)).guard_names
    out["guard_order"] = orders
    # 4. the recorded proposals replayed through a shield for a real robot
    spec = env.spec()
    robot = spec.model_copy(update={"capabilities": spec.capabilities.model_copy(update={"real_robot": True})})
    shield, abort_at = SafetyShield(robot, build_guards({"max_delta": 0.25})), None
    for i, p in enumerate(recorded_proposals):
        try:
            shield.review(Action(data=p))
        except SafetyAbort as exc:
            abort_at = {"step": i, "reason": str(exc)}
            break
    out["real_robot"] = {"proposals": len(recorded_proposals), "aborted_at": abort_at}
    rec_store.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    a = ap.parse_args()
    try:
        registry.reference("envs", "mujoco.planar_reach")
    except KeyError:
        sys.exit("'mujoco.planar_reach' is not registered: install hylo-env-mujoco (hylo-bench) into this venv")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_val, workers = (5, 4) if a.quick else (50, 8)

    # guards run in the order listed; the rate limit goes last so nothing after it can re-introduce a jump
    strict = {"guards": [{"type": "norm_limit", "params": {"max_norm": 1.0}},
                         {"type": "delta_limit", "params": {"max_delta": 0.25}}]}
    jittery = {"type": "examples.jittery_ik", "params": {}}
    variants = {
        "mandatory": (jittery, {}),
        "rate limit": (jittery, {"max_delta": 0.25}),
        "norm+rate": (jittery, strict),
        "rate+norm (wrong order)": (jittery, {"max_delta": 0.25,                  # the shorthand is always first
                                              "guards": [{"type": "norm_limit", "params": {"max_norm": 1.0}}]}),
        "control: planar_ik": ({"type": "baseline.planar_ik", "params": {"kp": 0.3}}, {"max_delta": 0.25}),
    }
    print(f"== 1. one unsafe agent, four safety configs ({n_val} seeds x 2 tasks each)")
    results, runs, logs = {}, {}, {}
    t0 = time.perf_counter()
    for name, (agent, safety) in variants.items():
        cfg = ExperimentConfig(name="core-shield-" + re.sub(r"[^a-z0-9]+", "-", name.split(":")[0]).strip("-"),
                               benchmark={"type": "mujoco/reach-v0", "params": {"validation": n_val}},
                               env={"type": "mujoco.planar_reach"}, agent=agent, safety=safety, workers=workers)
        res = run_experiment(cfg, run_root=out / "runs")
        runs[name], logs[name] = res.run_dir, res.log
        results[name] = {"safety": safety, **measure(res.run_dir, res.log)}
    wall = time.perf_counter() - t0
    table([[n, f"{r['success']:.2f}", f"{r['mean_steps']:.1f}", f"{r['interventions_per_episode']:.1f}",
            f"{r['modified_step_fraction']:.2f}", f"{r['max_abs_proposed']:.2f}", f"{r['max_abs_executed']:.2f}",
            f"{r['max_step_change_executed']:.2f}", f"{r['peak_joint_speed_rad_s']:.2f}",
            f"{r['peak_joint_accel_rad_s2']:.1f}"] for n, r in results.items()],
          ["config", "success", "steps", "interv./ep", "modified", "max|prop|", "max|exec|", "max|da|", "peak qvel",
           "peak accel"])
    print(f"  ({wall:.1f} s; max|da| = largest change of the executed action between two steps; joint speed in rad/s, "
          "acceleration in rad/s^2)")

    print("\n== 2. attempts to get around the shield")
    env = registry.build("envs", {"type": "mujoco.planar_reach"})
    task = registry.build("benchmarks", {"type": "mujoco/reach-v0"}).instances("validation")[0]
    first = TraceStore(runs["mandatory"]).read(logs["mandatory"]["samples"][0]["episode_id"])
    proposals = np.array([r["payload"]["proposed"] for r in first.of_type("safety.review")])
    by = bypass_attempts(env, task, proposals, out / "bypass")
    env.close()
    hg, nan, rr = by["hostile_guard"], by["nan_action"], by["real_robot"]
    table([
        ["guard plugin multiplies actions by 10", f"max |executed| = {hg['max_abs_executed']:.2f}; order {' -> '.join(hg['guards'])}"],
        ["agent emits NaN at step 10", f"{nan['status']} after {nan['steps_executed']} steps; executed actions all finite: {nan['all_executed_finite']}"],
        ["config lists extra guards", " -> ".join(by["guard_order"]["norm_limit + max_delta"])],
        ["same stream, real_robot=True", (f"aborted at step {rr['aborted_at']['step']} of {rr['proposals']}: "
                                          f"{rr['aborted_at']['reason'][:60]}") if rr["aborted_at"] else "never aborted"],
    ], ["attempt", "measured"])

    fig = plot(runs, logs, out / "shield_effect.png")
    result = {"example": SLUG, "quick": a.quick, "episodes_per_config": 2 * n_val, "configs": results,
              "bypass": by, "figure": str(out / "shield_effect.png") if fig else None}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {out / 'result.json'}" + (f"\nfigure: {out / 'shield_effect.png'}" if fig else ""))
    ok = hg["max_abs_executed"] <= 1.0 and nan["status"] == "safety_abort" and nan["all_executed_finite"] \
        and all(v[0] == "finite" and v[-1] == "bounds" for v in by["guard_order"].values()) and rr["aborted_at"] is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
