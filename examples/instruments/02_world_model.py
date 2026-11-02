"""World model as an environment: fit a state world model from runs, rank controllers in it, check against MuJoCo.

1. Runs on the development splits: the IK controller (demonstrations) and random actions (exploration).
2. The `world_model.state` learner (a `hylo learn` job) fits an ensemble of MLPs for the next state given (state,
   action), and a classifier for "task complete" in a state. Its multi-step error is measured by feeding recorded
   actions of unseen episodes (search_val) through the model open loop.
3. `world_model` is an environment plugin: the unchanged kernel runs any state-based agent inside the model,
   `env: {type: world_model, params: {path: ...}}`. Eight controller candidates (IK gains from 0.01 to 1.0, random
   actions) are evaluated in the model and in MuJoCo on the same split; the model's ranking is compared with the true
   one (Spearman rank correlation on success and on mean steps), which is what pre-screening in a campaign relies on.
   A second pass applies the simulator's success predicate to the model's predicted states (model_pred columns), to
   separate errors of the dynamics from errors of the learned completion classifier.

    uv run python examples/instruments/02_world_model.py [--quick] [--out DIR]

Runtime: about 1 min on 16 processes (quick: about 10 s). Requires hylo-env-mujoco (bench) and hylo-learn; the
figure (OUT/world_model.png) needs matplotlib.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

import numpy as np  # noqa: E402

SLUG = "world_model"
GAINS = (0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0)
HORIZONS = (1, 5, 10, 20, 40)


def rollout_error(wm_path: Path, demos: Path, max_episodes: int) -> dict[int, dict[str, float]]:
    """Open-loop fingertip error of the model after k steps, from the first state of unseen expert episodes, against
    a constant-state baseline (predict no motion)."""
    from hylo.core.trace import TraceStore
    from hylo.instruments.world_model import StateWorldModel, _vec

    wm = StateWorldModel.load(wm_path)
    i0 = 0
    for k, n in zip(wm.keys, wm.dims):
        if k == "tip_pos":
            break
        i0 += n
    err: dict[int, list[float]] = {h: [] for h in HORIZONS}
    base: dict[int, list[float]] = {h: [] for h in HORIZONS}
    used = 0
    for rd in sorted(demos.iterdir()):
        if not rd.is_dir() or "-search_val-" not in rd.name:
            continue
        store = TraceStore(rd)
        for s in C.evallog(rd)["samples"]:
            if used >= max_episodes:
                break
            tr = store.read(s["episode_id"])
            states = [_vec(e["payload"]["state"], wm.keys) for e in tr.of_type("observation")]
            acts = [np.asarray(e["payload"]["data"], float).reshape(-1) for e in tr.of_type("action")]
            x = states[0]
            for t, act in enumerate(acts, 1):
                x, _ = wm.predict(x, act)
                if t in err and t < len(states):
                    err[t].append(float(np.linalg.norm(x[i0:i0 + 2] - states[t][i0:i0 + 2])))
                    base[t].append(float(np.linalg.norm(states[0][i0:i0 + 2] - states[t][i0:i0 + 2])))
            used += 1
    return {h: {"episodes": len(err[h]), "model_m": float(np.mean(err[h])) if err[h] else None,
                "no_motion_m": float(np.mean(base[h])) if base[h] else None} for h in HORIZONS}


def true_predicate(run_dirs: list[Path], radius: float = 0.02) -> dict[str, float]:
    """Success and steps when the simulator's own predicate (fingertip within 2 cm of the target) is applied to the
    model's predicted states, instead of the model's learned completion classifier."""
    from hylo.core.trace import TraceStore

    succ, steps = [], []
    for d, s in C.samples(run_dirs):
        obs = TraceStore(d).read(s["episode_id"]).of_type("observation")
        dist = [float(np.linalg.norm(np.asarray(e["payload"]["state"]["tip_pos"]) -
                                     np.asarray(e["payload"]["state"]["target_pos"]))) for e in obs][1:]
        hit = next((i + 1 for i, v in enumerate(dist) if v < radius), None)
        succ.append(hit is not None)
        steps.append(hit if hit is not None else len(dist))
    return {"success": float(np.mean(succ)), "steps": float(np.mean(steps))}


def figure(rows: list[dict], horizon: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), dpi=110)
    ax = axes[0]
    hs = [h for h in HORIZONS if horizon[h]["model_m"] is not None]
    ax.plot(hs, [100 * horizon[h]["model_m"] for h in hs], "o-", color="#c8442d", label="world model")
    ax.plot(hs, [100 * horizon[h]["no_motion_m"] for h in hs], "o--", color="#5b6168", label="no-motion baseline")
    ax.axhline(2.0, color="#3a7d44", lw=0.8, ls=":", label="success radius (2 cm)")
    ax.set_xlabel("open-loop steps")
    ax.set_ylabel("fingertip error (cm)")
    ax.set_title("Multi-step prediction (unseen episodes)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot([0, 150], [0, 150], color="#5b6168", lw=0.8, ls=":")
    ax.scatter([r["real_steps"] for r in rows], [r["model_pred_steps"] for r in rows], facecolors="none",
               edgecolors="#c8442d", s=22, label="model states, simulator's predicate")
    ax.scatter([r["real_steps"] for r in rows], [r["model_steps"] for r in rows], color="#2a6f97", s=18,
               label="model, learned completion classifier")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for r in rows:
        ax.annotate(r["candidate"].replace("IK kp=", "kp "), (r["real_steps"], r["model_steps"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean steps in MuJoCo")
    ax.set_ylabel("mean steps in the world model")
    ax.set_title("Candidates: model vs simulator", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    a = C.parse_args(__doc__, SLUG)
    t0 = time.perf_counter()
    states = 50 if a.quick else 250
    bench = C.reach_bench(states)
    roots = C.demonstrations(a.out, bench, workers=a.workers, explore=True)
    wm_path, info = C.train("world_model.state", [roots["demos"], roots["explore"]], a.out / "artifacts" / "world_model",
                            {"env": {"type": "mujoco.planar_reach"}, **({"epochs": 30} if a.quick else {})})
    man = json.loads((wm_path / "artifact.json").read_text())
    val = (man.get("metrics") or {}).get("val") or {}
    C.log(f"world model: {info['transitions']} transitions from {info['episodes']} episodes, {info['seconds']} s; "
          f"one-step val mse {val.get('mse', float('nan')):.2e} (no-change baseline {val.get('mse_zero_baseline', float('nan')):.2e})")
    horizon = rollout_error(wm_path, roots["demos"], 20 if a.quick else 100)

    candidates = {f"IK kp={k}": {"type": "baseline.planar_ik", "params": {"kp": k}} for k in GAINS}
    candidates["random actions"] = {"type": "baseline.random", "params": {"dim": 2, "seed": 0}}
    split = "gate"
    real = C.run_many(candidates, bench, split, a.out / "runs" / "mujoco", procs=a.workers)
    wm_env = {"type": "world_model", "params": {"path": str(wm_path)}}
    model = C.run_many(candidates, bench, split, a.out / "runs" / "world_model", procs=a.workers, env=wm_env)
    # the same model dynamics, judged by the simulator's predicate instead of the learned completion classifier
    wm_free = {"type": "world_model", "params": {"path": str(wm_path), "done_threshold": 2.0}}   # never terminates
    free = C.run_many(candidates, bench, split, a.out / "runs" / "world_model_free", procs=a.workers, env=wm_free)
    rows = []
    for name in candidates:
        r, m, f = C.summarize(real[name]), C.summarize(model[name]), true_predicate(free[name])
        rows.append({"candidate": name, "real_success_%": round(100 * r["success"], 1), "real_steps": r["steps"],
                     "model_success_%": round(100 * m["success"], 1), "model_steps": m["steps"],
                     "model_pred_success_%": round(100 * f["success"], 1), "model_pred_steps": f["steps"],
                     "episodes": r["episodes"]})
    rho_success = C.spearman([r["model_success_%"] for r in rows], [r["real_success_%"] for r in rows])
    rho_steps = C.spearman([r["model_steps"] for r in rows], [r["real_steps"] for r in rows])
    rho_pred = C.spearman([r["model_pred_steps"] for r in rows], [r["real_steps"] for r in rows])
    best_real = min(rows, key=lambda r: (-r["real_success_%"], r["real_steps"]))["candidate"]
    best_model = min(rows, key=lambda r: (-r["model_success_%"], r["model_steps"]))["candidate"]
    print()
    C.table([{"steps": h, **v} for h, v in horizon.items()], ["steps", "episodes", "model_m", "no_motion_m"],
            title="Open-loop fingertip error after k steps (m), unseen search_val episodes")
    print()
    C.table(rows, ["candidate", "real_success_%", "model_success_%", "model_pred_success_%", "real_steps",
                   "model_steps", "model_pred_steps"],
            title=f"Candidates on the {split} split ({rows[0]['episodes']} episodes each): MuJoCo vs world model")
    print(f"\nSpearman (model vs MuJoCo): success {rho_success if rho_success is None else round(rho_success, 3)}, "
          f"mean steps {rho_steps if rho_steps is None else round(rho_steps, 3)}; best in MuJoCo: {best_real}, "
          f"best in the model: {best_model}; with the simulator's predicate on model states: mean steps "
          f"{rho_pred if rho_pred is None else round(rho_pred, 3)}")
    fig = a.out / "world_model.png"
    has_fig = figure(rows, horizon, fig)
    C.write_result(a.out, {"example": f"instruments/{SLUG}", "quick": a.quick, "benchmark": bench, "split": split,
                           "world_model": {"path": str(wm_path), **info, "val": val}, "rollout_error": horizon,
                           "candidates": rows, "spearman_success": rho_success, "spearman_steps": rho_steps,
                           "spearman_steps_true_predicate": rho_pred,
                           "best_real": best_real, "best_model": best_model, "figure": str(fig) if has_fig else None,
                           "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
