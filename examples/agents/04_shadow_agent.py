"""Shadow agent: candidates run next to the acting agent on every decision, but never act.

The acting agent is the IK controller (baseline.planar_ik, the "deployed" agent). Each candidate is wrapped with it
in a `shadow` agent:

    agent: {type: shadow, params: {active: {type: baseline.planar_ik, ...}, shadow: {type: policy, ...},
                                   gripper_dim: null, background: false}}

The candidate sees every observation and step result, its chunk is compared with what was executed (an `x.shadow`
trace event per decision: mean / max absolute action difference, its latency), and nothing it outputs reaches the
robot. Measured on held-out MuJoCo planar reach episodes:

1. no actions sent: every shadow run's executed actions are identical to the IK controller's run alone;
2. disagreement per candidate (learned policies trained from the controller's runs, slower controllers, random);
3. whether disagreement predicts how the candidate does when it acts itself (rank correlation);
4. control-loop latency per decision with an expensive candidate (CEM MPC) run inline vs in the background.

    uv run python examples/agents/04_shadow_agent.py [--quick] [--out DIR]

Runtime: about 75 s on 16 processes (quick: about 15 s). Requires hylo-env-mujoco (bench).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

import numpy as np  # noqa: E402

SLUG = "shadow_agent"


def shadow_events(run_dirs: list[Path]) -> list[dict]:
    from hylo.core.trace import TraceStore

    out = []
    for d, s in C.samples(run_dirs):
        out += [e["payload"] for e in TraceStore(d).read(s["episode_id"]).of_type("x.shadow")]
    return out


def main() -> int:
    a = C.parse_args(__doc__, SLUG)
    t0 = time.perf_counter()
    states = 50 if a.quick else 250
    bench = C.reach_bench(states)
    roots = C.demonstrations(a.out, bench, workers=a.workers, explore=True)
    jobs = {m: (m, [roots["demos"]], a.out / "artifacts" / m, {"chunk": 8}) for m in ("knn", "bc", "flow")}
    jobs["world_model"] = ("world_model.state", [roots["demos"], roots["explore"]], a.out / "artifacts" / "world_model",
                           {"env": {"type": "mujoco.planar_reach"}, **({"epochs": 30} if a.quick else {})})
    art = {k: p for k, (p, _) in C.train_many(jobs, procs=a.workers).items()}
    mpc = {"type": "policy", "params": {"policy": {"type": "policy.mpc", "params": {
        "world_model": str(art["world_model"]), "objective": {"reach": ["tip_pos", "target_pos"]}}}}}
    candidates = {
        "policy.bc (from its runs)": {"type": "policy", "params": {"policy": C.policy_ref("bc", art["bc"])}},
        "policy.flow (from its runs)": {"type": "policy", "params": {"policy": C.policy_ref("flow", art["flow"])}},
        "policy.knn (from its runs)": {"type": "policy", "params": {"policy": C.policy_ref("knn", art["knn"])}},
        "IK controller kp=0.1": {"type": "baseline.planar_ik", "params": {"kp": 0.1}},
        "IK controller kp=0.03": {"type": "baseline.planar_ik", "params": {"kp": 0.03}},
        "random actions": {"type": "baseline.random", "params": {"dim": 2, "seed": 0}},
        "mpc-cem (world model)": mpc,
    }

    def shadow(cand: dict, background: bool = False) -> dict:
        return {"type": "shadow", "params": {"active": C.EXPERT, "shadow": cand, "gripper_dim": None,
                                             "background": background}}

    agents = {"active alone": C.EXPERT,
              **{f"shadow: {k}": shadow(v) for k, v in candidates.items()},
              **{f"acting: {k}": v for k, v in candidates.items()},
              "shadow (background): mpc-cem (world model)": shadow(mpc, background=True)}
    C.log(f"evaluating {len(agents)} agents on the held-out split ...")
    runs = C.run_many(agents, bench, "heldout", a.out / "runs", procs=a.workers)

    base = C.episode_actions(runs["active alone"])
    base_s = C.summarize(runs["active alone"])
    rows = []
    for name in candidates:
        sh = runs[f"shadow: {name}"]
        eps = C.episode_actions(sh)
        same = all(x["key"] == y["key"] and x["actions"].shape == y["actions"].shape
                   and np.array_equal(x["actions"], y["actions"]) for x, y in zip(eps, base)) and len(eps) == len(base)
        ev = [e for e in shadow_events(sh) if "error" not in e]
        acting = C.summarize(runs[f"acting: {name}"])
        rows.append({"candidate": name, "decisions": len(ev),
                     "actions_identical": same, "shadow_run_success_%": round(100 * C.summarize(sh)["success"], 1),
                     "mean_abs_diff": float(np.mean([e["mean_abs"] for e in ev])),
                     "p95_abs_diff": float(np.percentile([e["max_abs"] for e in ev], 95)),
                     "disagree_%": round(100 * float(np.mean([e["max_abs"] > 0.25 for e in ev])), 1),
                     "shadow_ms": 1000 * float(np.median([e["latency_s"] for e in ev])),
                     "acting_success_%": round(100 * acting["success"], 1), "acting_steps": acting["steps"]})
    rho = C.spearman([-r["mean_abs_diff"] for r in rows], [r["acting_success_%"] for r in rows])
    learned = [r for r in rows if r["candidate"].startswith("policy.")]
    rho_learned = C.spearman([-r["mean_abs_diff"] for r in learned], [r["acting_success_%"] for r in learned])
    rho_tail = C.spearman([-r["disagree_%"] for r in rows], [r["acting_success_%"] for r in rows])
    lat = {}
    for label, name in (("active alone", "active alone"), ("+ mpc shadow, inline", "shadow: mpc-cem (world model)"),
                        ("+ mpc shadow, background", "shadow (background): mpc-cem (world model)")):
        s = C.summarize(runs[name])
        lat[label] = {"control_ms_per_decision": s["policy_ms"], "success_%": round(100 * s["success"], 1)}
    print()
    C.table(rows, ["candidate", "actions_identical", "decisions", "mean_abs_diff", "p95_abs_diff", "disagree_%",
                   "shadow_ms", "acting_success_%", "acting_steps"],
            title=f"Shadow runs next to the IK controller (held-out, {base_s['episodes']} episodes; active alone: "
                  f"{100 * base_s['success']:.0f}% success, {base_s['steps']:.1f} steps)")
    print(f"\nrank correlation, -disagreement vs acting success: all candidates {rho if rho is None else round(rho, 2)}, "
          f"learned policies only {rho_learned if rho_learned is None else round(rho_learned, 2)}; "
          f"-disagreement rate vs acting success: {rho_tail if rho_tail is None else round(rho_tail, 2)}")
    print()
    C.table([{"agent": k, **v} for k, v in lat.items()], ["agent", "control_ms_per_decision", "success_%"],
            title="Control-loop latency per decision (what the robot waits for)")
    C.write_result(a.out, {"example": f"agents/{SLUG}", "quick": a.quick, "benchmark": bench, "split": "heldout",
                           "active": C.EXPERT, "active_alone": base_s, "rows": rows,
                           "spearman_neg_disagreement_vs_success": rho, "spearman_learned_only": rho_learned,
                           "spearman_neg_disagree_rate_vs_success": rho_tail,
                           "disagree_threshold": 0.25, "latency": lat,
                           "runs": {k: [str(d) for d in v] for k, v in runs.items()},
                           "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
