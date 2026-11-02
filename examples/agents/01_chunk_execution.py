"""Chunk execution strategies: one learned policy, four ways to execute its action chunks.

A reference policy (policy.bc, chunk of 8 actions, trained from the IK expert's runs on MuJoCo planar reach) is run
by the same agents with different execution strategies, all selected by configuration:

- open loop:          agent "policy", execution_steps 8 (query once, run the whole chunk)
- replan interval:    agent "policy", execution_steps 4, 2, 1 (receding horizon: run a prefix, re-query)
- temporal ensemble:  agent "temporal_ensemble" (ACT): query every step, execute the exponentially weighted
                      average of every chunk that covers the current step (m = 0.01, the ACT default, and 0.5)

The same is repeated for the stochastic policy.flow (flow matching from seeded noise), where averaging also
removes sampling noise. Measured on held-out init states never used for training: success (95% Wilson interval),
mean steps, policy queries per episode and smoothness (mean |a_t - a_(t-1)|, lower is smoother).

    uv run python examples/agents/01_chunk_execution.py [--quick] [--out DIR]

Runtime: about 50 s on 16 processes (quick: about 10 s). Requires hylo-env-mujoco (bench) for mujoco.planar_reach.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

SLUG = "chunk_execution"


def strategies(art: dict[str, Path]) -> dict[str, tuple[str, str, dict]]:
    out = {}
    for method in ("bc", "flow"):
        pol = C.policy_ref(method, art[method])
        for k in (8, 4, 2, 1):
            label = "open loop (8)" if k == 8 else f"replan every {k}"
            out[f"{method}/exec{k}"] = (method, label, {"type": "policy", "params": {"policy": pol, "execution_steps": k}})
        for m in (0.01, 0.5):
            out[f"{method}/te{m}"] = (method, f"temporal ensemble m={m}",
                                      {"type": "temporal_ensemble", "params": {"policy": pol, "m": m}})
    return out


def main() -> int:
    a = C.parse_args(__doc__, SLUG)
    t0 = time.perf_counter()
    states = 50 if a.quick else 250                        # heldout = 20% of init states, two tasks each
    bench = C.reach_bench(states)
    roots = C.demonstrations(a.out, bench, workers=a.workers)
    epochs = {"bc": 150, "flow": 300}
    art, training = {}, {}
    for method in ("bc", "flow"):
        art[method], training[method] = C.train(method, [roots["demos"]], a.out / "artifacts" / method,
                                                {"chunk": 8, "epochs": epochs[method]})
        C.log(f"trained policy.{method}: {training[method]['samples']} samples from {training[method]['episodes']} "
              f"demonstrations in {training[method]['seconds']} s")
    jobs = strategies(art)
    runs = C.run_many({k: v[2] for k, v in jobs.items()}, bench, "heldout", a.out / "runs", procs=a.workers)
    rows = []
    for key, (method, label, agent) in jobs.items():
        s = C.summarize(runs[key])
        vs = C.paired(C.per_key(runs[key]), C.per_key(runs[f"{method}/exec8"]))
        rows.append({"policy": method, "strategy": label, "agent": agent["type"], **s, "vs_open_loop": vs,
                     "vs open loop": "-" if key.endswith("exec8") else vs["text"],
                     "success_%": round(100 * s["success"], 1), "ci95_%": f"{100 * s['ci95'][0]:.0f}-{100 * s['ci95'][1]:.0f}",
                     "runs": [str(d) for d in runs[key]]})
    print()
    C.table(rows, ["policy", "strategy", "success_%", "ci95_%", "steps", "queries_per_ep", "jerk", "policy_ms",
             "vs open loop"],
            title=f"MuJoCo planar reach, held-out split ({rows[0]['episodes']} episodes per row)")
    C.write_result(a.out, {"example": f"agents/{SLUG}", "quick": a.quick, "benchmark": bench, "split": "heldout",
                           "training": training, "rows": rows, "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
