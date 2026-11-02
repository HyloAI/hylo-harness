"""Reference policies trained from your own runs: a leaderboard and a data-efficiency curve on MuJoCo reach.

1. Runs: the IK controller (baseline.planar_ik) on the development splits gives the demonstrations; random-action
   runs give the exploration data a world model needs.
2. Learners (`hylo learn` jobs, sealed artifacts): policy.knn, policy.bc, policy.flow, policy.tokens from the
   demonstrations, world_model.state for policy.mpc; each imitation learner also from 20, 60 and 120
   demonstrations only (balanced over the two tasks).
3. Every policy is run by the same agents (`policy`, ACT-style `temporal_ensemble`, `steered` with 4 seeded samples)
   on held-out init states that no learner saw: success with 95% Wilson intervals, mean steps, query latency.

    uv run python examples/agents/03_reference_policies.py [--quick] [--out DIR]

Runtime: about 70 s on 16 processes (quick: about 15 s, 2 data sizes, fewer episodes). Requires hylo-env-mujoco
(bench). The figure (OUT/learning_curve.png) needs matplotlib.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

SLUG = "reference_policies"
FAMILIES = {"knn": "retrieval (VINN)", "bc": "regression BC, chunked", "flow": "flow matching",
            "tokens": "binned tokens (RT-1 style)"}


def figure(curve: list[dict], board: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = {"knn": "#2a6f97", "bc": "#c8442d", "flow": "#3a7d44", "tokens": "#8e6c8a"}
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=110)
    for m in FAMILIES:
        pts = sorted((r["demos"], r["success_%"]) for r in curve if r["policy"] == m)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[m], label=f"policy.{m}")
    for name, style in (("expert IK controller", ":"), ("mpc-cem (world model)", "--")):
        row = next((r for r in board if r["agent"] == name), None)
        if row:
            ax.axhline(row["success_%"], color="#5b6168", ls=style, lw=1, label=name)
    ax.set_xscale("log")
    sizes = sorted({r["demos"] for r in curve})
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("demonstrations used for training")
    ax.set_ylabel("held-out success (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title("Reference policies on MuJoCo planar reach", fontsize=9)
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
    n_all = int(0.6 * states) * 2                            # search + search_val demonstrations (two tasks)
    sizes = [20, n_all] if a.quick else [20, 60, 120, n_all]
    data = {n_all: roots["demos"]}
    for n in sizes[:-1]:
        # n demonstrations balanced over both tasks: the search split of a protocol with 1.25 n init states, whose
        # seeds are a prefix of the full protocol's search seeds (so never in its held-out split)
        data[n] = a.out / "data" / f"demos-n{n}"
        if not (data[n] / ".done").exists():
            C.run("expert", C.EXPERT, C.reach_bench(int(1.25 * n)), "search", data[n], workers=a.workers)
            (data[n] / ".done").touch()
    jobs = {f"{m}-n{n}": (m, [data[n]], a.out / "artifacts" / f"{m}-n{n}", {"chunk": 8})
            for m in FAMILIES for n in sizes}
    jobs["world_model"] = ("world_model.state", [roots["demos"], roots["explore"]], a.out / "artifacts" / "world_model",
                           {"env": {"type": "mujoco.planar_reach"}, **({"epochs": 30} if a.quick else {})})
    C.log(f"training {len(jobs)} artifacts ...")
    art = C.train_many(jobs, procs=a.workers)
    for k, (_, info) in art.items():
        C.log(f"  {k:<14} {info.get('episodes', '?'):>4} episodes  {info['seconds']:>5} s  digest {info['digest']}")

    full = {m: art[f"{m}-n{n_all}"][0] for m in FAMILIES}
    agents = {"expert IK controller": C.EXPERT,
              **{f"{m}/n{n}": {"type": "policy", "params": {"policy": C.policy_ref(m, art[f'{m}-n{n}'][0])}}
                 for m in FAMILIES for n in sizes},
              "bc + temporal ensemble": {"type": "temporal_ensemble",
                                         "params": {"policy": C.policy_ref("bc", full["bc"]), "m": 0.01}},
              "flow + steering (4 samples, medoid)": {"type": "steered", "params": {
                  "sampler": C.policy_ref("flow", full["flow"]), "steering": {"n_samples": 4, "selector": "medoid"}}},
              "mpc-cem (world model)": {"type": "policy", "params": {"policy": {"type": "policy.mpc", "params": {
                  "world_model": str(art["world_model"][0]), "objective": {"reach": ["tip_pos", "target_pos"]}}}}}}
    C.log(f"evaluating {len(agents)} agents on the held-out split ...")
    runs = C.run_many(agents, bench, "heldout", a.out / "runs", procs=a.workers)
    stats = {k: C.summarize(v) for k, v in runs.items()}

    def row(name: str, s: dict, **extra) -> dict:
        return {**extra, "success_%": round(100 * s["success"], 1), "ci95_%": f"{100 * s['ci95'][0]:.0f}-{100 * s['ci95'][1]:.0f}",
                "steps": s["steps"], "policy_ms": s["policy_ms"], "episodes": s["episodes"], "runs": [str(d) for d in runs[name]]}

    curve = [row(f"{m}/n{n}", stats[f"{m}/n{n}"], policy=m, demos=art[f"{m}-n{n}"][1].get("episodes", n),
                 train_s=art[f"{m}-n{n}"][1]["seconds"]) for m in FAMILIES for n in sizes]
    board = [row(k, stats[k], agent=k) for k in ("expert IK controller", "bc + temporal ensemble",
                                                  "flow + steering (4 samples, medoid)", "mpc-cem (world model)")]
    board += [row(f"{m}/n{n_all}", stats[f"{m}/n{n_all}"], agent=f"policy.{m}") for m in FAMILIES]
    board.sort(key=lambda r: (-r["success_%"], r["steps"]))
    for i, r in enumerate(board, 1):
        r["#"] = i
    print()
    C.table(board, ["#", "agent", "success_%", "ci95_%", "steps", "policy_ms"],
            title=f"Leaderboard: held-out split, {board[0]['episodes']} episodes, policies trained on {n_all} demonstrations")
    print()
    C.table(curve, ["policy", "demos", "success_%", "ci95_%", "steps", "train_s"], title="Data efficiency")
    fig = a.out / "learning_curve.png"
    has_fig = figure(curve, board, fig)
    C.write_result(a.out, {"example": f"agents/{SLUG}", "quick": a.quick, "benchmark": bench, "split": "heldout",
                           "leaderboard": board, "learning_curve": curve, "figure": str(fig) if has_fig else None,
                           "artifacts": {k: {"path": str(p), **i} for k, (p, i) in art.items()},
                           "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
