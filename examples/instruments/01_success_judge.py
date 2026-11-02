"""Success judge: train a trace judge on your own runs, then calibrate it against the simulator outcome.

A success judge says "is the task done?" from what the robot records, without the simulator's verdict; only a
calibrated judge may label data (memory, self-training sets) on its own. This example does the whole loop on
MuJoCo planar reach, where the simulator outcome is known, so the judge can be checked:

1. runs with mixed outcomes on the development splits: IK controllers too slow for the 150-step budget, and random
   actions (search: training, search_val: calibration);
2. the `judge.trace` learner (a `hylo learn` job) on the `search` runs, twice, with different observations:
   - "proprio + target": joint angles and velocities, fingertip and target position;
   - "proprio only": what the arm senses about itself (no target position), plus the task id;
3. calibration (`hylo calibrate`, hylo.learn.calibrate) on the `search_val` runs, which neither judge saw: AUC,
   Brier score, expected calibration error, the lowest threshold with 95% precision and the recall there, and
   whether the judge is ready to label; per task, and a reliability diagram and ROC curves (OUT/reliability.png).

    uv run python examples/instruments/01_success_judge.py [--quick] [--out DIR]

Runtime: about 80 s (quick: about 15 s). Requires hylo-env-mujoco (bench) and hylo-learn; the figure needs
matplotlib.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

import numpy as np  # noqa: E402

SLUG = "success_judge"
DATA_AGENTS = {"ik-kp0.015": {"type": "baseline.planar_ik", "params": {"kp": 0.015}},
               "ik-kp0.02": {"type": "baseline.planar_ik", "params": {"kp": 0.02}},
               "ik-kp0.03": {"type": "baseline.planar_ik", "params": {"kp": 0.03}},
               "random": {"type": "baseline.random", "params": {"dim": 2, "seed": 0}}}
JUDGES = {"all state": ["qpos", "qvel", "tip_pos", "target_pos"],
          "fingertip + target": ["tip_pos", "target_pos"],
          "proprio only": ["qpos", "qvel", "tip_pos"],
          "positions only": ["qpos", "tip_pos"]}


def reliability(records: dict[str, list[dict]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=110)
    colors = {"all state": "#c8442d", "fingertip + target": "#3a7d44", "proprio only": "#2a6f97",
              "positions only": "#8e6c8a"}
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#5b6168", lw=0.8, ls=":")
    for name, recs in records.items():
        p = np.array([r["p_success"] for r in recs])
        y = np.array([r["success"] for r in recs])
        bins = np.minimum((p * 10).astype(int), 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = bins == b
            if m.any():
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                ns.append(m.sum())
        ax.plot(xs, ys, "o-", color=colors[name], label=name, ms=3)
    ax.set_xlabel("judge p(success)")
    ax.set_ylabel("observed success rate")
    ax.set_title("Reliability (calibration split)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#5b6168", lw=0.8, ls=":")
    for name, recs in records.items():
        p = np.array([r["p_success"] for r in recs])
        y = np.array([r["success"] for r in recs])
        order = np.argsort(-p, kind="mergesort")
        tpr = np.concatenate([[0], np.cumsum(y[order]) / max(1, y.sum())])
        fpr = np.concatenate([[0], np.cumsum(1 - y[order]) / max(1, (1 - y).sum())])
        ax.plot(fpr, tpr, color=colors[name], label=name, lw=1.2)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC (calibration split)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
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
    data = a.out / "data"
    stamp = data / "bench.json"
    import json
    if not (stamp.exists() and json.loads(stamp.read_text()) == bench):
        import shutil
        shutil.rmtree(data, ignore_errors=True)
        for split in ("search", "search_val"):
            for name, agent in DATA_AGENTS.items():
                C.run(name, agent, bench, split, data, workers=a.workers)
        stamp.write_text(json.dumps(bench))
    outcomes = {}
    for split in ("search", "search_val"):
        st = [s["status"] for d in sorted(data.iterdir()) if d.is_dir() and f"-{split}-" in d.name
              for s in C.evallog(d)["samples"]]
        outcomes[split] = {"episodes": len(st), "success_rate": round(float(np.mean([s == "success" for s in st])), 3)}
    C.log(f"runs: {outcomes}")

    from hylo.core.registry import registry
    from hylo.learn.calibrate import calibrate_judge, calibration_report

    rows, records, per_task = [], {}, {}
    for name, keys in JUDGES.items():
        out = a.out / "artifacts" / ("judge-" + C.slugify(name))
        path, info = C.train("judge.trace", [data], out, {"cameras": [], "state_keys": keys, "workers": a.workers,
                                                          **({"epochs": 30} if a.quick else {})},
                             splits=("search",))
        info_splits = ("search",)
        judge = registry.build("instruments", {"type": "judge.trace", "params": {"path": str(path)}})
        recs: list[dict] = []
        t = time.perf_counter()
        rep = calibrate_judge(judge, [data], splits=("search_val",), judge_artifact=path, records=recs)
        judge_ms = 1000 * (time.perf_counter() - t) / max(1, len(recs))
        records[name] = recs
        per_task[name] = {}
        for task in sorted({r["task_id"] for r in recs}):
            sub = [r for r in recs if r["task_id"] == task]
            tr = calibration_report([r["p_success"] for r in sub], [r["success"] for r in sub], min_n=1)
            per_task[name][task] = {"n": tr.n, "base_rate": round(tr.base_rate, 3),
                                    "auc": None if tr.auc is None else round(tr.auc, 3), "recall_at_p95": tr.recall}
        rows.append({"judge": name, "observes": ", ".join(keys), "trained_on": info_splits, "train_s": info["seconds"],
                     "n": rep.n, "base_rate": round(rep.base_rate, 3), "auc": round(rep.auc, 3) if rep.auc else None,
                     "brier": round(rep.brier, 3), "ece": rep.ece,
                     "threshold": None if rep.threshold is None else round(rep.threshold, 3),
                     "precision": None if rep.precision is None else round(rep.precision, 3),
                     "recall": None if rep.recall is None else round(rep.recall, 3), "ready": rep.ready,
                     "judge_ms": round(judge_ms, 2), "artifact": str(path), "digest": info["digest"],
                     **{f"auc_{t.split('_')[-1]}": v["auc"] for t, v in per_task[name].items()}})
    print()
    C.table(rows, ["judge", "n", "base_rate", "auc", "auc_fixed", "auc_random", "brier", "ece", "threshold",
                   "precision", "recall", "ready"],
            title="Calibration on the search_val runs (unseen by both judges); threshold = lowest with precision >= 0.95")
    fig = a.out / "reliability.png"
    has_fig = reliability(records, fig)
    C.write_result(a.out, {"example": f"instruments/{SLUG}", "quick": a.quick, "benchmark": bench,
                           "data_agents": DATA_AGENTS, "outcomes": outcomes, "judges": rows, "per_task": per_task,
                           "figure": str(fig) if has_fig else None,
                           "records": {k: [{kk: r[kk] for kk in ("task_id", "success", "p_success")} for r in v]
                                       for k, v in records.items()},
                           "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
