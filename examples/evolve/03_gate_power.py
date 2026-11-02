"""The gate against noise: how often it promotes a change that does nothing (A/A), and how often it promotes a real
improvement, as a function of the number of episodes (power).

What it shows
    Every number comes from real MuJoCo rollouts: each arm is run once on a pool of fresh seeds (2 x 3000 episodes),
    then gate decisions are replayed many times on random subsets of that pool (a bootstrap over episodes).

    Arms (a flow policy trained here from IK-expert demonstrations, run with seeded noise):
        parent          noise_scale 3.0
        A/A             the same policy and noise scale with a fresh noise stream (a trigger rule resamples once at
                        step 0): a different random draw, exactly the same expected success
        tiny / small / medium / large   noise_scale 2.75 / 2.5 / 2.0 / 1.0 (true effects measured on the pool)
        medium, no CRN  noise_scale 2.0 on a fresh noise stream: the same effect without common random numbers

    A. The paired test alone (one-sided exact McNemar at alpha 0.05, the gate's default statistic) on n pairs,
       n = 10 ... 2560: false-promotion rate of A/A, power of each real effect.
    B. The whole campaign cycle (`run_cycle`: stage-1 early rejection, search_val screen, a 2-look group-sequential
       gate with alpha spending, regression vetoes, held-out report), fed with pooled outcomes, at the default split
       sizes of the `rsi` scheme (40 episodes per split) and at 0.5x, 2x, 4x and 8x those sizes.

    This is the project's open problem in one table: at the default sizes the gate is safe (it almost never promotes
    noise) but underpowered for gains of 5 to 10 points, the size most real improvements have.

Command (from the harness repository root)
    uv run python examples/evolve/03_gate_power.py [--quick] [--out DIR] [--workers N]

Typical runtime: about 2 min on 24 CPU workers (--quick: about 20 s).
Requires: hylo-env-mujoco (hylo-bench); matplotlib for the figure (optional).
Output: <out>/result.json, <out>/power_curve.png, <out>/pool.json (every pooled episode outcome).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import flow_agent, parse_args, pool_outcomes, reach_config, require, table, train_flow, wilson, write_result

SLUG = "gate_power"
ALPHA = 0.05
NOISE = "agent.params.steering"


def fresh_noise(scale: float) -> dict:
    """A trigger rule that fires once at step 0 and resamples with the same noise scale: only the noise stream changes."""
    return {"name": "fresh_noise", "feature": "steps", "op": ">=", "value": 0, "action": "resample",
            "params": {"noise_scale": scale}, "cooldown": 0, "max_fires": 1}


ARMS = {   # name -> steering config
    "parent": {"noise_scale": 3.0},
    "A/A": {"noise_scale": 3.0, "rules": [fresh_noise(3.0)]},
    "tiny": {"noise_scale": 2.75},
    "small": {"noise_scale": 2.5},
    "medium": {"noise_scale": 2.0},
    "large": {"noise_scale": 1.0},
    "medium, no CRN": {"noise_scale": 2.0, "rules": [fresh_noise(2.0)]},
}
CANDIDATES = [k for k in ARMS if k != "parent"]
RSI = {"search": 40, "search_val": 40, "gate": 40, "gate_r1": 40, "heldout": 40}     # episodes per split (rsi scheme)


# ---------------------------------------------------------------- A: the paired test alone

def paired_power(parent: np.ndarray, cand: np.ndarray, ns: list[int], trials: int, seed: int) -> dict[int, float]:
    from hylo.eval.scoring import one_sided_exact_mcnemar

    ok = ~(np.isnan(parent) | np.isnan(cand))
    p, c = parent[ok], cand[ok]
    out, cache = {}, {}
    rng = np.random.default_rng(seed)
    for n in ns:
        if n > len(p):
            continue
        hits = 0
        for _ in range(trials):
            idx = rng.choice(len(p), size=n, replace=False)
            cw = int(np.sum((c[idx] == 1) & (p[idx] == 0)))
            pw = int(np.sum((p[idx] == 1) & (c[idx] == 0)))
            if (cw, pw) not in cache:
                cache[(cw, pw)] = one_sided_exact_mcnemar(cw, pw)
            hits += cache[(cw, pw)] < ALPHA
        out[n] = hits / trials
    return out


# ---------------------------------------------------------------- B: the whole cycle on pooled outcomes

_POOL: dict = {}


def _init(pool: dict) -> None:
    _POOL.update(pool)


def _result(rows: list[dict], run_dir: Path):
    from hylo.eval.runner import EvalResult

    samples = [{"key": r["key"], "task_id": r["task_id"], "status": r["status"], "episode_id": r["key"],
                "scores": {"success": {"value": r["success"]}, "safety_interventions": {"value": 0.0},
                           "steps": {"value": float(r["steps"])}}} for r in rows]
    ok = [r["success"] for r in rows if r["success"] is not None]
    return EvalResult(run_dir, {"samples": samples, "results": {"success_mean": sum(ok) / max(len(ok), 1),
                                                                "safety_interventions_mean": 0.0, "n_episodes": len(rows)}})


def _cycles(job: tuple[str, float, list[int]]) -> list[bool]:
    """Run whole campaign cycles (run_cycle) for one arm and scale, each on a fresh random draw of pooled episodes."""
    from hylo.core.config import ExperimentConfig
    from hylo.core.improvement import Candidate
    from hylo.evolve.protocol import Archive, CandidateRegistry, ProtocolConfig, run_cycle

    arm, scale, trial_ids = job
    base = ExperimentConfig.model_validate(_POOL["config"])
    parent, cand = _POOL["arms"]["parent"], _POOL["arms"][arm]
    keys = sorted(set(parent) & set(cand))
    sizes = {s: max(2, round(n * scale)) for s, n in RSI.items()}
    proto = ProtocolConfig(alpha=ALPHA, gate_looks=2, stage1_pairs=max(2, round(20 * scale)))
    patch = {NOISE: ARMS[arm]}

    class One:
        name = "examples.fixed@0.1"

        def propose(self, diagnosis, cfg, parent_id, archive):
            c = Candidate(patch=patch, rationale=arm, parent_id=parent_id, generator=self.name)
            return [(c, arm)]

    out = []
    for t in trial_ids:
        rng = np.random.default_rng([CANDIDATES.index(arm), int(scale * 100), t])
        draw = list(rng.choice(len(keys), size=sum(sizes.values()), replace=False))
        split_keys, i = {}, 0
        for s, n in sizes.items():
            split_keys[s] = [keys[j] for j in draw[i:i + n]]
            i += n
        with tempfile.TemporaryDirectory() as tmp:
            def evaluate(cfg, split, label):
                src = parent if label.startswith("parent") else cand
                return _result([src[k] for k in split_keys[split]], Path(tmp))

            rep = run_cycle(base, CandidateRegistry(Path(tmp) / "registry"), Archive(Path(tmp) / "archive.jsonl"), One(),
                            evaluate, lambda r: {"buckets": {}}, protocol=proto)
        out.append(rep.promoted is not None)
    return out


def cycle_power(pool: dict, scales: list[float], trials: int, workers: int) -> dict[str, dict[float, float]]:
    import multiprocessing as mp

    jobs = [(arm, s, list(range(i, min(i + 25, trials)))) for arm in CANDIDATES for s in scales for i in range(0, trials, 25)]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1)), mp_context=mp.get_context("forkserver"),
                             initializer=_init, initargs=(pool,)) as ex:
        res = list(ex.map(_cycles, jobs))
    acc: dict[str, dict[float, list[bool]]] = {}
    for (arm, s, _), r in zip(jobs, res):
        acc.setdefault(arm, {}).setdefault(s, []).extend(r)
    return {arm: {s: sum(v) / len(v) for s, v in d.items()} for arm, d in acc.items()}


# ---------------------------------------------------------------- main

def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    t0 = time.time()
    flow = train_flow(a.out, epochs=60 if a.quick else 300)
    n_seeds = 150 if a.quick else 3000
    trials_a, trials_b = (300, 50) if a.quick else (4000, 2000)
    scales = [1.0] if a.quick else [0.5, 1.0, 2.0, 4.0, 8.0]
    base = reach_config(flow_agent(flow, 3.0), "evolve-gate-power", protocol="rsi", gate_looks=2)

    # ---- the pool: every arm on the same fresh seeds (common random numbers unless the arm changes the noise stream)
    seeds = range(100_000, 100_000 + n_seeds)
    arms: dict[str, dict[str, dict]] = {}
    for name, steering in ARMS.items():
        t = time.time()
        rows = pool_outcomes(base.with_overrides({NOISE: steering}), seeds, workers=a.workers, tmp_root=a.out / "tmp")
        arms[name] = {r["key"]: r for r in rows}
        ok = [r["success"] for r in rows if r["success"] is not None]
        print(f"  pool {name:<15} {len(rows)} episodes, success {sum(ok) / len(ok):.3f} ({time.time() - t:.0f}s)")
    shutil.rmtree(a.out / "tmp", ignore_errors=True)
    (a.out / "pool.json").write_text(json.dumps({k: list(v.values()) for k, v in arms.items()}))
    keys = sorted(arms["parent"])
    vec = {k: np.array([np.nan if v[x]["success"] is None else v[x]["success"] for x in keys]) for k, v in arms.items()}

    truth = {}
    for name in ARMS:
        v = vec[name][~np.isnan(vec[name])]
        lo, hi = wilson(v.sum(), len(v))
        row = {"success": float(v.mean()), "ci95": [lo, hi], "n": int(len(v))}
        if name != "parent":
            p, c = vec["parent"], vec[name]
            ok = ~(np.isnan(p) | np.isnan(c))
            row.update(effect=float(c[ok].mean() - p[ok].mean()), p_cand_only=float(np.mean((c[ok] == 1) & (p[ok] == 0))),
                       p_parent_only=float(np.mean((p[ok] == 1) & (c[ok] == 0))))
        truth[name] = row
    print("\narms on the pool (true values for the bootstrap)")
    table([[k, f"{r['success']:.3f}", f"{r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}",
            f"{r['effect']:+.3f}" if "effect" in r else "-",
            f"{r['p_cand_only']:.3f} / {r['p_parent_only']:.3f}" if "effect" in r else "-"] for k, r in truth.items()],
          ["arm", "success", "95% CI", "effect", "discordant: cand-only / parent-only"])

    # ---- A: the paired test alone
    ns = [n for n in (10, 20, 40, 80, 160, 320, 640, 1280, 2560) if n <= len(keys)]
    power_a = {arm: paired_power(vec["parent"], vec[arm], ns, trials_a, seed=i) for i, arm in enumerate(CANDIDATES)}
    print(f"\nA. one-sided exact McNemar, alpha {ALPHA}: promotion rate over {trials_a} draws of n pairs")
    table([[n, *[f"{power_a[arm].get(n, float('nan')):.3f}" for arm in CANDIDATES]] for n in ns], ["pairs", *CANDIDATES])
    need = {}
    for arm in CANDIDATES[1:]:
        hit = [n for n in ns if power_a[arm].get(n, 0) >= 0.8]
        need[arm] = hit[0] if hit else None
    print("  pairs needed for 80% power: " + ", ".join(f"{k} {v if v else '> ' + str(ns[-1])}" for k, v in need.items()))

    # ---- B: whole campaign cycles
    pool = {"config": base.model_dump(mode="json"), "arms": arms}
    power_b = cycle_power(pool, scales, trials_b, a.workers)
    print(f"\nB. whole campaign cycle (run_cycle, 2 gate looks): promotion rate over {trials_b} cycles")
    table([[f"{s:g}x", int(RSI["gate"] * s), *[f"{power_b[arm][s]:.3f}" for arm in CANDIDATES]] for s in scales],
          ["split size", "gate eps/look", *CANDIDATES])

    result = {"example": SLUG, "quick": a.quick, "alpha": ALPHA, "pool_seeds": n_seeds, "arms": ARMS, "truth": truth,
              "paired_test": {"trials": trials_a, "power": {k: {str(n): v for n, v in d.items()} for k, d in power_a.items()},
                              "pairs_for_80pct_power": need},
              "campaign_cycle": {"trials": trials_b, "split_episodes_at_1x": RSI, "gate_looks": 2,
                                 "power": {k: {str(s): v for s, v in d.items()} for k, d in power_b.items()}},
              "elapsed_s": round(time.time() - t0, 1)}
    plot(a.out / "power_curve.png", ns, power_a, power_b, truth, scales)
    write_result(a.out, result)


def plot(path, ns, power_a, power_b, truth, scales) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"A/A": "#5b6168", "tiny": "#8c6bb1", "small": "#e3a33b", "medium": "#c8442d", "large": "#1f5fa8", "medium, no CRN": "#c8442d"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=110, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for arm, d in power_a.items():
        xs = [n for n in ns if n in d]
        eff = truth[arm]["effect"]
        ax.plot(xs, [d[n] for n in xs], marker="o", ms=4, color=colors[arm], ls="--" if "no CRN" in arm else "-",
                label=f"{arm} ({eff * 100:+.1f} pts)")
    ax.axhline(0.05, color="#999", lw=1, ls=":")
    ax.axhline(0.8, color="#999", lw=1, ls=":")
    ax.text(ns[0], 0.06, "alpha 0.05", fontsize=7, color="#777")
    ax.text(ns[0], 0.81, "80% power", fontsize=7, color="#777")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("paired episodes per arm")
    ax.set_ylabel("promotion rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. paired McNemar test alone", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center right")
    ax = axes[1]
    xs = [RSI["gate"] * s for s in scales]
    for arm, d in power_b.items():
        ax.plot(xs, [d[s] for s in scales], marker="s", ms=4, color=colors[arm], ls="--" if "no CRN" in arm else "-", label=arm)
    ax.axhline(0.05, color="#999", lw=1, ls=":")
    ax.axhline(0.8, color="#999", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [f"{int(x)}" for x in xs])
    ax.set_xlabel("episodes per split (gate look, search_val)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("B. whole campaign cycle (rsi default: 40)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
