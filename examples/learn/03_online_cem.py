"""Online learning: a cross-entropy-method (CEM) learner tunes how a policy is run by rolling it out, iteration
after iteration, then the result is sealed and checked on held-out episodes.

What it shows
    1. An online learner (the `OnlineLearner` contract: start / policies / update / finished / seal), written here
       in a few lines: CEM over three steering settings of a flow policy (noise scale, samples per decision,
       actions executed per chunk), starting from a badly tuned setting (noise scale 3.0).
    2. `run_online` (what `hylo learn` does for an online learner): each iteration rolls out every proposed setting
       on the development split `search` (recorded as ordinary runs, listed in the artifact's provenance), then the
       learner updates its sampling distribution. A job asking to roll out on `gate` is refused.
    3. The learning curve on `search`, and the honest number: the starting setting, the CEM mean and the best
       setting found, each on 2 x 250 held-out episodes it never saw (paired with the starting setting).

Command (from the harness repository root)
    uv run python examples/learn/03_online_cem.py [--quick] [--out DIR]

Typical runtime: about 1.5 min on CPU (--quick: about 30 s).
Requires: hylo-env-mujoco (hylo-bench), hylo-policies; matplotlib for the figure (optional).
Output: <out>/result.json, <out>/cem_curve.png, <out>/cem (sealed artifact: best.json, history.json, rollouts/).
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from _common import EXPERT, parse_args, pool_outcomes, reach_config, require, table, wilson, write_result

SLUG = "online_cem"
DIMS = (("noise_scale", 0.05, 3.0, True), ("n_samples", 1, 6, False), ("execution_steps", 1, 8, False))


class CEMSteering:
    """CEM over steering settings. Parameters are searched in a Gaussian (log space for the noise scale)."""

    name = "examples.cem_steering@0.1"

    def start(self, job, agent) -> None:
        p = job.params
        self.job, self.base = job, agent
        self.pop, self.n_elite = int(p.get("population", 8)), int(p.get("elites", 3))
        self.iters, self.it = int(p.get("iterations", 5)), 0
        self.rng = np.random.default_rng(int(p.get("seed", 0)))
        st = agent["params"].get("steering", {})
        self.mu = np.array([np.log(st.get("noise_scale", 1.0)), st.get("n_samples", 1), st.get("execution_steps") or 4], float)
        self.sd = np.array([0.6, 1.5, 2.0])
        self.best: tuple[float, dict] | None = None
        self.history: list[dict] = []

    def _decode(self, x: np.ndarray) -> dict:
        out = {}
        for v, (name, lo, hi, log) in zip(x, DIMS):
            out[name] = round(float(np.clip(np.exp(v) if log else v, lo, hi)), 3) if log else int(np.clip(round(v), lo, hi))
        return out

    def _spec(self, setting: dict) -> dict:
        spec = copy.deepcopy(self.base)
        spec["params"]["steering"] = {**spec["params"].get("steering", {}), **setting,
                                      "selector": "medoid" if setting["n_samples"] > 1 else "first"}
        return spec

    def policies(self) -> list[dict]:
        self.x = self.mu + self.sd * self.rng.standard_normal((self.pop, len(DIMS)))
        self.settings = [self._decode(x) for x in self.x]
        return [self._spec(s) for s in self.settings]

    def update(self, results: list[dict]) -> dict:
        # score: success, then fewer steps (a tie-break worth less than one episode)
        score = np.array([r["metrics"]["success_mean"] - 1e-4 * r["metrics"]["steps_mean"] for r in results])
        elite = np.argsort(score)[::-1][: self.n_elite]
        self.mu = self.x[elite].mean(0)
        self.sd = np.maximum(self.x[elite].std(0), [0.1, 0.3, 0.3])
        i = int(elite[0])
        if self.best is None or score[i] > self.best[0]:
            self.best = (float(score[i]), self.settings[i])
        self.it += 1
        rec = {"success": [r["metrics"]["success_mean"] for r in results], "settings": self.settings,
               "elite_mean_success": float(np.mean([results[j]["metrics"]["success_mean"] for j in elite])),
               "mean_setting": self._decode(self.mu), "best": self.best[1]}
        self.history.append(rec)
        return {"mean_success": round(float(np.mean(rec["success"])), 3), "elite": round(rec["elite_mean_success"], 3),
                "mean_setting": rec["mean_setting"]}

    def finished(self) -> bool:
        return self.it >= self.iters

    def seal(self, out, sources):
        from hylo.core.artifact import Provenance, write_manifest

        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "best.json").write_text(json.dumps({"best": self.best[1], "mean": self._decode(self.mu)}, indent=2))
        return write_manifest(out, kind="other", role="behaviour", name="cem-steering",
                              metrics={"best": self.best[1], "best_search_score": self.best[0], "mean": self._decode(self.mu)},
                              provenance=Provenance(builder=self.name, sources=sources,
                                                    params={k: v for k, v in self.job.params.items() if k != "experiment"}))


def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    from hylo.core.artifact import SplitLeak, verify
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.eval.scoring import paired_compare
    from hylo.learn.jobs import run_job
    from hylo.learn.online import run_online

    t0 = time.time()
    demos = a.out / "demos"
    if not (demos / ".done").exists():
        shutil.rmtree(demos, ignore_errors=True)
        # 120 demonstrations on init states disjoint from the CEM rollouts (rsi_states 300: search_val is 30120-30179;
        # the rollouts use the default scheme's search split, 30000-30039)
        run_experiment(reach_config(EXPERT, "learn-cem-demos", protocol="rsi", rsi_states=300), split="search_val",
                       run_root=demos, label="search_val")
        (demos / ".done").touch()
    run_job(TrainJob(learner="policy.flow", inputs=[str(demos)], out=str(a.out / "flow"), params={"epochs": 60 if a.quick else 300}))
    start = {"noise_scale": 3.0, "n_samples": 1, "execution_steps": 4}
    agent = {"type": "steered", "params": {"sampler": {"type": "policy.flow", "params": {"path": str(a.out / "flow")}},
                                           "steering": start}}
    exp = reach_config(agent, "learn-online-cem", protocol="rsi")
    pop, elites, iters = (4, 2, 2) if a.quick else (8, 3, 6)

    try:
        run_online(TrainJob(learner="examples.cem", inputs=[], out=str(a.out / "cem-gate"), splits=("search",),
                            params={"experiment": exp.model_dump(mode="json"), "split": "gate"}), CEMSteering(), log=lambda s: None)
        refused = "accepted (unexpected)"
    except SplitLeak as exc:
        refused = f"refused: {exc}"
    print(f"online job rolling out on `gate`: {refused}")

    out = a.out / "cem"
    shutil.rmtree(out, ignore_errors=True)
    learner = CEMSteering()
    job = TrainJob(learner="examples.cem", inputs=[], out=str(out), splits=("search",),
                   params={"experiment": exp.model_dump(mode="json"), "iterations": iters, "population": pop,
                           "elites": elites, "split": "search", "seed": 0})
    m = run_online(job, learner, log=lambda s: print("  " + s))
    m = verify(out)
    print(f"sealed {out.name}: digest {m.digest}, {len(m.provenance.sources)} rollout runs in its provenance "
          f"({sum(s.n_episodes or 0 for s in m.provenance.sources)} episodes), best {m.metrics['best']}")

    print("\nlearning curve on `search` (80 episodes per setting)")
    table([[i + 1, f"{np.mean(h['success']):.3f}", f"{h['elite_mean_success']:.3f}", f"{max(h['success']):.3f}",
            json.dumps(h["mean_setting"])] for i, h in enumerate(learner.history)],
          ["iteration", "population mean", "elite mean", "best", "CEM mean after update"])

    # ---- held-out check: the starting setting, the CEM mean and the best setting found
    n_seeds = 30 if a.quick else 250
    seeds = range(20_000, 20_000 + n_seeds)
    candidates = {"start": start, "CEM mean": learner._decode(learner.mu), "CEM best": learner.best[1]}
    rows_by = {}
    for name, st in candidates.items():
        spec = learner._spec(st) if name != "start" else agent
        rows_by[name] = {r["key"]: r for r in pool_outcomes(exp.with_overrides({"agent": spec}), seeds, workers=a.workers,
                                                            tmp_root=a.out / "tmp")}
    shutil.rmtree(a.out / "tmp", ignore_errors=True)
    held = {}
    base = {k: r["success"] for k, r in rows_by["start"].items()}
    for name, rows in rows_by.items():
        ok = [r["success"] for r in rows.values() if r["success"] is not None]
        pc = paired_compare({k: r["success"] for k, r in rows.items()}, base)
        held[name] = {"setting": candidates[name], "success": sum(ok) / len(ok), "ci95": wilson(sum(ok), len(ok)),
                      "vs_start": {"wins": pc.candidate_wins, "losses": pc.parent_wins, "p": pc.p_value}}
    print(f"\nheld-out check ({2 * n_seeds} episodes, seeds 20000+)")
    table([[k, json.dumps(v["setting"]), f"{v['success']:.3f}", f"{v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}",
            f"{v['vs_start']['wins']}:{v['vs_start']['losses']} p={v['vs_start']['p']:.1e}" if k != "start" else "-"]
           for k, v in held.items()], ["setting", "values", "success", "95% CI", "vs start (paired)"])
    best_search = max(max(h["success"]) for h in learner.history)
    print(f"  best setting scored {best_search:.3f} on search, {held['CEM best']['success']:.3f} held-out "
          f"(selection on 80 episodes is optimistic)")

    plot(a.out / "cem_curve.png", learner.history, held)
    write_result(a.out, {"example": SLUG, "quick": a.quick, "population": pop, "elites": elites, "iterations": iters,
                         "gate_job": refused, "artifact": {"digest": m.digest, "rollout_runs": len(m.provenance.sources),
                                                           "metrics": m.metrics},
                         "history": learner.history, "heldout": held, "elapsed_s": round(time.time() - t0, 1)})


def plot(path, history, held) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=110)
    its = np.arange(1, len(history) + 1)
    for i, h in zip(its, history):
        ax.scatter([i] * len(h["success"]), h["success"], s=12, color="#9aa4ad", zorder=2)
    ax.plot(its, [np.mean(h["success"]) for h in history], marker="o", color="#1f5fa8", lw=2, label="population mean (search)")
    ax.plot(its, [h["elite_mean_success"] for h in history], marker="s", color="#c8442d", lw=1.5, label="elite mean (search)")
    ax.axhline(held["start"]["success"], color="#5b6168", ls=":", lw=1, label="start, held-out")
    ax.axhline(held["CEM mean"]["success"], color="#1f5fa8", ls="--", lw=1, label="CEM mean, held-out")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(its)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Online CEM over steering settings (flow policy, MuJoCo reach)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
