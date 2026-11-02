"""Learner jobs: recorded runs in, a sealed policy artifact out, then evaluated on episodes it never saw.

What it shows
    1. Demonstrations: the IK controller recorded on the development split `search` (nested sets of 10 to 160
       episodes).
    2. `TrainJob` + `run_job` (what `hylo learn JOB.json` does) for three reference policy families (kNN, BC, flow
       matching); each output is a sealed artifact: kind, role, content digest, and the runs it read.
    3. The seal at work: `verify` passes; one byte changed in the artifact fails it; a job asking for `gate` runs is
       refused before reading anything; the same job run again is skipped (already sealed with the job's digest).
    4. Each policy on 2 x 150 held-out episodes (standard `heldout` seeds, never used for learning), with Wilson 95%
       intervals, and the flow policy's success against the number of demonstrations.

Command (from the harness repository root)
    uv run python examples/learn/01_learner_job.py [--quick] [--out DIR]

Typical runtime: about 35 s on CPU (--quick: about 15 s).
Requires: hylo-env-mujoco (hylo-bench), hylo-policies; matplotlib for the figure (optional).
Output: <out>/result.json, <out>/data_scaling.png, <out>/demos (runs), <out>/artifacts (sealed artifacts).
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor

from _common import EXPERT, parse_args, pool_outcomes, reach_config, require, table, wilson, write_result

SLUG = "learner_job"


def _train(args: tuple[str, list[str], str, dict]) -> dict:
    from hylo.core.learning import TrainJob
    from hylo.learn.jobs import run_job

    learner, inputs, out, params = args
    t = time.time()
    m = run_job(TrainJob(learner=learner, inputs=inputs, out=out, params=params))
    return {"learner": learner, "out": out, "digest": m.digest, "kind": m.kind, "role": m.role, "metrics": m.metrics,
            "splits": sorted(m.provenance.splits()), "sources": len(m.provenance.sources), "train_s": round(time.time() - t, 1)}


def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    from hylo.core.artifact import ArtifactError, verify
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.learn.jobs import run_job

    t0 = time.time()
    epochs = {"bc": 40, "flow": 60} if a.quick else {"bc": 150, "flow": 300}
    sizes = [20, 160] if a.quick else [10, 20, 40, 80, 160]          # demonstration episodes (2 tasks x seeds)

    # ---- 1. demonstrations on `search`: rsi_states chosen so that search holds sizes[i] / 2 seeds (nested sets)
    demo_dirs = {}
    for n in sizes:
        d = a.out / "demos" / f"n{n:03d}"
        if not (d / ".done").exists():
            shutil.rmtree(d, ignore_errors=True)
            states = (n // 2 * 10 + 3) // 4                              # ceil(seeds / 0.4)
            run_experiment(reach_config(EXPERT, f"learn-demos-{n}", protocol="rsi", rsi_states=states),
                           split="search", run_root=d, label=f"expert-{n}")
            (d / ".done").touch()
        demo_dirs[n] = d
    full = demo_dirs[sizes[-1]]

    # ---- 2. learner jobs (in worker processes); the flow policy once per data size
    arts = a.out / "artifacts"
    jobs = [("policy.knn", [str(full)], str(arts / "knn"), {}),
            ("policy.bc", [str(full)], str(arts / "bc"), {"epochs": epochs["bc"], "execution_steps": 4}),
            *[("policy.flow", [str(demo_dirs[n])], str(arts / f"flow-n{n:03d}"), {"epochs": epochs["flow"]}) for n in sizes]]
    with ProcessPoolExecutor(max_workers=min(len(jobs), a.workers)) as ex:
        trained = {j[2]: r for j, r in zip(jobs, ex.map(_train, jobs))}
    flow_full = str(arts / f"flow-n{sizes[-1]:03d}")
    print("sealed artifacts")
    table([[r["out"].split("/")[-1], r["kind"], r["role"], r["digest"], r["metrics"]["episodes"], r["metrics"]["samples"],
            ",".join(r["splits"]), f"{r['train_s']:.0f}s"] for r in trained.values()],
          ["artifact", "kind", "role", "digest", "episodes", "samples", "splits read", "train"])

    # ---- 3. the seal at work
    checks = {}
    m = verify(flow_full)
    checks["verify"] = f"ok: digest {m.digest}, sources {[s.split for s in m.provenance.sources]}"
    tampered = arts / "flow-tampered"
    shutil.rmtree(tampered, ignore_errors=True)
    shutil.copytree(flow_full, tampered)
    victim = next(p for p in sorted(tampered.iterdir()) if p.is_file() and p.name != "artifact.json")
    victim.write_bytes(victim.read_bytes()[:-1] + bytes([victim.read_bytes()[-1] ^ 1]))
    try:
        verify(tampered)
        checks["tampered"] = "accepted (unexpected)"
    except ArtifactError as exc:
        checks["tampered"] = f"refused: {str(exc).split(': ', 1)[-1]}"
    try:
        run_job(TrainJob(learner="policy.flow", inputs=[str(full)], out=str(arts / "flow-gate"), splits=("search", "gate")))
        checks["gate job"] = "accepted (unexpected)"
    except (ArtifactError, ValueError) as exc:
        checks["gate job"] = f"refused ({type(exc).__name__}): {str(exc).splitlines()[0][:150]}"
    t = time.time()
    again = run_job(TrainJob(learner="policy.flow", inputs=[str(full)], out=flow_full, params={"epochs": epochs["flow"]}))
    checks["rerun"] = f"skipped, already sealed (digest {again.digest}, {time.time() - t:.2f}s)"
    print("\nchecks")
    for k, v in checks.items():
        print(f"  {k:<9} {v}")

    # ---- 4. evaluation on held-out seeds (standard heldout: 20000+), never used for learning
    n_seeds = 30 if a.quick else 150
    seeds = range(20_000, 20_000 + n_seeds)
    pol = lambda kind, path: {"type": kind, "params": {"path": path}}  # noqa: E731
    agents = {
        "expert (IK controller)": EXPERT,
        "knn": {"type": "policy", "params": {"policy": pol("policy.knn", str(arts / "knn"))}},
        "bc": {"type": "policy", "params": {"policy": pol("policy.bc", str(arts / "bc"))}},
        "bc + temporal ensemble": {"type": "temporal_ensemble", "params": {"policy": pol("policy.bc", str(arts / "bc")), "m": 0.1}},
        "flow": {"type": "policy", "params": {"policy": pol("policy.flow", flow_full)}},
        "flow + steering (4 samples, medoid)": {"type": "steered", "params": {
            "sampler": pol("policy.flow", flow_full), "steering": {"n_samples": 4, "selector": "medoid"}}},
    }
    evals = {}
    for name, agent in agents.items():
        rows = pool_outcomes(reach_config(agent, "learn-eval"), seeds, workers=a.workers, tmp_root=a.out / "tmp")
        ok = [r["success"] for r in rows if r["success"] is not None]
        evals[name] = {"success": sum(ok) / len(ok), "ci95": wilson(sum(ok), len(ok)), "n": len(ok),
                       "mean_steps": sum(r["steps"] for r in rows) / len(rows)}
    scaling = {}
    for n in sizes:
        path = str(arts / f"flow-n{n:03d}")
        rows = pool_outcomes(reach_config({"type": "policy", "params": {"policy": pol("policy.flow", path)}}, "learn-eval"),
                             seeds, workers=a.workers, tmp_root=a.out / "tmp")
        ok = [r["success"] for r in rows if r["success"] is not None]
        scaling[n] = {"success": sum(ok) / len(ok), "ci95": wilson(sum(ok), len(ok)), "n": len(ok)}
    shutil.rmtree(a.out / "tmp", ignore_errors=True)
    print(f"\npolicies on {2 * n_seeds} held-out episodes (seeds 20000+, both tasks)")
    table([[k, f"{v['success']:.3f}", f"{v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}", f"{v['mean_steps']:.0f}"] for k, v in evals.items()],
          ["policy", "success", "95% CI", "mean steps"])
    print("\nflow policy against the number of demonstrations")
    table([[n, f"{v['success']:.3f}", f"{v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}"] for n, v in scaling.items()],
          ["demo episodes", "success", "95% CI"])

    plot(a.out / "data_scaling.png", scaling, evals["expert (IK controller)"]["success"])
    write_result(a.out, {"example": SLUG, "quick": a.quick, "epochs": epochs, "artifacts": list(trained.values()),
                         "checks": checks, "heldout_episodes": 2 * n_seeds, "policies": evals,
                         "data_scaling": {str(k): v for k, v in scaling.items()}, "elapsed_s": round(time.time() - t0, 1)})


def plot(path, scaling, expert) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ns = list(scaling)
    ys = [scaling[n]["success"] for n in ns]
    err = [[y - scaling[n]["ci95"][0] for n, y in zip(ns, ys)], [scaling[n]["ci95"][1] - y for n, y in zip(ns, ys)]]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=110)
    ax.errorbar(ns, ys, yerr=err, marker="o", capsize=3, color="#1f5fa8", lw=2, label="flow policy (held-out)")
    ax.axhline(expert, color="#5b6168", ls=":", lw=1, label="expert it imitates")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("demonstration episodes")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Flow matching on MuJoCo reach: success vs demonstrations", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
