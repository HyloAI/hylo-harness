"""Success judges: learn one from traces, calibrate it against simulator outcomes, let it label data that has none.

What it shows
    1. Development runs with mixed outcomes on MuJoCo reach (IK controllers with weak gains, flow policies with
       noisy sampling): `search` runs to train on, `search_val` runs to calibrate on.
    2. Two `judge.trace` learner jobs (sealed `success_judge` artifacts, role `instrument`), which see only the last
       recorded observation of an episode: one from proprioception plus the observed target position, one from
       proprioception only.
    3. `calibrate_judge` (what `hylo calibrate` does) on runs the judges were not built from: AUC, Brier score,
       expected calibration error, the lowest threshold reaching 95% precision, the recall there, and whether the
       judge is `ready` to label.
    4. Labelling: episodes from another "cell" are ingested as an external run without outcomes; `label_run` (what
       `hylo label` does) labels the ones the ready judge scores above its threshold. The hidden simulator outcome
       measures what the labels are worth. A judge that is not ready, and a run with simulator outcomes, are refused.

Command (from the harness repository root)
    uv run python examples/learn/02_judge_calibration.py [--quick] [--out DIR]

Typical runtime: about 1.5 min on CPU (--quick: about 30 s).
Requires: hylo-env-mujoco (hylo-bench), hylo-instruments, hylo-policies.
Output: <out>/result.json, the runs (<out>/runs, or runs-quick), the judges (<out>/judges), the external run (<out>/external).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from _common import EXPERT, parse_args, reach_config, require, table, write_result

SLUG = "judge_calibration"
PROPRIO = ("qpos", "qvel", "tip_pos")


def agents(flow: str) -> dict[str, dict]:
    st = lambda n: {"type": "steered", "params": {"sampler": {"type": "policy.flow", "params": {"path": flow}},  # noqa: E731
                                                  "steering": {"noise_scale": n}}}
    return {"ik-kp0.015": {"type": "baseline.planar_ik", "params": {"kp": 0.015}},
            "ik-kp0.02": {"type": "baseline.planar_ik", "params": {"kp": 0.02}},
            "ik-kp0.03": {"type": "baseline.planar_ik", "params": {"kp": 0.03}},
            "flow-noise1": st(1.0), "flow-noise2": st(2.0), "flow-noise3": st(3.0)}


def episodes_of(run_dir: Path):
    """A finished run as external episodes without outcomes (what another robot's logs look like), plus the hidden
    truth kept aside for scoring the labels."""
    from hylo.core.datasource import ExternalEpisode, ExternalFrame
    from hylo.core.trace import TraceStore

    log = json.loads((run_dir / "evallog.json").read_text())
    store, eps, truth = TraceStore(run_dir), [], {}
    for s in log["samples"]:
        tr = store.read(s["episode_id"])
        frames = [ExternalFrame(state={k: np.asarray(v) for k, v in e["payload"]["state"].items()})
                  for e in tr.of_type("observation")]
        eid = f"{run_dir.name[-6:]}-{s['episode_id']}"
        eps.append(ExternalEpisode(episode_id=eid, task_id=s["task_id"], instruction="touch the red target", frames=frames))
        truth[eid] = s["status"] == "success"
    return eps, truth


def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.instruments.judges import TraceSuccessJudge
    from hylo.learn.calibrate import calibrate_judge, label_run, summarize
    from hylo.learn.data import ingest
    from hylo.learn.jobs import run_job

    t0 = time.time()
    shutil.rmtree(a.out / "external", ignore_errors=True)
    # a flow policy for diverse failures (demonstrations on search only)
    demos = a.out / "demos"
    if not (demos / ".done").exists():
        shutil.rmtree(demos, ignore_errors=True)
        run_experiment(reach_config(EXPERT, "learn-judge-demos", protocol="rsi"), split="search", run_root=demos, label="expert")
        (demos / ".done").touch()
    run_job(TrainJob(learner="policy.flow", inputs=[str(demos)], out=str(a.out / "flow"),
                            params={"epochs": 60 if a.quick else 300}))
    flow_path = str(a.out / "flow")

    # ---- 1. development runs with mixed outcomes
    states = 100 if a.quick else 300                                    # search: 40%, search_val: 20% of the seeds
    runs = a.out / ("runs-quick" if a.quick else "runs")
    if not (runs / ".done").exists():
        shutil.rmtree(runs, ignore_errors=True)
        for name, agent in agents(flow_path).items():
            cfg = reach_config(agent, f"learn-judge-{name}", protocol="rsi", rsi_states=states)
            for split in ("search", "search_val"):
                run_experiment(cfg, split=split, run_root=runs / split, label=f"{name}-{split}")
        (runs / ".done").touch()
    stat = {}
    for split in ("search", "search_val"):
        ss = [s for rd in sorted((runs / split).iterdir()) for s in json.loads((rd / "evallog.json").read_text())["samples"]]
        stat[split] = {"episodes": len(ss), "success_rate": sum(s["status"] == "success" for s in ss) / len(ss)}
    print("development runs: " + ", ".join(f"{k} {v['episodes']} episodes (success {v['success_rate']:.2f})" for k, v in stat.items()))

    # ---- 2. two judges, trained on search runs only
    judges = {}
    for name, keys in (("proprio + target", (*PROPRIO, "target_pos")), ("proprio only", PROPRIO)):
        out = a.out / "judges" / name.replace(" + ", "-").replace(" ", "-")
        if out.exists():
            shutil.rmtree(out)                                          # calibration rewrites the manifest: rebuild
        t = time.time()
        m = run_job(TrainJob(learner="judge.trace", inputs=[str(runs / "search")], out=str(out), splits=("search",),
                             params={"cameras": [], "state_keys": list(keys), "workers": a.workers,
                                     **({"epochs": 60} if a.quick else {})}))
        judges[name] = {"path": out, "digest": m.digest, "train_metrics": m.metrics, "train_s": round(time.time() - t, 1)}

    uncalibrated = a.out / "judges" / "uncalibrated-copy"
    shutil.rmtree(uncalibrated, ignore_errors=True)
    shutil.copytree(judges["proprio only"]["path"], uncalibrated)       # kept as trained: no calibration report

    # ---- 3. calibration on search_val runs (not in either judge's provenance)
    for name, j in judges.items():
        rep = calibrate_judge(TraceSuccessJudge(str(j["path"])), [runs / "search_val"], judge_artifact=j["path"])
        j["calibration"] = summarize(rep)
    print("\ncalibration against simulator outcomes (search_val runs, not used for training)")
    table([[n, j["calibration"]["n"], f"{j['calibration']['base_rate']:.2f}", f"{j['calibration']['auc']:.3f}",
            f"{j['calibration']['brier']:.3f}", f"{j['calibration']['ece']:.3f}",
            f"{j['calibration']['threshold']:.3f}" if j["calibration"]["threshold"] is not None else "-",
            f"{j['calibration']['precision']:.3f}" if j["calibration"]["precision"] is not None else "-",
            f"{j['calibration']['recall']:.3f}" if j["calibration"]["recall"] is not None else "-",
            j["calibration"]["ready"]] for n, j in judges.items()],
          ["judge", "episodes", "base rate", "AUC", "Brier", "ECE", "thr@95%prec", "precision", "recall", "ready"])

    # ---- 4. labelling an external run that has no outcomes
    other = a.out / "other-cell"
    shutil.rmtree(other, ignore_errors=True)
    n_other = 25 if a.quick else 100
    for name in ("ik-kp0.02", "flow-noise2"):                          # standard `train` seeds: not a dev split
        run_experiment(reach_config(agents(flow_path)[name], f"learn-judge-other-{name}", train=n_other), split="train",
                       run_root=other, label=name)
    eps, truth = [], {}
    for rd in sorted(other.iterdir()):
        e, t_ = episodes_of(rd)
        eps += e
        truth.update(t_)
    ext = ingest(eps, a.out / "external", name="other-cell", origin="examples: MuJoCo reach, outcomes withheld",
                 licence="Apache-2.0")
    print(f"\ningested {ext['episodes']} episodes as {ext['split']} ({ext['labeled']} labelled)")
    labelling, refusals = {}, {}
    for name, j in judges.items():
        run_dir = Path(ext["run_dir"] + "-" + name.replace(" + ", "-").replace(" ", "-"))   # one copy per judge
        shutil.copytree(ext["run_dir"], run_dir)
        try:
            res = label_run(run_dir, TraceSuccessJudge(str(j["path"])), j["path"])
        except ValueError as exc:
            refusals[name] = str(exc)
            continue
        log = json.loads((run_dir / "evallog.json").read_text())
        lab = [s for s in log["samples"] if s["status"] == "success"]
        tp = sum(truth[s["episode_id"]] for s in lab)
        n_true = sum(truth.values())
        labelling[name] = {**res, "true_successes": n_true, "labelled_correct": tp,
                           "precision": tp / len(lab) if lab else None, "recall": tp / n_true if n_true else None,
                           "false_labels": len(lab) - tp}
    try:
        label_run(Path(ext["run_dir"]), TraceSuccessJudge(str(uncalibrated)), uncalibrated)
    except ValueError as exc:
        refusals["uncalibrated judge"] = str(exc)
    try:
        label_run(next((runs / "search_val").iterdir()), TraceSuccessJudge(str(judges["proprio + target"]["path"])),
                  judges["proprio + target"]["path"])
    except ValueError as exc:
        refusals["simulator run"] = str(exc)
    for name, r in labelling.items():
        print(f"labels by '{name}' (threshold {r['threshold']:.3f}): {r['labeled_success']} labelled success, "
              f"{r['left_unlabeled']} left unlabelled; against the withheld outcomes: precision {r['precision']:.3f}, "
              f"recall {r['recall']:.3f} ({r['false_labels']} false labels, {r['true_successes']} true successes)")
    for name, why in refusals.items():
        print(f"refused ({name}): {why}")

    for j in judges.values():
        j["path"] = str(j["path"])
    write_result(a.out, {"example": SLUG, "quick": a.quick, "development_runs": stat, "judges": judges,
                         "external": {k: ext[k] for k in ("run_id", "split", "episodes", "artifact")},
                         "labelling": labelling, "refusals": refusals, "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
