"""Online learning runner (extension point): propose -> roll out -> update, then seal.

The rollouts are ordinary evaluations of a development split (search by default; gate and held-out are refused),
recorded under `<out>/rollouts/iter-NNN/` like any run, so replay, judges and the catalog see them and the sealed
artifact's provenance lists them. The learner decides what to try and how to update: policy search (CEM),
on-policy RL (a GRPO-style learner would return its current policy and update from the rollouts' traces),
iterative self-training. Episodes per evaluation are the split's.

    TrainJob(learner="examples.cem", inputs=[], out="runs/learned/cem", params={
        "experiment": <ExperimentConfig as dict>, "iterations": 4, "split": "search", ...learner params})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hylo.eval.experiment import run_experiment
from hylo.core.artifact import ArtifactManifest, SourceRef, check_splits
from hylo.core.config import ExperimentConfig
from hylo.core.learning import TrainJob

RUNNER_KEYS = ("experiment", "iterations", "split")


def is_online(learner: Any) -> bool:
    return callable(getattr(learner, "policies", None)) and callable(getattr(learner, "update", None))


def run_online(job: TrainJob, learner: Any, *, log=print) -> ArtifactManifest:
    p = dict(job.params)
    if "experiment" not in p:
        raise ValueError("an online learner needs params.experiment (the experiment config to roll out in)")
    split = str(p.get("split", job.splits[0]))
    check_splits([split])
    cfg = ExperimentConfig.model_validate(p["experiment"])
    out = Path(job.out)
    learner.start(job, cfg.agent.model_dump(mode="json"))
    sources: list[SourceRef] = []
    history = []
    for it in range(int(p.get("iterations", 3))):
        results = []
        for i, spec in enumerate(learner.policies()):
            res = run_experiment(cfg.with_overrides({"agent": spec}), split=split, run_root=out / "rollouts" / f"iter-{it:03d}",
                                 label=f"it{it}-p{i}")
            sources.append(SourceRef(kind="run", id=res.log["run_id"], split=split, path=str(res.run_dir),
                                     n_episodes=len(res.log["samples"])))
            results.append({"agent": spec, "run_dir": str(res.run_dir), "metrics": res.metrics,
                            "samples": [{k: s.get(k) for k in ("key", "task_id", "seed", "status", "steps")} for s in res.log["samples"]]})
        m = learner.update(results)
        history.append({"iteration": it, **m})
        log(f"[online] iteration {it}: {json.dumps(m, default=str)[:200]}")
        if learner.finished():
            break
    (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
    return learner.seal(str(out), sources)
