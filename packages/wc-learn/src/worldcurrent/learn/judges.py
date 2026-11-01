"""Learner "judge.trace": a success judge trained on development traces (RFC-0004 A3).

Training data, from development-split runs only:
- the last recorded frame of every finished episode, labelled with the simulator outcome;
- early frames (progress < `negative_before`) of successful episodes as extra negatives: same scene, task not yet
  done, which teaches the scene change rather than the scene identity.
The sealed artifact is loaded at run time by worldcurrent.agents.judges.TraceSuccessJudge.

    learner: {learner: judge.trace, params: {cameras: [agentview, wrist]}}  -> sealed success_judge artifact
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from worldcurrent.agents.critic import ChunkCritic, FeatureConfig
from worldcurrent.agents.judges import JudgeConfig, imaged_frames, judge_features
from worldcurrent.core.artifact import ArtifactManifest, Provenance, check_splits, development_runs, write_manifest
from worldcurrent.core.trace import TraceStore

from .calibrate import calibration_report


def _episode_rows(job: tuple[str, str, str, dict[str, Any], float]) -> list[tuple[np.ndarray, int, int]]:
    run_dir, episode_id, status, cfg_d, negative_before = job
    cfg = JudgeConfig.model_validate(cfg_d)
    tr = TraceStore(run_dir).read(episode_id)
    task_id = tr.first("episode.start")["payload"]["task_id"]
    frames = imaged_frames(tr, cfg.cameras)
    if not frames:
        return []
    n_obs = sum(1 for _ in tr.of_type("observation"))
    rows = [(judge_features(frames[-1][1]["state"], frames[-1][1]["images"], task_id, cfg), int(status == "success"), 1)]
    if status == "success":
        rows += [(judge_features(p["state"], p["images"], task_id, cfg), 0, 0) for i, p in frames[:-1]
                 if i / max(1, n_obs) < negative_before]
    return rows                                    # (features, label, is_final_frame)


class TraceJudgeLearner:
    """Learner "judge.trace": job.params = JudgeConfig fields + negative_before, members, hidden, epochs, workers."""

    name = "worldcurrent.learn.judges@0.1"

    def fit(self, job) -> ArtifactManifest:
        p = dict(job.params)
        splits = check_splits(job.splits)
        negative_before = float(p.pop("negative_before", 0.5))
        kw = {k: p.pop(k) for k in ("members", "hidden", "epochs") if k in p}
        workers = int(p.pop("workers", 16))
        task_prefix = p.pop("task_prefix", None)
        sources = development_runs(job.inputs, splits)
        samples = [(str(rd), s["episode_id"], s["status"], s["task_id"], ref.split) for rd, ref in sources
                   for s in json.loads((rd / "evallog.json").read_text())["samples"]
                   if s["status"] in ("success", "failure") and (not task_prefix or s["task_id"].startswith(task_prefix))]
        if not samples:
            raise SystemExit("no finished development-split episodes")
        cfg = JudgeConfig.model_validate({**p, "tasks": tuple(sorted({s[3] for s in samples}))})
        jobs = [(rd, eid, st, cfg.model_dump(mode="json"), negative_before) for rd, eid, st, _, _ in samples]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_episode_rows, jobs, chunksize=4))
        split_of = [s[4] for s in samples]

        def stack(keep):
            r = [row for rs, sp in zip(rows, split_of) if keep(sp) for row in rs]
            return (np.stack([a for a, _, _ in r]), np.array([b for _, b, _ in r], float),
                    np.array([c for _, _, c in r], bool)) if r else (None, None, None)

        metrics: dict[str, Any] = {"episodes": len(samples), "frames": int(sum(len(r) for r in rows))}
        train_split = splits[0]
        xt, yt, _ = stack(lambda sp: sp == train_split)
        xv, yv, fv = stack(lambda sp: sp != train_split)
        if xt is not None and xv is not None and fv.any():          # final-frame accuracy on the other split
            m = ChunkCritic.fit(xt, yt, FeatureConfig(), **kw)
            rep = calibration_report(m.predict(xv[fv])[0], yv[fv].astype(int), min_n=1)
            metrics["val"] = {"split": [s for s in splits if s != train_split], "auc": rep.auc, "brier": round(rep.brier, 5),
                              "ece": rep.ece, "n": rep.n, "threshold_p95": rep.threshold, "recall_p95": rep.recall}
        x, y, _ = stack(lambda sp: True)
        model = ChunkCritic.fit(x, y, FeatureConfig(), **kw)
        model.meta.update({"judge": cfg.model_dump(mode="json"), "kind": "success_judge"})
        model.save(job.out)
        return write_manifest(job.out, kind="success_judge", role="instrument", metrics=metrics,
                              provenance=Provenance(builder=self.name, sources=[r for _, r in sources],
                                                    params={**job.params, "tasks": list(cfg.tasks)}))
