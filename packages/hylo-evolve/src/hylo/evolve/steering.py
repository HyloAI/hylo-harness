"""Trace-based failure diagnosis and a rule-based proposer for frozen-policy steering (RFC-0002 L0/L1).

Diagnosis reads only the search split's traces (it may use privileged state; the gate and held-out splits never
reach the proposer). The proposer turns failure buckets into steering patches, each with a falsifiable hypothesis.
An LLM proposer implements the same `Proposer` protocol (hylo.evolve.protocol).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping

import numpy as np

from hylo.eval.runner import EvalResult
from hylo.core.config import ExperimentConfig
from hylo.core.trace import TraceStore

from .candidate import Candidate
from .protocol import ArchiveEntry


def episode_features(trace, eef_key: str = "eef_pos", gripper_key: str = "gripper_qpos",
                     still_eps: float = 0.0015, closed_below: float = 0.06) -> dict[str, float]:
    obs = [e["payload"]["state"] for e in trace.events if e["type"] == "observation"]
    feats = {"steps": len(obs) - 1, "max_still": 0.0, "grasp_attempts": 0.0}
    if obs and eef_key in obs[0]:
        eef = np.array([o[eef_key] for o in obs])
        still, run = 0, 0
        for d in np.linalg.norm(np.diff(eef, axis=0), axis=1):
            run = run + 1 if d < still_eps else 0
            still = max(still, run)
        feats["max_still"] = float(still)
    if obs and gripper_key in obs[0]:
        g = np.array([o[gripper_key] for o in obs])
        width = g[:, 0] - g[:, -1] if g.shape[1] > 1 else g[:, 0]
        closed = (width < closed_below).astype(int)
        feats["grasp_attempts"] = float(np.sum(np.diff(closed) == 1) + closed[0])
    return feats


def diagnose_traces(result: EvalResult, stall_steps: int = 30, regrasp: int = 3, **kw: Any) -> dict[str, Any]:
    """Failure buckets: stall (a still period >= stall_steps), regrasp (>= regrasp grasp attempts), other."""
    store = TraceStore(result.run_dir)
    buckets: Counter[str] = Counter()
    per_task: dict[str, Counter] = {}
    examples: dict[str, list[str]] = {}
    n_fail = 0
    successes: list[str] = []
    for s in result.log["samples"]:
        if s["status"] == "success":
            successes.append(s["episode_id"])
            continue
        n_fail += 1
        f = episode_features(store.read(s["episode_id"]), **kw)
        labels = [b for b, hit in (("stall", f["max_still"] >= stall_steps), ("regrasp", f["grasp_attempts"] >= regrasp)) if hit] or ["other"]
        for b in labels:
            buckets[b] += 1
            per_task.setdefault(s["task_id"].split("/")[-1], Counter())[b] += 1
            examples.setdefault(b, []).append(s["episode_id"])
    n = len(result.log["samples"])
    return {"n_episodes": n, "n_failures": n_fail, "success": result.metrics["success_mean"],
            "buckets": dict(buckets), "bucket_rate": {k: v / max(n_fail, 1) for k, v in buckets.items()},
            "per_task": {k: dict(v) for k, v in per_task.items()}, "examples": {k: v[:5] for k, v in examples.items()},
            "success_examples": successes[:5], "run_dir": str(result.run_dir)}


STEERING_PATH = "agent.params.params.steering"


def _get(d: Mapping[str, Any], path: str) -> Any:
    for p in path.split("."):
        d = d.get(p, {}) if isinstance(d, Mapping) else {}
    return d


class SteeringRuleProposer:
    """Maps failure buckets to steering patches. Each candidate is one coherent change with a hypothesis."""

    name = "rule.steering@0.1"

    def __init__(self, min_rate: float = 0.2, path: str = STEERING_PATH) -> None:
        self.min_rate, self.path = min_rate, path

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        current = dict(_get(cfg.model_dump(mode="json"), self.path) or {})
        rules = list(current.get("rules", []))
        names = {r["name"] for r in rules}
        rate = diagnosis.get("bucket_rate", {})
        out: list[tuple[dict, str]] = []
        if rate.get("stall", 0) >= self.min_rate and "stall_backoff" not in names:
            for thr in (20, 30):
                rule = {"name": "stall_backoff", "feature": "still_steps", "value": thr, "action": "backoff",
                        "params": {"steps": 8, "dz": 0.5, "open_gripper": 1}, "cooldown": 40, "max_fires": 3}
                out.append(({**current, "rules": [*rules, rule]},
                            f"{rate['stall']:.0%} of failures stall; backing off after {thr} still steps converts some stalls into successes"))
        if rate.get("regrasp", 0) >= self.min_rate and "regrasp_resample" not in names:
            for scale in (1.5, 2.0):
                rule = {"name": "regrasp_resample", "feature": "grasp_attempts", "value": 3, "action": "resample",
                        "params": {"noise_scale": scale}, "cooldown": 60, "max_fires": 2}
                out.append(({**current, "rules": [*rules, rule]},
                            f"{rate['regrasp']:.0%} of failures retry grasps; resampling with noise x{scale} after 3 attempts escapes the repeated failure"))
        if current.get("n_samples", 1) < 4:
            out.append(({**current, "n_samples": 4, "selector": "gripper_vote_medoid"},
                        "voting over 4 sampled chunks (gripper majority, medoid) avoids unlucky samples"))
        if (current.get("execution_steps") or 50) > 25:
            out.append(({**current, "execution_steps": 25}, "replanning every 25 steps instead of 50 reacts to drift sooner"))
        seen = {json.dumps(e.candidate.patch, sort_keys=True) for e in archive}
        cands = []
        for steering, hyp in out:
            c = Candidate(patch={self.path: steering}, rationale=hyp, parent_id=parent_id, generator=self.name,
                          evidence={"buckets": diagnosis.get("buckets", {})})
            if json.dumps(c.patch, sort_keys=True) not in seen:
                cands.append((c, hyp))
        return cands
