"""Generic learning proposer (RFC-0004): train an artifact from this cycle's development runs and propose using it.

    proposer: {type: learn, params: {
        learner: critic.chunk_value,                       # "learners" plugin (runs in `python`'s venv)
        params: {image_camera: agentview, task_prefix: libero/libero_10/},
        python_path: agent.params.python,                  # or python: envs/libero/.venv/bin/python
        require: {cv.same_state_pair_accuracy: 0.55},      # manifest metrics gate (offline sanity, not the RSI gate)
        patch: {agent.params.ref: lerobot.critic_steered,
                agent.params.params.critic: {path: "{artifact}", beta: 0.5}},
        variants: [{agent.params.params.critic.beta: 0.0}],
        hypothesis: "a critic ranking same-state chunks {metrics.cv.same_state_pair_accuracy} picks better chunks"}}

The learner seals its output (artifact.json: kind, role, digest, the runs it read); the proposer verifies it before
proposing, and stage 0 of the gate verifies it again. Development splits only; a job whose artifact is already
sealed with the same job digest is reused (campaign resume). CriticProposer and ExpertIterationProposer are the
two original special cases of this pattern.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Mapping

from hylo.core.artifact import DEV_SPLITS, ArtifactManifest, check_splits
from hylo.core.config import ExperimentConfig
from hylo.core.learning import TrainJob
from hylo.learn.jobs import launch
from hylo.core.remote.protocol import child_env
from hylo.core.store import Store

from .candidate import Candidate
from .protocol import ArchiveEntry

_FIELD = re.compile(r"\{(artifact|metrics(?:\.[A-Za-z0-9_]+)+)\}")


def _get(d: Any, path: str) -> Any:
    for p in path.split("."):
        if not isinstance(d, Mapping) or p not in d:
            return None
        d = d[p]
    return d


def _fill(obj: Any, artifact: str, metrics: Mapping[str, Any]) -> Any:
    """Substitute {artifact} and {metrics.a.b} in every string of a patch template."""
    if isinstance(obj, dict):
        return {k: _fill(v, artifact, metrics) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, artifact, metrics) for v in obj]
    if not isinstance(obj, str):
        return obj
    whole = _FIELD.fullmatch(obj)
    if whole:                                             # a bare placeholder keeps the value's type
        return artifact if whole.group(1) == "artifact" else _get(metrics, whole.group(1)[len("metrics."):])
    return _FIELD.sub(lambda m: artifact if m.group(1) == "artifact"
                      else str(_get(metrics, m.group(1)[len("metrics."):])), obj)


class LearningProposer:
    def __init__(self, learner: str, patch: dict[str, Any], params: dict[str, Any] | None = None,
                 variants: list[dict[str, Any]] | None = None, data_roots: list[str] | None = None,
                 splits: tuple[str, ...] = DEV_SPLITS, out_root: str = "runs/learned", python: str | None = None,
                 python_path: str | None = None, env_path: str | None = None,
                 require: dict[str, float] | None = None, hypothesis: str = "", unit: str = "L3",
                 timeout_s: float | None = None, online: bool = False) -> None:
        self.learner, self.patch, self.params = learner, dict(patch), dict(params or {})
        self.variants = list(variants or [{}])
        self.data_roots, self.splits = list(data_roots or []), check_splits(splits)
        self.out_root, self.python, self.python_path, self.env_path = Path(out_root), python, python_path, env_path
        self.require, self.hypothesis, self.unit, self.timeout_s = dict(require or {}), hypothesis, unit, timeout_s
        self.online = online                     # online learners roll out in the parent's experiment config
        self.name = f"learn.{learner}@0.1"
        self.last: ArtifactManifest | None = None

    def _interpreter(self, d: Mapping[str, Any]) -> tuple[str, dict[str, str] | None]:
        python = self.python or (_get(d, self.python_path) if self.python_path else None) or sys.executable
        env = child_env(_get(d, self.env_path) or None) if self.env_path else None
        return str(Path(python).expanduser()), env

    def train(self, inputs: list[str], out: Path, d: Mapping[str, Any]) -> ArtifactManifest:
        params = {**self.params, "experiment": dict(d)} if self.online else self.params
        job = TrainJob(learner=self.learner, inputs=inputs, out=str(out), splits=self.splits, params=params)
        python, env = self._interpreter(d)
        return launch(job, python=python, env=env, timeout_s=self.timeout_s)

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        inputs = [*self.data_roots]
        if diagnosis.get("run_dir"):
            inputs.append(str(Path(diagnosis["run_dir"]).parent))     # this cycle's runs (non-dev splits are skipped)
        version = sum(1 for e in archive if e.candidate.generator == self.name)
        out = (self.out_root / f"{parent_id or 'base'}-{version}").resolve()
        d = cfg.model_dump(mode="json")
        m = self.last = self.train(inputs, out, d)
        for path, floor in self.require.items():
            v = _get(m.metrics, path)
            if not isinstance(v, (int, float)) or v < floor:
                return []                                    # offline sanity check failed: nothing worth gating
        ref = Store().ref_for(out)                           # artifact:<name>@<digest> when out is in the store
        hyp = _fill(self.hypothesis, ref, m.metrics) or f"{m.kind} learned by {self.learner} improves success"
        cands = []
        for i, variant in enumerate(self.variants):
            patch = _fill({**self.patch, **variant}, ref, m.metrics)
            ev = {"unit": self.unit, "artifact": {"id": ref, "path": str(out), "kind": m.kind, "role": m.role, "digest": m.digest,
                                                   "splits": sorted(m.provenance.splits())},
                  "learner": self.learner, "variant": i}
            h = hyp if len(self.variants) == 1 else f"{hyp} (variant {i}: {variant})"
            cands.append((Candidate(patch=patch, rationale=h, parent_id=parent_id, generator=self.name, evidence=ev), h))
        return cands
