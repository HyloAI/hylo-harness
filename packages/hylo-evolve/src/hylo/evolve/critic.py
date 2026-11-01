"""Chunk-critic proposer (RFC-0002 L3): learn Q(s, chunk) from the campaign's own development traces and propose
steering the frozen VLA with it (sample N chunks, execute the best-valued one).

    proposer: {type: critic.chunk_value, params: {data_roots: [runs/rsi-libero10/runs], n_samples: 4}}

The critic is trained in a subprocess (hylo.learn.critic) on search / search_val episodes only,
so the gate still judges it on unseen init states. The offline same-state pair accuracy is recorded as evidence;
a critic that cannot rank chunks from identical states better than chance (min_pair_accuracy) is not proposed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from hylo.core.config import ExperimentConfig
from hylo.core.registry import registry
from hylo.core.store import Store

from .candidate import Candidate
from .protocol import ArchiveEntry

CRITIC_FACTORY = "lerobot.critic_steered"          # agents plugin, resolved in the policy's own venv
CRITIC_LEARNER = "critic.chunk_value"


class CriticProposer:
    name = "critic.chunk_value@0.1"

    def __init__(self, data_roots: list[str] | None = None, out_root: str = "runs/critics", n_samples: int = 4,
                 betas: tuple[float, ...] = (0.5,), image_camera: str | None = "agentview", task_prefix: str | None = None,
                 min_pair_accuracy: float = 0.55, agent_path: str = "agent.params",
                 python: str | None = None) -> None:
        self.data_roots, self.out_root, self.n_samples = list(data_roots or []), Path(out_root), n_samples
        self.betas, self.image_camera, self.task_prefix = tuple(betas), image_camera, task_prefix
        self.min_pair_accuracy, self.agent_path = min_pair_accuracy, agent_path
        self.python = python or sys.executable          # the critic trainer only needs numpy + Hylo

    def _train(self, roots: list[str], out: Path) -> dict[str, Any]:
        module = registry.reference("learners", CRITIC_LEARNER).partition(":")[0]   # the trainer's CLI module
        cmd = [self.python, "-m", module, "--runs", *roots, "--out", str(out),
               "--splits", "search", "search_val"]
        if self.image_camera:
            cmd += ["--image-camera", self.image_camera]
        if self.task_prefix:
            cmd += ["--task-prefix", self.task_prefix]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"critic training failed:\n{res.stderr[-2000:]}")
        meta = json.loads((out / "critic.json").read_text())
        return meta

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        roots = [*self.data_roots]
        if diagnosis.get("run_dir"):
            roots.append(str(Path(diagnosis["run_dir"]).parent))     # this cycle's runs, incl. the parent's search run
        version = sum(1 for e in archive if e.candidate.generator == self.name)
        out = self.out_root / f"{parent_id or 'base'}-{version}"
        meta = self._train(roots, out)
        cv = meta.get("trained_on", {}).get("cv", {})
        acc = cv.get("same_state_pair_accuracy")
        if acc is None or acc < self.min_pair_accuracy:
            return []                                    # the critic cannot tell chunks apart: steering with it is noise
        agent = cfg.model_dump(mode="json")
        for p in self.agent_path.split("."):
            agent = agent[p]
        steering = {**(agent.get("params", {}).get("steering") or {}), "n_samples": self.n_samples,
                    "selector": "gripper_vote_medoid", "seed_noise": True}
        cands = []
        for beta in self.betas:
            hyp = (f"a chunk critic that ranks same-state chunks correctly {acc:.0%} of the time offline "
                   f"(n={cv.get('same_state_pairs')}) picks better chunks among {self.n_samples} samples (beta={beta})")
            patch = {f"{self.agent_path}.ref": CRITIC_FACTORY,
                     f"{self.agent_path}.params.steering": steering,
                     f"{self.agent_path}.params.critic": {"path": Store().ref_for(out), "beta": beta, "within_vote": True}}
            cands.append((Candidate(patch=patch, rationale=hyp, parent_id=parent_id, generator=self.name,
                                    evidence={"unit": "L3", "critic": str(out), "cv": cv,
                                              "weights_sha256_16": meta.get("weights_sha256_16")}), hyp))
        return cands
