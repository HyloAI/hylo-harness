"""Expert-iteration proposer (RFC-0002 L4): improve the policy's weights from its own successes.

Each call collects the parent's episodes on the search split with full camera recording, fine-tunes the policy on
the successful ones in the policy's own venv (worldcurrent.policies.lerobot.selftrain), and proposes the new
weights as a config patch. The candidate then goes through the same cascade and sequential gate as any steering
change, so a fine-tune that only memorised the search states is rejected on unseen init states.

    proposer: {type: selftrain.expert_iteration, params: {steps: 300, batch_size: 8, lr: 1.0e-5}}

Only development splits are trained on; the proposer reuses the parent's `search` traces if they already have
per-step images, otherwise it records a fresh collection run (trace.image_every = 1).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.registry import registry
from worldcurrent.core.remote.protocol import child_env

from .candidate import Candidate
from .protocol import ArchiveEntry


def _get(d: Mapping[str, Any], path: str) -> Any:
    for p in path.split("."):
        d = d.get(p, {}) if isinstance(d, Mapping) else {}
    return d


class ExpertIterationProposer:
    name = "selftrain.expert_iteration@0.1"

    def __init__(self, out_root: str = "runs/selftrain", steps: int = 300, batch_size: int = 8, lr: float = 1e-5,
                 full: bool = False, collect_split: str = "search", min_successes: int = 20,
                 weights_path: str = "agent.params.params.repo_id", python_path: str = "agent.params.python",
                 env_path: str = "agent.params.env", timeout_s: float = 6 * 3600,
                 learner: str = "lerobot.expert_iteration") -> None:
        self.out_root, self.steps, self.batch_size, self.lr, self.full = Path(out_root), steps, batch_size, lr, full
        self.collect_split, self.min_successes, self.timeout_s = collect_split, min_successes, timeout_s
        self.weights_path, self.python_path, self.env_path = weights_path, python_path, env_path
        self.learner = learner

    def _collect(self, cfg: ExperimentConfig, out: Path) -> str:
        result = run_experiment(cfg.with_overrides({"trace.image_every": 1}), split=self.collect_split,
                                run_root=out / "collect", label=f"selftrain-collect-{self.collect_split}")
        return str(result.run_dir)

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        d = cfg.model_dump(mode="json")
        current = _get(d, self.weights_path)
        if not isinstance(current, str) or not current:
            raise ValueError(f"no policy weights reference at {self.weights_path}")
        tag = f"{parent_id or 'base'}-{sum(1 for e in archive if e.candidate.generator == self.name)}"
        out = self.out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        if cfg.trace.image_every == 1 and diagnosis.get("run_dir"):
            run_dir = str(diagnosis["run_dir"])
        else:
            run_dir = self._collect(cfg, out)
        n_success = sum(s["status"] == "success" for s in json.loads((Path(run_dir) / "evallog.json").read_text())["samples"])
        if n_success < self.min_successes:
            return []                                    # too little signal to imitate
        python = os.path.expanduser(_get(d, self.python_path) or "python")
        module = registry.reference("learners", self.learner).partition(":")[0]     # runs in the policy's venv
        cmd = [python, "-m", module, "--runs", run_dir, "--out", str(out / "train"),
               "--pretrained", current, "--steps", str(self.steps), "--batch-size", str(self.batch_size),
               "--lr", str(self.lr), "--splits", "search", "search_val", *(["--full"] if self.full else [])]
        env = child_env(_get(d, self.env_path) or None)
        with open(out / "selftrain.log", "w") as logf:
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=self.timeout_s).returncode
        if rc != 0:
            raise RuntimeError(f"selftrain failed ({rc}); see {out / 'selftrain.log'}")
        report = json.loads((out / "train" / "selftrain.json").read_text())
        hyp = (f"fine-tuning {'the whole policy' if self.full else 'the action expert'} on its own "
               f"{report['dataset']['episodes']} successful search episodes ({report['dataset']['frames']} frames, "
               f"{self.steps} steps) makes successful behaviour more likely on unseen init states")
        cand = Candidate(patch={self.weights_path: report["weights"]}, rationale=hyp, parent_id=parent_id,
                         generator=self.name,
                         evidence={"unit": "L4", "selftrain": str(out / "train" / "selftrain.json"),
                                   "weights_sha256_16": report["weights_sha256_16"], "source_run": run_dir,
                                   "pretrained": current, "successes": n_success})
        return [(cand, hyp)]
