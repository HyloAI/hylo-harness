"""Expert iteration for a LeRobot VLA (RFC-0002 L4): fine-tune the policy on its own successful episodes.

    python -m worldcurrent.policies.lerobot.selftrain --runs runs/<search-run> ... --out runs/selftrain/<name> \
        --pretrained lerobot/smolvla_libero --steps 300 --batch-size 8

1. export: successful episodes of development-split runs (search / search_val only; gate and held-out data
   must stay unseen) -> LeRobotDataset v3, cameras named as the policy expects, with provenance;
2. pin normalisation: the dataset's state/action statistics are replaced by the pretrained policy's own, so
   fine-tuning changes weights, not the input/output scaling (lerobot-train otherwise re-normalises with the
   new dataset's stats);
3. fine-tune with lerobot-train (action expert only by default: cheap on CPU, little room to overfit);
4. write selftrain.json: dataset + weights digests, training arguments, and the config patch that turns the
   result into an improvement candidate (it is then gated like any other candidate); the weights directory is
   sealed as a `policy_weights` artifact (RFC-0004) whose provenance lists the runs and the dataset artifact.
Runs in the policy venv (envs/libero/.venv: lerobot + torch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldcurrent.core.artifact import ArtifactManifest, Provenance, SourceRef, check_splits, read_manifest, write_manifest

from .export import POLICY_CAMERAS, export

PINNED = ("observation.state", "action")


def pretrained_dir(ref: str) -> Path:
    if Path(ref).is_dir():
        return Path(ref)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ref, local_files_only=True))


def pin_stats(dataset_dir: str | Path, pretrained: str) -> dict[str, Any]:
    """Overwrite meta/stats.json entries for state and action with the pretrained normaliser's statistics."""
    from safetensors.numpy import load_file

    src = sorted(pretrained_dir(pretrained).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not src:
        raise FileNotFoundError(f"no normaliser statistics in {pretrained}")
    flat = load_file(str(src[0]))
    path = Path(dataset_dir) / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    before = {k: stats.get(k, {}).get("mean") for k in PINNED}
    for key in PINNED:
        stats[key] = {name[len(key) + 1:]: arr.tolist() for name, arr in flat.items() if name.startswith(key + ".")}
    path.write_text(json.dumps(stats, indent=2))
    return {"source": src[0].name, "dataset_mean_before": before}


def finetune(dataset_dir: str | Path, pretrained: str, out_dir: str | Path, *, steps: int, batch_size: int, lr: float,
             num_workers: int = 2, expert_only: bool = True, seed: int = 0, extra: list[str] | None = None) -> Path:
    out_dir = Path(out_dir)
    cli = Path(sys.executable).with_name("lerobot-train")
    cmd = [str(cli), f"--policy.path={pretrained}", "--dataset.repo_id=local/worldcurrent-selftrain",
           f"--dataset.root={dataset_dir}", f"--output_dir={out_dir}", f"--steps={steps}", f"--batch_size={batch_size}",
           f"--save_freq={steps}", f"--log_freq={max(1, steps // 20)}", f"--num_workers={num_workers}", f"--seed={seed}",
           "--policy.device=cpu", "--policy.push_to_hub=false", "--wandb.enable=false",
           f"--policy.optimizer_lr={lr}", f"--policy.scheduler_warmup_steps={max(1, steps // 10)}",
           f"--policy.scheduler_decay_steps={steps}", f"--policy.scheduler_decay_lr={lr / 10}",
           f"--policy.train_expert_only={str(expert_only).lower()}", "--policy.freeze_vision_encoder=true",
           *(extra or [])]
    env = {**os.environ, "TORCHDYNAMO_DISABLE": "1", "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
           "TOKENIZERS_PARALLELISM": "false"}
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / "train.log", "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"lerobot-train exited with {rc}; see {out_dir.parent / 'train.log'}")
    weights = out_dir / "checkpoints" / "last" / "pretrained_model"
    if not (weights / "config.json").exists():
        raise FileNotFoundError(f"no checkpoint at {weights}")
    return weights


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def selftrain(run_dirs: list[str], out: str, *, pretrained: str = "lerobot/smolvla_libero", steps: int = 300,
              batch_size: int = 8, lr: float = 1e-5, splits: tuple[str, ...] = ("search", "search_val"),
              expert_only: bool = True, num_workers: int = 2, seed: int = 0,
              patch_path: str = "agent.params.params.repo_id") -> dict[str, Any]:
    check_splits(splits)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = export(run_dirs, str(root / "dataset"), only_success=True, cameras=POLICY_CAMERAS, splits=splits)
    pinned = pin_stats(root / "dataset", pretrained)
    ds = read_manifest(root / "dataset")                 # pinning rewrote the stats: seal the dataset again
    write_manifest(root / "dataset", kind=ds.kind, role=ds.role, metrics=ds.metrics, provenance=ds.provenance.model_copy(
        update={"params": {**ds.provenance.params, "stats_pinned_to": pretrained}}))
    t1 = time.time()
    weights = finetune(root / "dataset", pretrained, root / "train", steps=steps, batch_size=batch_size, lr=lr,
                       num_workers=num_workers, expert_only=expert_only, seed=seed)
    report = {"schema": "wc.selftrain/v0.1", "pretrained": pretrained, "runs": run_dirs, "splits": list(splits),
              "dataset": data, "pinned_stats": pinned, "weights": str(weights.resolve()),
              "weights_sha256_16": _sha16(weights / "model.safetensors"),
              "train": {"steps": steps, "batch_size": batch_size, "lr": lr, "expert_only": expert_only, "seed": seed},
              "export_s": round(t1 - t0, 1), "train_s": round(time.time() - t1, 1),
              "candidate_patch": {patch_path: str(weights.resolve())}}
    dataset = read_manifest(root / "dataset")
    report["artifact"] = write_manifest(weights, kind="policy_weights", role="behaviour", name=f"{root.name}-weights",
                                        metrics={"dataset": data, "train": report["train"]},
                                        provenance=Provenance(builder=ExpertIterationLearner.name, sources=[
                                            *dataset.provenance.sources,
                                            SourceRef(kind="artifact", id=dataset.digest, path=str(root / "dataset")),
                                            SourceRef(kind="external", id=pretrained, split="external:pretrained")],
                                            params={**report["train"], "pretrained": pretrained})).digest
    (root / "selftrain.json").write_text(json.dumps(report, indent=2))
    return report


class ExpertIterationLearner:
    """Learner interface (core.learning): job.params are `selftrain` keyword arguments; seals the weights dir."""

    name = "worldcurrent.policies.lerobot.selftrain@0.1"

    def fit(self, job) -> ArtifactManifest:
        report = selftrain(list(job.inputs), job.out, splits=tuple(job.splits), **dict(job.params))
        return read_manifest(report["weights"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="development-split run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default="lerobot/smolvla_libero")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--splits", nargs="+", default=["search", "search_val"])
    ap.add_argument("--full", action="store_true", help="train the VLM too, not only the action expert")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if any(s.startswith(("gate", "heldout")) for s in a.splits):
        ap.error("gate / heldout episodes must not be used for training")
    print(json.dumps(selftrain(a.runs, a.out, pretrained=a.pretrained, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                               splits=tuple(a.splits), expert_only=not a.full, num_workers=a.num_workers, seed=a.seed),
                     indent=2))


if __name__ == "__main__":
    main()
