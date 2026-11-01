"""Run learner jobs (RFC-0004): in this process, or with another venv's interpreter.

`launch` starts `python -m hylo.learn.jobs job.json` with any interpreter that has hylo-learn installed
(it is light: hylo-core + hylo-instruments), which resolves the learner through the plugin registry ("learners" group).
A job whose output is already sealed with the same job digest is not re-run (resumable campaigns).

    job = TrainJob(learner="critic.chunk_value", inputs=["runs/rsi/cycle-01/runs"], out="runs/critics/c1",
                   params={"image_camera": "agentview"})
    manifest = launch(job, python="envs/libero/.venv/bin/python")
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from hylo.core.artifact import MANIFEST, ArtifactError, ArtifactManifest, check_splits, verify
from hylo.core.learning import TrainJob


def _sealed(job: TrainJob) -> ArtifactManifest | None:
    out = Path(job.out)
    if not (out / MANIFEST).is_file():
        return None
    try:
        m = verify(out)
    except ArtifactError:
        return None
    return m if m.provenance.params.get("job_digest") == job.digest() else None


def run_job(job: TrainJob) -> ArtifactManifest:
    """Run a learner in this process (its dependencies must be importable here)."""
    from hylo.core.registry import registry

    check_splits(job.splits)
    done = _sealed(job)
    if done is not None:
        return done
    learner = registry.build("learners", {"type": job.learner, "params": {}})
    from .online import is_online, run_online

    m = run_online(job, learner) if is_online(learner) else learner.fit(job)
    if m.provenance.params.get("job_digest") != job.digest():   # stamp the job so a rerun can be skipped
        from hylo.core.artifact import write_manifest

        prov = m.provenance.model_copy(update={"params": {**m.provenance.params, "job_digest": job.digest(),
                                                          "learner": job.learner}})
        m = write_manifest(job.out, kind=m.kind, role=m.role, provenance=prov, name=m.name, metrics=m.metrics,
                           mutable=m.mutable, calibration=m.calibration)
    return m


def launch(job: TrainJob, *, python: str | None = None, env: Mapping[str, str] | None = None,
           timeout_s: float | None = None, log_path: str | None = None) -> ArtifactManifest:
    """Run a job with another interpreter (e.g. the policy's venv) and return the verified manifest."""
    check_splits(job.splits)
    done = _sealed(job)
    if done is not None:
        return done
    out = Path(job.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = out.parent / f"{out.name}.job.json"
    spec.write_text(job.model_dump_json(indent=2))
    cmd = [python or sys.executable, "-m", "hylo.learn.jobs", str(spec)]
    log = open(log_path or out.parent / f"{out.name}.log", "w")
    with log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(env) if env else None,
                            timeout=timeout_s).returncode
    if rc != 0:
        raise RuntimeError(f"learner {job.learner} failed ({rc}); see {log.name}")
    return verify(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m hylo.learn.jobs JOB.json", file=sys.stderr)
        return 2
    m = run_job(TrainJob.model_validate_json(Path(args[0]).read_text()))
    print(json.dumps({"kind": m.kind, "digest": m.digest, "metrics": m.metrics}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
