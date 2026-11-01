import json
import sys

import pytest

from hylo.core.artifact import SplitLeak, read_manifest
from hylo.core.learning import TrainJob
from hylo.learn.calibrate import calibration_report
from hylo.learn.jobs import launch, run_job

LEARNER = "hylo.testing.learners:SummaryLearner"


def _run(root, name, split, statuses):
    rd = root / name
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({"benchmark": {"split": split}}))
    (rd / "evallog.json").write_text(json.dumps({"run_id": name, "samples": [
        {"episode_id": f"e{i}", "status": s} for i, s in enumerate(statuses)]}))
    return rd


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    _run(root, "a-search", "search", ["success", "failure"])
    _run(root, "a-val", "search_val", ["success", "success"])
    _run(root, "a-gate", "gate", ["success"] * 5)
    _run(root, "a-ho", "heldout", ["failure"] * 5)
    return root


def test_run_job_seals_once_and_reuses_the_artifact(runs, tmp_path):
    job = TrainJob(learner=LEARNER, inputs=[str(runs)], out=str(tmp_path / "out"), params={"kp": 2.0})
    m = run_job(job)
    assert m.metrics["episodes"] == 4 and m.metrics["success_rate"] == 0.75     # gate / heldout never read
    assert m.provenance.splits() == {"search", "search_val"} and m.provenance.params["job_digest"] == job.digest()
    mtime = (tmp_path / "out" / "summary.json").stat().st_mtime_ns
    assert run_job(job).digest == m.digest and (tmp_path / "out" / "summary.json").stat().st_mtime_ns == mtime
    with pytest.raises(SplitLeak):
        run_job(job.model_copy(update={"splits": ("search", "gate")}))


def test_launch_runs_the_learner_in_another_interpreter(runs, tmp_path):
    job = TrainJob(learner=LEARNER, inputs=[str(runs)], out=str(tmp_path / "sub"))
    m = launch(job, python=sys.executable, timeout_s=120)
    assert read_manifest(tmp_path / "sub").digest == m.digest and m.metrics["episodes"] == 4


def test_calibration_report_finds_the_precision_threshold():
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95] * 10
    labels = [0, 0, 0, 0, 1, 1, 1, 1] * 10
    rep = calibration_report(scores, labels, target_precision=0.95)
    assert rep.auc == pytest.approx(1.0) and rep.threshold == 0.7 and rep.recall == 1.0 and rep.ready
    weak = calibration_report([0.5] * 60, [0, 1] * 30)
    assert weak.auc == pytest.approx(0.5) and weak.threshold is None and not weak.ready
