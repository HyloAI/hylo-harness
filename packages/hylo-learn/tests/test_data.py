import json

import numpy as np
import pytest

from hylo.core.artifact import SourceRef, development_runs, verify
from hylo.core.datasource import ExternalEpisode, ExternalFrame
from hylo.learn.data import contamination_report, ingest
from hylo.core.events import Event
from hylo.core.instrument import Judgement
from hylo.learn.calibrate import label_run
from hylo.core.trace import TraceStore


def _episode(i, eef, success=None, instruction="put the bowl on the plate"):
    frames = [ExternalFrame(state={"eef_pos": np.array(eef, float) + 0.01 * k}, images={"agentview": np.full((8, 8, 3), k, np.uint8)},
                            action=np.zeros(7)) for k in range(5)]
    return ExternalEpisode(episode_id=f"demo-{i}", task_id="demo/bowl", instruction=instruction, frames=frames, success=success)


def _protected_run(root, eef):
    store = TraceStore(root / "gate-run")
    store.write(Event(type="observation", episode_id="g0", seq=0,
                      payload={"state": {"eef_pos": np.array(eef, float)}, "instruction": "Put the bowl on the plate."}))
    store.close_episode("g0")
    (root / "gate-run" / "manifest.json").write_text(json.dumps({"benchmark": {"split": "gate"}}))
    (root / "gate-run" / "evallog.json").write_text(json.dumps({"run_id": "gate-run", "samples": [
        {"episode_id": "g0", "task_id": "libero/x/KITCHEN_put_the_bowl_on_the_plate", "status": "success"}]}))


def test_ingest_writes_a_sealed_external_run_with_honest_labels(tmp_path):
    rep = ingest([_episode(0, [0, 0, 0], success=True), _episode(1, [1, 0, 0])], tmp_path, name="demos",
                 origin="unit-test", licence="CC-BY-4.0", image_every=2)
    log = json.loads((tmp_path / rep["run_id"] / "evallog.json").read_text())
    assert [s["status"] for s in log["samples"]] == ["success", "unlabeled"]      # no label is never a success
    assert log["samples"][0]["labels"]["success"]["by"] == "source"
    m = verify(tmp_path / rep["run_id"])
    assert m.kind == "dataset" and m.provenance.sources[0].licence == "CC-BY-4.0"
    tr = TraceStore(tmp_path / rep["run_id"]).read("demo-0")
    kept = [bool(e["payload"]["images"]) for e in tr.of_type("observation")]
    assert kept == [True, False, True, False, True]                               # every 2nd frame and the last
    # builders read external runs only when asked to
    assert development_runs([tmp_path]) == []
    assert [r.split for _, r in development_runs([tmp_path], ("search", "external"))] == ["external:demos"]


def test_contamination_flags_external_episodes_that_reuse_protected_init_states(tmp_path):
    _protected_run(tmp_path / "bench", [0.5, 0.0, 0.2])
    ingest([_episode(0, [0.5, 0.0, 0.2]), _episode(1, [0.4, 0.0, 0.2])], tmp_path / "ext", name="d", origin="t", licence="x")
    rep = contamination_report([tmp_path / "ext"], [tmp_path / "bench"], state_keys=("eef_pos",))
    assert rep["checked"] == 2 and not rep["clean"]
    assert [h["external"].split("/")[-1] for h in rep["matches"]] == ["demo-0"]


class _Judge:
    name = "test.judge"

    def judge(self, episode):
        return Judgement(p_success=0.99 if episode.episode_id == "demo-0" else 0.4)


def test_label_run_requires_a_calibrated_judge_and_only_labels_confident_successes(tmp_path):
    from hylo.core.artifact import Provenance, write_manifest

    rep = ingest([_episode(0, [0, 0, 0]), _episode(1, [1, 0, 0])], tmp_path / "ext", name="v", origin="t", licence="x")
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "w").write_text("0")
    write_manifest(judge_dir, kind="success_judge", role="instrument", provenance=Provenance(builder="t"))
    run = tmp_path / "ext" / rep["run_id"]
    with pytest.raises(ValueError, match="not calibrated"):
        label_run(run, _Judge(), judge_dir)
    write_manifest(judge_dir, kind="success_judge", role="instrument", provenance=Provenance(builder="t"),
                   calibration={"ready": True, "threshold": 0.9, "precision": 0.96})
    out = label_run(run, _Judge(), judge_dir)
    assert out["labeled_success"] == 1 and out["left_unlabeled"] == 1
    log = json.loads((run / "evallog.json").read_text())
    assert log["samples"][0]["labels"]["success"]["by"].startswith("judge:") and log["samples"][1]["status"] == "unlabeled"
    write_manifest(judge_dir, kind="success_judge", role="instrument", calibration={"ready": True, "threshold": 0.9},
                   provenance=Provenance(builder="t", sources=[SourceRef(kind="run", id=rep["run_id"], split="external:v")]))
    with pytest.raises(ValueError, match="built from"):
        label_run(run, _Judge(), judge_dir)
