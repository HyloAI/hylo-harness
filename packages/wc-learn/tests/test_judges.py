import json

import numpy as np
import pytest

from worldcurrent.agents.judges import TraceSuccessJudge
from worldcurrent.learn.judges import TraceJudgeLearner
from worldcurrent.core.artifact import SplitLeak, verify
from worldcurrent.core.events import Event
from worldcurrent.learn.calibrate import calibrate_judge
from worldcurrent.core.learning import TrainJob
from worldcurrent.core.trace import TraceStore

STATE = {"eef_pos": [0.0, 0.0, 0.0], "eef_axis_angle": [0.0, 0.0, 0.0], "gripper_qpos": [0.04, -0.04]}


def _scene(done: bool, rng) -> dict:
    """A 16x16 'camera': the task is done when the bright square sits in the top-left corner."""
    img = rng.integers(0, 40, size=(16, 16, 3)).astype(np.uint8)
    y, x = (0, 0) if done else (8, 8)
    img[y:y + 8, x:x + 8] = 230
    return {"agentview": img, "wrist": img[::-1].copy()}


def _run(root, name, split, outcomes, rng):
    store = TraceStore(root / name)
    samples = []
    for i, ok in enumerate(outcomes):
        eid = f"{name}-e{i}"
        seq = 0
        store.write(Event(type="episode.start", episode_id=eid, seq=seq, payload={"task_id": "t/a", "split": split}))
        n = 10
        for k in range(n):
            seq += 1
            done = ok and k == n - 1
            store.write(Event(type="observation", episode_id=eid, seq=seq,
                              payload={"state": STATE, "images": _scene(done, rng), "instruction": "move it"}))
        store.close_episode(eid)
        samples.append({"episode_id": eid, "status": "success" if ok else "failure", "task_id": "t/a"})
    (root / name / "manifest.json").write_text(json.dumps({"benchmark": {"split": split}}))
    (root / name / "evallog.json").write_text(json.dumps({"run_id": name, "samples": samples}))


def test_trace_judge_learns_the_scene_change_and_calibrates_on_other_runs(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "runs"
    _run(root, "train", "search", [True, False] * 20, rng)
    _run(root, "val", "search_val", [True, False] * 20, rng)
    _run(root, "gate", "gate", [True] * 10, rng)
    job = TrainJob(learner="judge.trace", inputs=[str(root)], out=str(tmp_path / "judge"), splits=("search",),
                   params={"image_px": 4, "workers": 2, "epochs": 80, "members": 2})
    m = TraceJudgeLearner().fit(job)
    assert m.kind == "success_judge" and m.role == "instrument" and m.provenance.splits() == {"search"}
    judge = TraceSuccessJudge(str(tmp_path / "judge"))
    rep = calibrate_judge(judge, [root], judge_artifact=tmp_path / "judge", min_n=10)
    assert rep.runs == ["val"]                                    # trained-on and gate runs are both excluded
    assert rep.auc > 0.95 and rep.ready
    assert verify(tmp_path / "judge").calibration["auc"] == rep.auc   # stored in the (re-sealed) manifest
    with pytest.raises(SplitLeak):
        TraceJudgeLearner().fit(job.model_copy(update={"splits": ("gate",)}))
