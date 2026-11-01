from worldcurrent.eval.experiment import run_experiment
from worldcurrent.catalog.episodes import TraceCache, frame_array, timeline
from worldcurrent.catalog.index import Catalog
from worldcurrent.core.config import ExperimentConfig

CFG = ExperimentConfig(name="cat", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                       env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})


def test_catalog_lists_runs_replays_episodes_and_pairs_runs(tmp_path):
    a = run_experiment(CFG, run_root=tmp_path / "runs", label="a")
    b = run_experiment(CFG.with_overrides({"agent.params.kp": 2.0}), run_root=tmp_path / "runs", label="b")
    cat = Catalog(tmp_path / "runs")
    assert {r["run_id"] for r in cat.runs()} == {a.log["run_id"], b.log["run_id"]}
    ep = a.log["samples"][0]["episode_id"]
    tl = timeline(TraceCache().get(cat.run_dir(a.log["run_id"]), ep))
    assert tl["steps"] == len(tl["actions"]) > 0
    cmp = cat.compare(a.log["run_id"], b.log["run_id"])
    assert cmp["paired"]["n_pairs"] == len(a.log["samples"]) and cmp["paired"]["candidate_wins"] > 0
    assert cat.campaigns() == [] and cat.artifacts() == []
    try:
        frame_array(TraceCache().get(cat.run_dir(a.log["run_id"]), ep), 0, "agentview")
        raise AssertionError("the test double records no camera")
    except KeyError:
        pass
