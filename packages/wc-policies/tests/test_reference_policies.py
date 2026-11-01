import numpy as np
import pytest

from worldcurrent.core.artifact import SplitLeak, verify
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.learning import TrainJob
from worldcurrent.core.registry import registry
from worldcurrent.core.task import TaskInstance
from worldcurrent.eval.experiment import run_experiment
from worldcurrent.learn.jobs import run_job
from worldcurrent.testing.conformance import check_policy
from worldcurrent.testing.point_reach import PointReachEnv, reach_tasks

BASE = ExperimentConfig(name="ref", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
                        env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}},
                        safety={"max_delta": 0.25})
FAST = {"knn": {}, "bc": {"epochs": 60}, "flow": {"epochs": 120}, "tokens": {"epochs": 60}}


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    root = tmp_path_factory.mktemp("ref")
    run_experiment(BASE, split="search", run_root=root / "demos", label="expert")        # the demonstrator
    for m, params in FAST.items():
        run_job(TrainJob(learner=f"policy.{m}", inputs=[str(root / "demos")], out=str(root / m), params=params))
    run_job(TrainJob(learner="world_model.state", inputs=[str(root / "demos")], out=str(root / "wm"),
                     params={"env": {"type": "testing.point_reach"}, "epochs": 40}))
    return root


def _policy(art, m):
    if m == "mpc":
        return {"type": "policy.mpc", "params": {"world_model": str(art / "wm"), "objective": {"reach": ["pos", "goal"]}}}
    return {"type": f"policy.{m}", "params": {"path": str(art / m)}}


def _success(art, agent, tmp_path, label):
    return run_experiment(BASE.with_overrides({"agent": agent}), split="heldout", run_root=tmp_path, label=label).metrics["success_mean"]


@pytest.mark.parametrize("m", [*FAST, "mpc"])
def test_every_reference_policy_passes_the_policy_conformance_check(art, m):
    rep = check_policy(registry.build("policies", _policy(art, m)), PointReachEnv(), reach_tasks()[0])
    assert rep.ok, rep.summary()


@pytest.mark.parametrize("m", ["bc", "flow", "tokens", "mpc"])
def test_learned_policies_reach_held_out_goals(art, m, tmp_path):
    assert _success(art, {"type": "policy", "params": {"policy": _policy(art, m)}}, tmp_path, m) >= 0.8


def test_artifacts_are_sealed_behaviour_built_from_development_runs_only(art, tmp_path):
    m = verify(art / "flow")
    assert (m.kind, m.role) == ("policy_params", "behaviour") and m.provenance.splits() == {"search"}
    with pytest.raises(SplitLeak):
        run_job(TrainJob(learner="policy.bc", inputs=[str(art / "demos")], out=str(tmp_path / "x"), splits=("heldout",)))


def test_generative_policies_draw_their_noise_from_the_seeds(art):
    for m in ("flow", "tokens"):
        p = registry.build("policies", _policy(art, m))
        obs = PointReachEnv().reset(TaskInstance(spec=reach_tasks()[1], seed=3))
        a, b = p.sample(obs, [1, 2], 1.0), p.sample(obs, [1, 2], 1.0)
        assert np.allclose(a, b) and not np.allclose(a[0], a[1])


def test_one_policy_runs_chunked_ensembled_and_steered(art, tmp_path):
    flow = _policy(art, "flow")
    te = {"type": "temporal_ensemble", "params": {"policy": _policy(art, "bc"), "m": 0.1}}
    steered = {"type": "steered", "params": {"sampler": flow, "steering": {"n_samples": 4, "selector": "medoid"}}}
    assert _success(art, te, tmp_path, "te") >= 0.8
    assert _success(art, steered, tmp_path, "steered") >= 0.8
