from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.artifact import development_runs
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.learning import TrainJob
from worldcurrent.learn.jobs import run_job
from worldcurrent.evolve.candidate import Candidate, CandidateRegistry
from worldcurrent.evolve.protocol import Archive, ProtocolConfig, run_cycle
from worldcurrent.evolve.steering import diagnose_traces

CFG = ExperimentConfig(name="wm", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
                       env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})


def _model(tmp_path):
    for kp in (0.3, 1.0, 2.0):
        run_experiment(CFG.with_overrides({"agent.params.kp": kp}), split="search", run_root=tmp_path / "data", label=f"kp{kp}")
    m = run_job(TrainJob(learner="world_model.state", inputs=[str(tmp_path / "data")], out=str(tmp_path / "wm"),
                         splits=("search",), params={"env": {"type": "testing.point_reach"}, "epochs": 25, "members": 2}))
    return m, {"type": "world_model", "params": {"path": str(tmp_path / "wm")}}


def test_world_model_environment_ranks_controllers_like_the_real_env(tmp_path):
    m, env = _model(tmp_path)
    assert m.kind == "world_model" and m.role == "instrument"
    sim = {kp: run_experiment(CFG.with_overrides({"agent.params.kp": kp, "env": env}), split="search_val",
                              run_root=tmp_path / "sim", label=f"s{kp}").metrics["success_mean"] for kp in (0.1, 2.0)}
    assert sim[2.0] > sim[0.1] + 0.5
    assert development_runs([tmp_path / "sim"]) == []          # model rollouts never count as development data


def test_prescreen_drops_a_predicted_regression_before_any_real_rollout(tmp_path):
    _, env = _model(tmp_path)

    class Two:
        name = "test.two"

        def propose(self, diagnosis, cfg, parent_id, archive):
            return [(Candidate(patch={"agent.params.kp": kp}, rationale="", parent_id=parent_id, generator=self.name), f"kp {kp}")
                    for kp in (0.1, 2.0)]

    labels = []

    def evaluate(c, split, label):
        labels.append(label)
        return run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)

    rep = run_cycle(CFG.with_overrides({"agent.params.kp": 1.0}), CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "a.jsonl"), Two(), evaluate, diagnose_traces,
                    protocol=ProtocolConfig(prescreen={"env": env, "split": "search", "margin": 0.1}))
    bad, good = rep.entries
    assert bad.verdict == "rejected" and bad.evidence["stage0"].startswith("prescreen")
    assert not any(l.startswith(bad.candidate.id + "-search_val") for l in labels)
    assert good.stage_reached >= 2 and good.evidence["prescreen"]["predicted"] > good.evidence["prescreen"]["parent_predicted"]
    assert rep.prescreen["rejected"] == 1
