import json
import sys

from hylo.eval.experiment import run_experiment
from hylo.core.artifact import Provenance, write_manifest
from hylo.core.config import ExperimentConfig
from hylo.evolve.candidate import Candidate, CandidateRegistry, check_protected_scope
from hylo.evolve.learning import LearningProposer
from hylo.evolve.protocol import Archive, run_cycle
from hylo.evolve.steering import diagnose_traces

LEARNER = "hylo.testing.learners:SummaryLearner"
CFG = ExperimentConfig(name="p", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
                       env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})


def _runs(tmp_path):
    root = tmp_path / "cycle" / "runs"
    for name, split in (("parent-search", "search"), ("x-gate", "gate")):
        rd = root / name
        rd.mkdir(parents=True)
        (rd / "manifest.json").write_text(json.dumps({"benchmark": {"split": split}}))
        (rd / "evallog.json").write_text(json.dumps({"run_id": name, "samples": [{"status": "success"}] * 3}))
    return root


def test_learning_proposer_trains_verifies_and_fills_the_patch(tmp_path):
    root = _runs(tmp_path)
    prop = LearningProposer(learner=LEARNER, params={"kp": 2.0}, python=sys.executable, out_root=str(tmp_path / "learned"),
                            patch={"agent.params.kp": "{metrics.kp}", "agent.params.note": "from {artifact}"},
                            variants=[{}, {"agent.params.kd": 0.1}], require={"episodes": 1},
                            hypothesis="gain {metrics.kp} from {metrics.episodes} episodes")
    out = prop.propose({"run_dir": str(root / "parent-search")}, CFG, "v1", [])
    assert len(out) == 2
    cand, hyp = out[0]
    check_protected_scope(cand)
    assert cand.patch["agent.params.kp"] == 2.0                      # a bare placeholder keeps the number type
    assert cand.patch["agent.params.note"].startswith("from /") and out[1][0].patch["agent.params.kd"] == 0.1
    assert hyp.startswith("gain 2.0 from 3 episodes")                # the gate run was not read
    assert cand.evidence["artifact"]["splits"] == ["search"] and cand.evidence["artifact"]["role"] == "behaviour"


def test_unmet_requirement_proposes_nothing(tmp_path):
    prop = LearningProposer(learner=LEARNER, python=sys.executable, out_root=str(tmp_path / "learned"),
                            patch={"agent.params.kp": "{metrics.kp}"}, require={"episodes": 100})
    assert prop.propose({"run_dir": str(_runs(tmp_path) / "parent-search")}, CFG, None, []) == []


def test_gate_stage0_rejects_a_candidate_whose_artifact_saw_heldout_data(tmp_path):
    art = tmp_path / "leaky"
    art.mkdir()
    (art / "w").write_text("0")
    write_manifest(art, kind="value_model", role="behaviour", provenance=Provenance(builder="t"))
    raw = json.loads((art / "artifact.json").read_text())
    raw["provenance"]["sources"] = [{"kind": "run", "id": "h", "split": "heldout"}]
    (art / "artifact.json").write_text(json.dumps(raw))

    class Leaky:
        name = "test.leaky"

        def propose(self, diagnosis, cfg, parent_id, archive):
            return [(Candidate(patch={"agent.params.kp": 2.0, "agent.params.critic": str(art)}, rationale="",
                               parent_id=parent_id, generator=self.name), "uses a leaky critic")]

    evaluated = []

    def evaluate(c, split, label):
        evaluated.append(label)
        return run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)

    rep = run_cycle(CFG, CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "a.jsonl"), Leaky(), evaluate,
                    diagnose_traces)
    [e] = rep.entries
    assert e.verdict == "rejected" and "heldout" in e.evidence["stage0"]
    assert evaluated == ["parent-search", "parent-search_val"]       # never rolled out
