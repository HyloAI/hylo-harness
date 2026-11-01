import pytest

from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.errors import ProtectedScopeViolation
from worldcurrent.evolve.candidate import Candidate, CandidateRegistry, check_protected_scope
from worldcurrent.evolve.loop import GainProposer, improvement_cycle
from worldcurrent.eval.experiment import run_experiment


@pytest.mark.parametrize("path", ["safety.max_delta", "benchmark.split", "env.params.dt", "scorers.success"])
def test_candidates_cannot_touch_protected_scope(path):
    with pytest.raises(ProtectedScopeViolation):
        check_protected_scope(Candidate(patch={path: 1}, rationale="", parent_id=None, generator="t"))


def test_registry_promote_rollback_and_lineage(tmp_path):
    reg = CandidateRegistry(tmp_path)
    a = Candidate(patch={"agent.params.kp": 1.0}, rationale="", parent_id=None, generator="t")
    reg.register(a)
    reg.promote(a.id, gate={"passed": True}, approved_by="test")
    b = Candidate(patch={"agent.params.kp": 2.0}, rationale="", parent_id=a.id, generator="t")
    reg.register(b)
    with pytest.raises(ValueError):
        reg.promote(b.id, gate={"passed": False}, approved_by="test")
    reg.promote(b.id, gate={"passed": True}, approved_by="test")
    assert reg.lineage() == [b.id, a.id] and reg.effective_patch() == {"agent.params.kp": 2.0}
    stale = Candidate(patch={"agent.params.kp": 9.0}, rationale="", parent_id=a.id, generator="t")
    reg.register(stale)
    with pytest.raises(ValueError, match="stale"):
        reg.promote(stale.id, gate={"passed": True}, approved_by="test")
    assert reg.rollback(reason="regression in the field", by="test") == a.id
    assert [e["op"] for e in reg.ledger()].count("rollback") == 1


def test_improvement_cycle_promotes_a_validated_candidate(tmp_path):
    cfg = ExperimentConfig(name="rsi", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "train": 4,
                                                                                         "validation": 10, "heldout": 6}},
                           env={"type": "testing.point_reach"},
                           agent={"type": "testing.p_controller", "params": {"kp": 0.5}})
    reg = CandidateRegistry(tmp_path / "registry")

    def evaluate(c, split, label):
        return run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)

    report = improvement_cycle(cfg, reg, GainProposer(factors=(2.0, 4.0, 8.0), adaptive=True), evaluate)
    assert report.diagnosis.buckets.get("timeout_far", 0) > 0
    assert report.promoted is not None and reg.active == report.promoted
    assert report.heldout["used_for_selection"] is False
    assert reg.effective_patch()["agent.params.kp"] > 0.5
