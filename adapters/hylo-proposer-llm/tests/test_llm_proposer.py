import json

from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.evolve.candidate import Candidate
from hylo.evolve.protocol import ArchiveEntry, StageResult
from hylo.evolve.steering import STEERING_PATH, diagnose_traces
from hylo.reasoners.llm.client import LLMClient
from hylo.proposers.llm.proposer import LLMSteeringProposer


def test_proposals_are_validated_and_the_prompt_hides_selection_numbers(tmp_path, monkeypatch):
    cfg = ExperimentConfig(name="m", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "validation": 2}},
                           env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})
    diag = diagnose_traces(run_experiment(cfg, split="validation", run_root=tmp_path))
    archived = ArchiveEntry(candidate=Candidate(patch={STEERING_PATH: {"n_samples": 4}}, rationale="", parent_id=None, generator="x"),
                            hypothesis="voting helps", stage_reached=2, verdict="archived",
                            stages=[StageResult(stage=2, success=0.12345, parent_success=0.54321)])
    reply = {"candidates": [
        {"hypothesis": "replanning sooner reacts to drift", "steering": {"execution_steps": 20}},
        {"hypothesis": "huge vote", "steering": {"n_samples": 50}},                       # out of bounds
        {"hypothesis": "", "steering": {"execution_steps": 10}},                          # no hypothesis
        {"hypothesis": "same as now", "steering": {}},                                    # no change
    ]}
    sent = []

    def fake_post(self, path, body, headers):
        sent.append(body)
        return {"message": {"content": json.dumps(reply)}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    prop = LLMSteeringProposer()
    out = prop.propose(diag, cfg, None, [archived])
    assert [h for _, h in out] == ["replanning sooner reacts to drift"]
    assert out[0][0].patch[STEERING_PATH]["execution_steps"] == 20
    assert len(prop.rejected) == 3
    prompt = sent[0]["messages"][-1]["content"]
    assert "voting helps" in prompt and "archived" in prompt
    assert "0.12345" not in prompt and "0.54321" not in prompt            # search_val numbers never reach the proposer
