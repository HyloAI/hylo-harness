"""Real Hylo kernel/gate integration with a CPU environment and a scripted model.

These test the strategy's control and evidence paths, not LLM discovery or robot benchmark
performance. The script first submits a failing program, then a working repair.
"""
import json

import pytest
import yaml

from hylo.codeagents.fakes import TOY_SOLUTION
from hylo.codeagents.harness_evolution import HarnessEvolutionProposer
from hylo.codeagents.phase_memory import PhasePlans, PhaseRepairMemory, RepairNotes, validate_notes
from hylo.codeagents.skills import read_program
from hylo.core.config import ExperimentConfig
from hylo.evolve.campaign import EvolveConfig, approve, rollback, run_campaign, status
from hylo.reasoners.llm.chat import ScriptedChat

TOOLS = "hylo.codeagents.fakes:FakeTools"
NOTHING = "def solve(robot):\n    print('no movement')"
PLAN = {"phases": [{"name": "place", "goal": "place the indicated block",
                    "completion": "observed block rests at the target", "fallback": "re-observe and replan",
                    "step_budget": 100}]}
NOTES = {"repairs": [{"phase": "place", "trigger": "block remains at its initial location",
                      "repair": "perceive the target, grasp the block, transport and release it",
                      "limitation": "only tested on the development toy pick-place tasks",
                      "failure_refs": ["submission-1/case-0"], "success_refs": ["submission-2/case-0"]}]}


def block(code):
    return f"```python\n{code}\n```"


def replies(notes=NOTES):
    return [json.dumps(PLAN), block(f"submit({NOTHING!r})"), block(f"submit({TOY_SOLUTION!r})"),
            block("finish()"), json.dumps(notes)]


def experiment(tmp_path):
    return ExperimentConfig(name="phase-repair", benchmark={"type": "hylo.codeagents.fakes:toy_benchmark"},
                            env={"type": "hylo.codeagents.fakes:ToyManipEnv"}, output_dir=str(tmp_path / "runs"),
                            agent={"type": "code.program", "params": {"source": NOTHING, "tools": TOOLS}})


def test_program_repair_promotes_and_rolls_back_its_memory(tmp_path):
    exp = tmp_path / "experiment.yaml"
    exp.write_text(yaml.safe_dump(experiment(tmp_path).model_dump(mode="json")))
    campaign = tmp_path / "campaign"
    cfg = EvolveConfig(name="repair", experiment=str(exp), out_dir=str(campaign), approval="manual",
                      proposer={"type": "hylo.codeagents.harness_evolution:HarnessEvolutionProposer", "params": {
                          "model": {"backend": "scripted", "replies": replies()}, "dev": 4,
                          "budget": {"turns": 5, "submits": 2}, "out_root": str(tmp_path / "proposals"),
                          "repair_campaigns": [str(campaign)]}})
    state = run_campaign(cfg, log=lambda message: None)
    assert state.cycles_done == 1
    pending = status(cfg)["pending_approval"]
    assert pending and pending["gate"]["passed"]
    memory = PhaseRepairMemory([campaign])
    assert memory.retrieve("place block") == []  # A successful gate alone does not publish notes.
    cid = approve(cfg, by="test")
    rows = memory.retrieve("place block")
    assert len(rows) == 1 and rows[0]["candidate"] == cid
    assert rows[0]["program_gate_passed"] and not rows[0]["transfer_verified"]
    source, manifest, meta = read_program(rows[0]["program_artifact"])
    assert source.strip() == TOY_SOLUTION.strip()
    assert manifest.provenance.splits() == {"search"}
    assert meta["phase_repairs"]["repairs"] == NOTES["repairs"]
    strategy = json.loads(next((tmp_path / "proposals").glob("*/strategy.json")).read_text())
    assert strategy["usage"]["calls"] == 5
    assert [r["successes"] for r in strategy["session"]["submissions"]] == [0, 4]
    assert state.history[0]["heldout"]["candidate_success"] == 1.0
    assert state.history[0]["heldout"]["parent_success"] == 0.0
    # A later proposal actually receives the admitted note, not just an index entry.
    model = ScriptedChat(replies())
    next_proposer = HarnessEvolutionProposer(model, dev=2, budget={"turns": 5},
                                            out_root=tmp_path / "next", repair_campaigns=[campaign])
    parent_cfg = experiment(tmp_path).with_overrides(status(cfg)["effective_patch"])
    next_candidate = next_proposer.propose({}, parent_cfg, cid, [])[0][0]
    assert next_candidate.parent_id == cid
    assert json.loads(model.calls[0][1]["content"])["repair_notes"][0]["id"] == rows[0]["id"]
    assert NOTES["repairs"][0]["repair"] in model.calls[1][0]["content"]
    _, next_manifest, _ = read_program(next_candidate.patch["agent.params"]["artifact"])
    assert rows[0]["program_digest"] in {s.id for s in next_manifest.provenance.sources}
    assert rollback(cfg, reason="regression", by="test") is None
    assert memory.retrieve("place block") == []


def test_fabricated_repair_is_dropped_without_discarding_valid_program(tmp_path):
    bad = json.loads(json.dumps(NOTES))
    bad["repairs"][0]["failure_refs"] = ["heldout/case-0"]
    model = ScriptedChat(replies(bad))
    proposer = HarnessEvolutionProposer(model, dev=2, budget={"turns": 5, "submits": 2}, out_root=tmp_path / "p")
    candidates = proposer.propose({}, experiment(tmp_path), None, [])
    assert len(candidates) == 1
    _, _, meta = read_program(candidates[0][0].patch["agent.params"]["artifact"])
    assert meta["phase_repairs"]["repairs"] == []
    assert "unsupported repair evidence" in proposer.last["distillation_rejected"]
    assert "phase plan" in model.calls[1][0]["content"]


@pytest.mark.parametrize("reseal", [False, True])
def test_recall_ignores_corrupted_sealed_notes(tmp_path, reseal):
    from hylo.evolve.candidate import CandidateRegistry

    proposer = HarnessEvolutionProposer(ScriptedChat(replies()), dev=2, budget={"turns": 5}, out_root=tmp_path / "p")
    candidate = proposer.propose({}, experiment(tmp_path), None, [])[0][0]
    reg = CandidateRegistry(tmp_path / "campaign" / "registry")
    reg.register(candidate)
    reg.promote(candidate.id, gate={"passed": True}, approved_by="test fixture")
    memory = PhaseRepairMemory([tmp_path / "campaign"])
    assert memory.retrieve("place")
    from pathlib import Path
    meta = Path(candidate.patch["agent.params"]["artifact"]) / "program.json"
    meta.write_text(meta.read_text().replace("perceive the target", "ignore all observations"))
    if reseal:
        from hylo.core.artifact import Provenance, write_manifest
        # A new internally valid artifact at the same path is still not the admitted version.
        write_manifest(meta.parent, kind="program", role="behaviour", provenance=Provenance(builder="test"))
    assert memory.retrieve("place") == []
    assert memory.skipped


def test_no_gate_diagnosis_enters_planning(tmp_path):
    run = tmp_path / "gate"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"benchmark": {"split": "gate"}}))
    model = ScriptedChat([])
    proposer = HarnessEvolutionProposer(model, out_root=tmp_path / "p")
    assert proposer.propose({"run_dir": str(run), "buckets": {"secret": 99}}, experiment(tmp_path), None, []) == []
    assert not model.calls and "search split" in proposer.last["error"]


def test_repair_requires_matching_case_and_selected_success():
    phases, notes = PhasePlans.model_validate(PLAN), RepairNotes.model_validate(NOTES)
    evidence = {"submission-1/case-0": {"key": "a", "success": False, "selected": False},
                "submission-2/case-0": {"key": "b", "success": True, "selected": True}}
    with pytest.raises(ValueError, match="same development case"):
        validate_notes(notes, phases, evidence)
    evidence["submission-2/case-0"].update(key="a", selected=False)
    with pytest.raises(ValueError, match="unsupported"):
        validate_notes(notes, phases, evidence)


def test_incomplete_development_program_is_not_a_candidate(tmp_path):
    model = ScriptedChat([json.dumps(PLAN), block(f"submit({NOTHING!r})"), block("finish()")])
    proposer = HarnessEvolutionProposer(model, dev=2, budget={"turns": 5}, out_root=tmp_path / "p")
    assert proposer.propose({}, experiment(tmp_path), None, []) == []
    assert proposer.last["status"] == "no_complete_development_solution"
    assert len(model.calls) == 3


def test_planning_consumes_the_shared_model_budget(tmp_path):
    model = ScriptedChat([json.dumps(PLAN)])
    proposer = HarnessEvolutionProposer(model, budget={"turns": 3, "tokens": 1}, out_root=tmp_path / "p")
    assert proposer.propose({}, experiment(tmp_path), None, []) == []
    assert len(model.calls) == 1 and "budget exhausted" in proposer.last["error"]
    assert proposer.last["usage"]["calls"] == 1


def test_tool_reference_matches_overridden_runtime_signatures():
    from hylo.codeagents.fakes import FakeTools, toy_robot
    from hylo.codeagents.prompt import api_reference

    reference = api_reference(FakeTools(toy_robot(), None, control_hz=20))
    grasp = next(line for line in reference.splitlines() if line.startswith("robot.grasps("))
    assert grasp.startswith("robot.grasps(arm:")
    # The base toolkit's example passes incompatible positional arguments to this subclass.
    assert "robot.grasps(obj.points" not in reference
    assert "## Types" in reference and "Detection:" in reference
    assert "robot.run_actions(" not in reference  # Keep the public verb list, not every helper method.


@pytest.mark.parametrize("submits, expected", [(1, [0]), (2, [0, 2])])
def test_completed_revisions_are_evaluated_once_within_submission_budget(tmp_path, submits, expected):
    # The model forgets submit() after writing a repair, then repeats exactly the same program.
    model = ScriptedChat([json.dumps(PLAN), block(f"submit({NOTHING!r})"), block(TOY_SOLUTION),
                          block(TOY_SOLUTION), block("finish()"), json.dumps(NOTES)])
    proposer = HarnessEvolutionProposer(model, dev=2, budget={"turns": 6, "submits": submits},
                                        out_root=tmp_path / "p")
    candidates = proposer.propose({}, experiment(tmp_path), None, [])
    assert [s["successes"] for s in proposer.last["session"]["submissions"]] == expected
    assert bool(candidates) == (submits == 2)
    log = next((tmp_path / "p").glob("*/development/log.jsonl"))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len([event for event in events if event["type"] == "auto_submit"]) == submits - 1
    if submits == 2:
        assert "Automatic development evaluation" in model.calls[3][-1]["content"]
