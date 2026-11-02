"""Research sessions, program candidates through an evolve campaign, program skills, `hylo research`."""

import json

import pytest
import yaml

from hylo.codeagents.agents import CodeLLMAgent
from hylo.codeagents.fakes import TOY_SOLUTION, ToyManipEnv, toy_benchmark
from hylo.codeagents.research import ResearchSession
from hylo.codeagents.skills import demote, program_skill, promote_from_campaign, read_program
from hylo.core.artifact import verify
from hylo.core.config import ExperimentConfig
from hylo.core.errors import ProtectedScopeViolation
from hylo.core.improvement import Candidate
from hylo.core.kernel import EpisodeRunner
from hylo.core.trace import TraceStore
from hylo.evolve.candidate import check_protected_scope
from hylo.memory.skills import SkillLibrary
from hylo.reasoners.llm.chat import ScriptedChat

TOOLS = "hylo.codeagents.fakes:FakeTools"
NOTHING = "def solve(robot):\n    print('nothing to do')"


def block(code: str) -> str:
    return f"Plan.\n```python\n{code}\n```"


SCRIPT = [
    block("print(cases)\nprint(robot.status()['instruction'])"),
    block("out = reset(1)\nprint(out['case'], out['reason'])\nprint(robot.status()['instruction'])"),
    block(f"r = submit({NOTHING!r})\nprint(r['successes'], r['results'][0]['output'])"),
    block("submit('import os')"),
    block(f"r = submit({TOY_SOLUTION!r})\nprint(r['successes'], r['cases'])"),
    block("print(finish())"),
]


def experiment(tmp_path, **agent_params):
    return ExperimentConfig(name="toy", benchmark={"type": "hylo.codeagents.fakes:toy_benchmark", "params": {}},
                            env={"type": "hylo.codeagents.fakes:ToyManipEnv"}, output_dir=str(tmp_path / "runs"),
                            agent={"type": "code.program", "params": {"source": NOTHING, "tools": TOOLS, **agent_params}})


def test_a_research_session_submits_freezes_and_yields_a_code_candidate(tmp_path):
    model = ScriptedChat(SCRIPT)
    s = ResearchSession(experiment(tmp_path), model, tmp_path / "session", config={"dev": 4, "tools": TOOLS})
    summary = s.run()
    assert summary["stopped"] == "finish() called" and summary["turns"] == 6 and summary["resets"] == 1
    assert [(x["successes"], x["cases"]) for x in summary["submissions"]] == [(0, 4), (4, 4)]
    assert {c["key"].split("#")[1] for c in summary["dev_cases"]} == {"0", "1"}        # search split seeds only
    log = [json.loads(l) for l in (tmp_path / "session" / "log.jsonl").read_text().splitlines()]
    turns = [r for r in log if r["type"] == "turn"]
    assert "rejected" in turns[3]["error"] and turns[1]["output"].startswith("0 agent_done: reset")
    assert "put the blue block" in turns[1]["output"]
    assert turns[2]["output"].startswith("0 nothing to do")
    for r in turns:
        assert "gate" not in json.dumps(r.get("reply", "")) and "heldout" not in r.get("output", "")
    frozen = summary["frozen"]
    m = verify(frozen["artifact"])
    assert m.kind == "program" and m.role == "behaviour" and m.provenance.splits() == {"search"}
    source, _, meta = read_program(frozen["artifact"])
    assert source.strip() == TOY_SOLUTION.strip() and meta["submission"] == 2 and meta["family"] == "pick_place+toy"
    cand = s.candidate(parent_id="v1")
    assert cand.kind == "code" and cand.patch["agent.type"] == "code.program"
    assert cand.patch["agent.params"]["artifact"] == frozen["artifact"] and cand.parent_id == "v1"
    check_protected_scope(cand)                                         # stage 0 accepts it
    assert (tmp_path / "session" / "submissions" / "sub-02" / "results.json").exists()
    assert json.loads((tmp_path / "session" / "summary.json").read_text())["frozen"]["digest"] == m.digest


def test_code_candidates_must_carry_a_sealed_program(tmp_path):
    inline = Candidate(patch={"agent.type": "code.program", "agent.params": {"source": TOY_SOLUTION}}, rationale="",
                       parent_id=None, generator="t", kind="code")
    with pytest.raises(ProtectedScopeViolation, match="inline source"):
        check_protected_scope(inline)
    outside = Candidate(patch={"env.params.x": 1}, rationale="", parent_id=None, generator="t", kind="code")
    with pytest.raises(ProtectedScopeViolation):
        check_protected_scope(outside)
    nothing = Candidate(patch={"agent.type": "code.program"}, rationale="", parent_id=None, generator="t", kind="code")
    with pytest.raises(ProtectedScopeViolation, match="no sealed program"):
        check_protected_scope(nothing)


def test_budgets_stop_a_research_session(tmp_path):
    model = ScriptedChat([], then=block("reset(0)"))
    s = ResearchSession(experiment(tmp_path), model, tmp_path / "s",
                        config={"dev": 2, "tools": TOOLS, "budget": {"turns": 5, "resets": 2, "submits": 1}})
    summary = s.run()
    assert summary["stopped"] == "budget: 5 turns used" and summary["resets"] == 2 and summary["frozen"] is None
    log = [json.loads(l) for l in (tmp_path / "s" / "log.jsonl").read_text().splitlines()]
    assert "reset budget used up" in [r for r in log if r["type"] == "turn"][-1]["error"]
    assert s.candidate() is None


def test_a_research_candidate_goes_through_the_evolve_gate_and_becomes_a_program_skill(tmp_path):
    from hylo.evolve.campaign import EvolveConfig, run_campaign, status

    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump(experiment(tmp_path).model_dump(mode="json")))
    cfg = EvolveConfig(name="code", experiment=str(exp), out_dir=str(tmp_path / "camp"), cycles=1,
                       proposer={"type": "research.llm", "params": {
                           "model": {"backend": "scripted", "replies": SCRIPT}, "dev": 4,
                           "out_root": str(tmp_path / "research")}})
    state = run_campaign(cfg, log=lambda m: None)
    st = status(cfg)
    entry = state.history[0]
    assert entry["promoted"] is not None and st["active"] == entry["promoted"]
    assert entry["heldout"]["candidate_success"] == 1.0 and entry["heldout"]["parent_success"] == 0.0
    assert st["effective_patch"]["agent.type"] == "code.program"
    report = json.loads((tmp_path / "camp" / "cycle-01" / "cycle_report.json").read_text())
    stages = report["entries"][0]["stages"]
    assert report["entries"][0]["candidate"]["kind"] == "code" and stages[-1]["stage"] == 3 and stages[-1]["wins"] > 5
    assert "artifacts" in report["entries"][0]["evidence"]                # stage 0 verified the sealed program

    lib = SkillLibrary(str(tmp_path / "skills"))
    added = promote_from_campaign(tmp_path / "camp", lib)
    skill = lib.get(added[0])
    assert skill.kind == "program" and skill.status == "verified" and skill.evidence["gate"]["passed"]
    assert skill.body["digest"] == verify(skill.body["artifact"]).digest
    assert promote_from_campaign(tmp_path / "camp", lib) == added           # idempotent

    # round trip: code.llm retrieves the skill by instruction similarity and calls it
    model = ScriptedChat([block(f"print(skills)\nskills.{skill.name}(robot)"), block("finish()")])
    agent = CodeLLMAgent(model=model, tools=TOOLS, skills=str(tmp_path / "skills"))
    store = TraceStore(tmp_path / "skill-run")
    inst = toy_benchmark().instances("heldout")[0]
    rec = EpisodeRunner(ToyManipEnv(), agent, trace=store).run(inst)
    tr = store.read(rec.outcome.episode_id)
    assert rec.outcome.status == "success"
    assert f"skills.{skill.name}(robot" in model.calls[0][0]["content"]
    assert tr.first("agent.code.skills")["payload"]["retrieved"][0]["skill"] == added[0]
    assert "placed" in tr.first("agent.code.turn")["payload"]["output"]
    assert "skills" in agent.spec().components

    demote(lib, added[0], {"revalidation": "failed on a new robot"})
    assert lib.get(added[0]).status == "deprecated" and lib.similar(inst.spec.instruction, kinds=("program",)) == []


def test_a_tampered_program_skill_is_not_loaded(tmp_path):
    from hylo.codeagents.skills import add_program_skill, load_program_skills
    from hylo.codeagents.sandbox import Sandbox

    s = ResearchSession(experiment(tmp_path), ScriptedChat(SCRIPT), tmp_path / "s", config={"dev": 2, "tools": TOOLS})
    art = s.run()["frozen"]["artifact"]
    lib = SkillLibrary(str(tmp_path / "lib"))
    sid = add_program_skill(lib, program_skill(art, name="put block"), {"passed": True})
    sb = Sandbox()
    ns, text, used = load_program_skills(lib, "put the red block in the bin", sb.define)
    assert hasattr(ns, "put_block") and "skills.put_block(robot" in text and used[0]["skill"] == sid
    (tmp_path / "s" / "frozen").joinpath(art.rsplit("/", 1)[-1], "solve.py").write_text(NOTHING)
    ns, text, used = load_program_skills(lib, "put the red block in the bin", sb.define)
    assert not vars(ns) and "digest" in used[0]["skipped"]


def test_hylo_research_cli_writes_a_run_dir_and_evaluates_the_frozen_program(tmp_path, capsys):
    from hylo.codeagents.cli import research

    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump(experiment(tmp_path).model_dump(mode="json")))
    model = json.dumps({"backend": "scripted", "replies": SCRIPT})
    code = research([str(exp), "--dev", "4", "--budget", "turns=10,submits=3", "--model", model, "--out",
                     str(tmp_path / "out"), "--evaluate", "search_val", "gate", "--with-parent"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["submissions"] == [[1, "0/4"], [2, "4/4"]]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    ev = summary["evaluation"]
    assert ev["search_val"]["success_mean"] == 1.0 and ev["gate"]["parent_success_mean"] == 0.0
    assert ev["gate"]["paired"]["candidate_wins"] == ev["gate"]["n"]
    assert summary["budget"]["submits"] == 3 and summary["usage"]["calls"] == 6
    assert (tmp_path / "out" / "log.jsonl").exists() and (tmp_path / "out" / "live" / "episodes").is_dir()


def test_the_research_command_is_registered():
    from hylo.core.cli import commands

    assert commands()["research"].value == "hylo.codeagents.cli:research"
