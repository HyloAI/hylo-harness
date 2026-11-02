"""What reaches the model: the prompt stays inside a context budget, the first message previews the scene, a campaign's
research session sees the program it should improve."""

from hylo.codeagents.agents import CodeLLMAgent
from hylo.codeagents.fakes import ToyManipEnv, toy_benchmark
from hylo.codeagents.generator import ResearchProposer
from hylo.codeagents.sandbox import Sandbox
from hylo.codeagents.turns import CodeLoop
from hylo.core.config import ExperimentConfig
from hylo.core.kernel import EpisodeRunner
from hylo.core.trace import TraceStore
from hylo.reasoners.llm.chat import ScriptedChat

TOOLS = "hylo.codeagents.fakes:FakeTools"


def test_old_turns_are_dropped_to_keep_the_prompt_under_the_context_budget():
    loop = CodeLoop(ScriptedChat(then="x"), Sandbox({}), system="S" * 1000, first="F" * 100, emit=lambda r: None,
                    status=lambda: {}, max_history_turns=50, max_context_chars=5000)
    for i in range(20):
        loop.messages += [{"role": "assistant", "content": f"a{i}" + "A" * 300},
                          {"role": "user", "content": f"u{i}" + "U" * 300}]
    msgs = loop._history()
    assert sum(len(m["content"]) for m in msgs) <= 5000 + 200
    assert msgs[0]["content"].startswith("S") and msgs[1]["content"].startswith("F")
    assert "earlier turns omitted" in msgs[2]["content"]                  # the model is told turns were dropped
    tail = msgs[4:]
    assert tail[0]["role"] == "assistant" and tail[-1]["content"].startswith("u19")   # whole, most recent turns
    loop.max_context_chars = None
    assert len(loop._history()) == 2 + 40                                  # no budget: every turn (up to max turns)


def test_the_first_message_previews_the_scene_for_a_vision_model(tmp_path):
    model = ScriptedChat(["Done.\n```python\nfinish('nothing')\n```"])
    agent = CodeLLMAgent(model=model, tools=TOOLS, max_turns=2)
    store = TraceStore(tmp_path / "run")
    EpisodeRunner(ToyManipEnv(), agent, trace=store).run(toy_benchmark().instances("search")[0])
    first = model.calls[0][1]
    assert first["role"] == "user" and "Image attached" in first["content"] and len(first["images"]) == 1


def test_a_campaign_research_session_is_shown_the_parent_program(tmp_path):
    src = "def solve(robot):\n    print('weak parent')\n"
    cfg = ExperimentConfig(name="toy", benchmark={"type": "hylo.codeagents.fakes:toy_benchmark", "params": {}},
                           env={"type": "hylo.codeagents.fakes:ToyManipEnv"}, output_dir=str(tmp_path),
                           agent={"type": "code.program", "params": {"source": src, "tools": TOOLS}})
    assert ResearchProposer._parent_program(cfg) == src
    llm = cfg.with_overrides({"agent.type": "code.llm", "agent.params": {"model": "scripted"}})
    assert ResearchProposer._parent_program(llm) is None


def test_program_source_collects_what_the_blocks_defined():
    from hylo.codeagents.sandbox import Sandbox, validate

    sb = Sandbox({})
    sb.run("import math\nHOVER = 0.01\nview = abs(-3)\ndef helper(x):\n    return x + HOVER")
    assert sb.program_source("solve") is None
    sb.run("def solve(robot):\n    print(helper(1))\nsolve(None)\nfor i in range(2):\n    pass")
    sb.run("def helper(x):\n    return x * 2")                       # redefined: the latest wins
    src = sb.program_source("solve")
    validate(src)
    assert src.startswith("import math") and "HOVER = 0.01" in src and "view = abs" not in src
    assert "solve(None)" not in src and "for i" not in src and "x * 2" in src and "x + HOVER" not in src


def test_a_research_session_starts_from_the_parent_program_and_submits_it(tmp_path):
    from hylo.codeagents.fakes import TOY_SOLUTION
    from hylo.codeagents.research import ResearchConfig, ResearchSession

    cfg = ExperimentConfig(name="toy", benchmark={"type": "hylo.codeagents.fakes:toy_benchmark", "params": {}},
                           env={"type": "hylo.codeagents.fakes:ToyManipEnv"}, output_dir=str(tmp_path),
                           agent={"type": "code.program", "params": {"source": TOY_SOLUTION, "tools": TOOLS}})
    model = ScriptedChat(["Try it.\n```python\nr = submit()\nprint(r['successes'], r['cases'])\n```",
                          "Done.\n```python\nprint(finish())\n```"])
    rc = ResearchConfig(dev=2, tools=TOOLS, parent_program=TOY_SOLUTION)
    s = ResearchSession(cfg, model, tmp_path / "s", config=rc).run()
    assert s["submissions"][0]["successes"] == 2 and s["frozen"] is not None
    assert "already defined in your namespace" in model.calls[0][0]["content"]
    assert all("<" not in c["instruction"] for c in s["dev_cases"])        # the episodes' own instructions


def test_a_block_repeated_verbatim_is_pointed_out():
    reply = "Again.\n```python\nprint('try')\n```"
    model = ScriptedChat([reply, reply])
    loop = CodeLoop(model, Sandbox({}), system="S", first="F", emit=lambda r: None, status=lambda: {})
    loop.step({})
    assert "same as your previous one" not in loop.messages[-1]["content"]
    loop.step({})
    assert loop.messages[-1]["content"].startswith("This block is the same as your previous one")


def test_skills_can_be_imported_from_the_namespace():
    from types import SimpleNamespace

    sb = Sandbox({})
    assert "no 'skills'" in (sb.run("from skills import stack").error or "")
    sb.add(skills=SimpleNamespace(stack=lambda robot: "stacked"))
    res = sb.run("from skills import stack\nimport skills\nprint(stack(None), skills.stack(None))")
    assert res.error is None and res.output.strip() == "stacked stacked"
    assert sb.run("import os").rejected
