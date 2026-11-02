"""code.llm and code.program through a real EpisodeRunner (SafetyShield, trace) on the toy manipulation double."""

import threading

import numpy as np
import pytest

from hylo.codeagents.agents import CodeLLMAgent, CodeProgramAgent
from hylo.codeagents.fakes import TOY_SOLUTION, ToyManipEnv, toy_benchmark
from hylo.codeagents.sandbox import CodeRejected
from hylo.core.kernel import EpisodeRunner
from hylo.core.trace import TraceStore
from hylo.reasoners.llm.chat import ScriptedChat

TOOLS = "hylo.codeagents.fakes:FakeTools"


def block(code: str) -> str:
    return f"Next step.\n```python\n{code}\n```"


def run(agent, tmp_path, *, max_steps=None, index=0, safety=None):
    inst = toy_benchmark().instances("search")[index]
    if max_steps is not None:
        inst = inst.model_copy(update={"spec": inst.spec.model_copy(update={"max_steps": max_steps})})
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(ToyManipEnv(), agent, trace=store, safety=safety).run(inst)
    return rec, store.read(rec.outcome.episode_id)


def no_worker_threads():
    return not [t for t in threading.enumerate() if t.name.startswith(("hylo-code", "hylo-sandbox"))]


def test_code_llm_solves_the_task_through_the_kernel_and_traces_every_turn(tmp_path):
    model = ScriptedChat([
        block("view = robot.look()\nrobot.show(view, 'top camera')\nprint(robot.status()['instruction'])"),
        block(TOY_SOLUTION + "\nsolve(robot)\nprint(robot.holding('arm'))"),
        block("finish('placed it')"),
    ])
    agent = CodeLLMAgent(model=model, tools=TOOLS, max_turns=6)
    rec, tr = run(agent, tmp_path)
    o = rec.outcome
    assert o.status == "success" and o.termination_reason == "agent_done: placed it"
    turns = [e["payload"] for e in tr.of_type("agent.code.turn")]
    assert [t["turn"] for t in turns] == [1, 2, 3]
    assert turns[0]["output"].strip() == "put the red block in the bin" and "system" in turns[0]["prompt"]
    assert len(turns[0]["images"]) == 1 and turns[0]["images"][0]["note"] == "top camera"
    assert turns[0]["images"][0]["image"].shape == (48, 48, 3)
    assert all(t["tokens_in"] > 0 and t["cost_usd"] == 0.0 and "latency_s" in t for t in turns)
    assert "placed red" in turns[1]["output"] and turns[1]["error"] is None
    chunks = list(tr.of_type("agent.chunk"))
    assert len(chunks) == 4 and chunks[-1]["payload"]["provenance"]["end_episode"] == "placed it"
    assert chunks[0]["payload"]["provenance"]["turn"] == 2                 # the motion is attributed to its turn
    assert sum(1 for _ in tr.of_type("safety.review")) == o.steps          # every step went through the shield
    assert len(model.calls[1][-1]["images"]) == 1                         # the image reached the (vision) model
    system = model.calls[0][0]["content"]
    assert "robot.run" in system and "arm 'arm'" in system and "success" not in model.calls[-1][-1]["content"]
    summary = tr.last("agent.code.summary")["payload"]
    assert summary["turns"] == 3 and summary["finished"] and summary["usage"]["calls"] == 3
    assert no_worker_threads()


def test_the_episode_ending_mid_motion_unblocks_the_program(tmp_path):
    model = ScriptedChat([block(TOY_SOLUTION.replace("    robot.run(grasp)", "    print(robot.run(grasp))")
                                + "\nsolve(robot)\nprint('after', robot.status()['ended'])")], then=block("print('x')"))
    agent = CodeLLMAgent(model=model, tools=TOOLS, max_turns=5)
    rec, tr = run(agent, tmp_path, max_steps=8)
    assert rec.outcome.termination_reason == "max_steps" and rec.outcome.steps == 8
    out = tr.last("agent.code.turn")["payload"]["output"]
    assert "'ended': True" in out and "after True" in out and "'success': None" in out   # never told the score
    assert len(list(tr.of_type("agent.code.turn"))) == 1                   # no turn after the end
    assert no_worker_threads()


def test_a_motion_cut_short_by_the_shield_returns_interrupted(tmp_path):
    code = ("import numpy as np\nq = robot.status()['q']['arm']\n"
            "plan = robot.plan(robot.Goal('reach', 'arm', joints=np.array([0.35, 0.35, 0.3])))\n"
            "plan.actions[3:, 2] = 0.9\nres = robot.run(plan)\nprint(res['interrupted'], res['steps'] < len(plan.actions))")
    agent = CodeLLMAgent(model=ScriptedChat([block(code), block("finish()")]), tools=TOOLS)
    rec, tr = run(agent, tmp_path)
    assert tr.first("agent.code.turn")["payload"]["output"].strip() == "True True"
    assert any(e["payload"]["modified"] for e in tr.of_type("safety.review"))
    assert rec.outcome.termination_reason == "agent_done: finish() called"


def test_budgets_end_the_episode_and_errors_go_back_to_the_model(tmp_path):
    model = ScriptedChat([block("robot.plan(None)"), "I am thinking without code."], then=block("print(1)"))
    agent = CodeLLMAgent(model=model, tools=TOOLS, max_turns=3)
    rec, tr = run(agent, tmp_path)
    assert rec.outcome.termination_reason == "agent_done: budget: 3 turns used" and rec.outcome.status == "failure"
    first, second = [e["payload"] for e in tr.of_type("agent.code.turn")][:2]
    assert first["error"] and "TypeError" in first["error"] and first["error"].startswith("Traceback")
    assert second["code"] is None and "No ```python block" in model.calls[2][-1]["content"]


def test_a_model_that_never_answers_times_out_cleanly(tmp_path):
    gate = threading.Event()

    def slow(messages):
        gate.wait(1.5)
        return block("print('too late')")

    agent = CodeLLMAgent(model=ScriptedChat(slow), tools=TOOLS, act_wait_s=0.3, block_timeout_s=1.0)
    rec, tr = run(agent, tmp_path)
    gate.set()
    assert rec.outcome.termination_reason.startswith("agent_done: agent_timeout")
    assert no_worker_threads()


def test_code_program_runs_a_frozen_solve_without_a_model(tmp_path):
    agent = CodeProgramAgent(source=TOY_SOLUTION, tools=TOOLS)
    for i in range(3):
        rec, tr = run(agent, tmp_path, index=i)
        assert rec.outcome.status == "success" and rec.outcome.termination_reason == "agent_done: solve returned"
        assert "placed" in tr.last("agent.code.program")["payload"]["output"]
    assert agent.spec().components["program"].startswith("sha256:")
    broken = CodeProgramAgent(source="def solve(robot):\n    raise ValueError('no grasp found')", tools=TOOLS)
    rec, tr = run(broken, tmp_path)
    assert rec.outcome.status == "failure" and "ValueError: no grasp found" in rec.outcome.termination_reason
    with pytest.raises(CodeRejected):
        CodeProgramAgent(source="import os\ndef solve(robot):\n    pass")
    with pytest.raises(CodeRejected):
        CodeProgramAgent(source="def main(robot):\n    pass")
    assert no_worker_threads()


def test_status_waits_for_the_first_observation_and_its_instruction():
    # the program thread starts inside act(), possibly before act() stored the observation: status() must report
    # the episode's own instruction, not the task family's template
    from hylo.codeagents.bridge import EpisodeBridge
    from hylo.core.types import Observation

    env = ToyManipEnv()
    inst = toy_benchmark().instances("search")[0]
    bridge = EpisodeBridge(env.spec(), inst)
    seen = {}
    t = threading.Thread(target=lambda: seen.update(bridge.status()))
    t.start()
    obs = Observation(state={"q.arm": np.zeros(7), "hand.arm": np.ones(1)}, instruction="put the red block there")
    threading.Thread(target=bridge.act, args=(obs,), daemon=True).start()
    t.join(5.0)
    assert seen["instruction"] == "put the red block there"
    bridge.cancel()


def test_code_agents_refuse_an_environment_without_a_robot_spec(tmp_path):
    from hylo.testing.point_reach import PointReachEnv, reach_tasks
    from hylo.core.task import TaskInstance

    agent = CodeProgramAgent(source=TOY_SOLUTION, tools=TOOLS)
    agent.spec = lambda: CodeProgramAgent.spec(agent).model_copy(update={"action_semantics": None})
    rec = EpisodeRunner(PointReachEnv(), agent).run(TaskInstance(spec=reach_tasks()[0], seed=0))
    assert rec.outcome.status == "error" and "RobotSpec" in rec.outcome.termination_reason
    assert np.isfinite(rec.outcome.steps)
