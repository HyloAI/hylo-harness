"""tools.llm (RFC-0007): native tool calls through the typed robot surface, the kernel and the SafetyShield, on
hylo-manip's kinematic test kit (real perception, kinematics and planning; ray-cast RGB-D and segmentation)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.codeagents.robot_surface import calc  # noqa: E402
from hylo.codeagents.tool_agent import Inbox, ToolLLMAgent, post  # noqa: E402
from hylo.codeagents.tool_context import ToolContext  # noqa: E402
from hylo.core.kernel import EpisodeRunner  # noqa: E402
from hylo.core.task import TaskInstance, TaskSpec  # noqa: E402
from hylo.core.tooling import ToolError  # noqa: E402
from hylo.core.trace import TraceStore  # noqa: E402
from hylo.core.types import ActionSemantics, BoxSpace, EnvSpec, ObservationSpec, StepResult  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402

KIT = Path(__file__).resolve().parents[3] / "packages" / "hylo-manip" / "tests" / "manip_testkit.py"
if not KIT.exists():
    pytest.skip("hylo-manip test kit not found", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("hylo_manip_testkit_for_toolagent", KIT)
tk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tk
_spec.loader.exec_module(tk)

RED, BLUE = (0.9, 0.1, 0.1, 1), (0.1, 0.2, 0.9, 1)
TARGET = (0.55, 0.2)


class KitEnv:
    def __init__(self, directory: Path) -> None:
        self.robot = tk.arm_spec(tk.write_arm(directory), cameras=("front", "side", "back"))
        self.dir = directory

    def spec(self) -> EnvSpec:
        arm = self.robot.arms[0]
        cams = {c: (96, 128, 3) for c in ("front", "side", "back")}
        return EnvSpec(name="testing.manip_kit", version="0.1",
                       observation=ObservationSpec(state={"q.arm": (7,), "hand.arm": (1,)}, cameras=cams),
                       action=BoxSpace(shape=(8,), low=(*arm.lower, 0.0), high=(*arm.upper, 1.0),
                                       semantics=ActionSemantics(control_mode="joint_pos", frame="base")),
                       control_hz=20.0, robot=self.robot)

    def reset(self, task: TaskInstance):
        objects = {"red_block": ((0.45, 0.07, 0.025), (0.025, 0.02, 0.025), RED, 0.3),
                   "blue_block": ((0.42, -0.12, 0.04), (0.03, 0.03, 0.04), BLUE, 0.0)}
        self.ex = tk.StubExecutor(self.dir, objects, self.robot, hw=(96, 128))
        self.instruction = task.spec.instruction
        return self.ex.observation().model_copy(update={"instruction": self.instruction})

    def step(self, action):
        self.ex.run(np.asarray(action.data, dtype=float)[None])
        T = self.ex.pose("red_block")
        ok = self.ex.attached is None and bool(np.allclose(T[:2, 3], TARGET, atol=0.02))
        obs = self.ex.observation().model_copy(update={"instruction": self.instruction})
        return StepResult(observation=obs, reward=float(ok), terminated=False, truncated=False, success=ok)

    def close(self) -> None:
        pass


def task(max_steps=600):
    return TaskInstance(spec=TaskSpec(id="testing/kit_place", version="1", max_steps=max_steps,
                                      instruction="put the red block at x 0.55, y 0.2"), seed=0)


def results(msgs, name):
    return [json.loads(m["content"]) for m in msgs if m.get("role") == "tool" and m.get("name") == name]


def test_tools_llm_picks_and_places_through_the_kernel(tmp_path):
    inbox = tmp_path / "inbox.jsonl"
    post(inbox, "an old message from an earlier run")          # present before the run: never shown

    def script(msgs):
        n = sum(1 for m in msgs if m.get("role") == "assistant")
        if n == 0:
            post(inbox, "the red block is the small one")
            return {"text": "Find the block.", "tool_calls": [{"name": "locate", "arguments": {"text": "red block"}},
                                                              {"name": "robot_state", "arguments": {}}]}
        if n == 1:
            assert any("OPERATOR MESSAGE" in str(m.get("content")) and "small one" in m["content"] for m in msgs)
            assert not any("earlier run" in str(m.get("content")) for m in msgs)
            t = results(msgs, "locate")[-1]["candidates"][0]["target"]
            return {"tool_calls": [{"name": "grasp_options", "arguments": {"target": t}},
                                   {"name": "calc", "arguments": {"code": "a = np.array([1, 2, 3])\nnp.linalg.norm(a)"}}]}
        if n == 2:
            g = results(msgs, "grasp_options")[-1]["options"][0]["option"]
            return {"tool_calls": [{"name": "grasp", "arguments": {"option": g}}, {"name": "look", "arguments": {}}]}
        if n == 3:
            assert results(msgs, "grasp")[-1]["error"]["code"] == "act_not_alone"
            assert results(msgs, "look")[-1]["ok"]                     # the read call of that reply ran
            g = results(msgs, "grasp_options")[-1]["options"][0]["option"]
            return {"tool_calls": [{"name": "grasp", "arguments": {"option": g}}]}
        if n == 4:
            r = results(msgs, "grasp")[-1]
            assert r["status"] == "arrived" and r["holding_after_lift"] is True, r
            assert any(str(m.get("content", "")).startswith("OBSERVATION after your action") and m.get("images")
                       for m in msgs if m["role"] == "user")
            return {"tool_calls": [{"name": "move_tcp", "arguments": {"position": [3.0, 0.0, 0.5]}}]}
        if n == 5:
            r = results(msgs, "move_tcp")[-1]
            assert r["status"] == "refused" and r["counted_as_decision"] is False and r["steps"] == 0
            g = results(msgs, "grasp_options")[-1]["options"][0]["option"]
            return {"tool_calls": [{"name": "grasp_options", "arguments": {"target": "t1"}}]}
        if n == 6:
            assert results(msgs, "grasp_options")[-1]["error"]["code"] == "stale"
            return {"tool_calls": [{"name": "place", "arguments": {"at": list(TARGET), "yaw_deg": 0, "hover_m": 0.005}}]}
        return {"tool_calls": [{"name": "finish", "arguments": {"outcome": "success", "reason": "block released at "
                                                                                                "the target"}}]}

    model = ScriptedChat(script)
    agent = ToolLLMAgent(model=model, max_decisions=10, inbox=str(inbox))
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(KitEnv(tmp_path), agent, trace=store, image_every=0).run(task())
    tr = store.read(rec.outcome.episode_id)
    assert rec.outcome.status == "success", [e["payload"]["result"][:200] for e in tr.of_type("agent.tool.result")]
    assert rec.outcome.termination_reason.startswith("agent_done: finish(success)")
    s = tr.last("agent.tool.summary")["payload"]
    assert s["decisions"] == 2 and s["refused"] == 1 and s["human_assisted"] and s["operator_messages"] == 1
    assert s["outcome"]["outcome"] == "success"
    start = tr.last("agent.tool.start")["payload"]
    assert {"locate", "grasp", "place", "finish", "read_artifact"} <= set(start["tools"]) and len(start["surface"]) == 16
    assert {"protocol", "evidence", "boundary", "embodiment", "bundle"} <= set(start["prompt"])
    props, receipts = list(tr.of_type("agent.tool.proposal")), list(tr.of_type("agent.tool.receipt"))
    assert len(props) == len(receipts) == 3
    assert sum(1 for _ in tr.of_type("safety.review")) == rec.outcome.steps > 50
    names = [t.name for t in model.tools[0]]
    assert "locate" in names and all(t.parameters["additionalProperties"] is False for t in model.tools[0])
    calc_res = [json.loads(e["payload"]["result"]) for e in tr.of_type("agent.tool.result") if e["payload"]["name"] == "calc"]
    assert calc_res[0]["value"] == pytest.approx(14 ** 0.5, abs=1e-5)
    # the conversation as a training record: steered, so left out unless asked for
    from hylo.codeagents.chats import conversation, main as chats_main

    conv = conversation(tmp_path / "run" / "episodes" / f"{rec.outcome.episode_id}.jsonl")
    roles = [m["role"] for m in conv["messages"]]
    assert roles[:3] == ["system", "user", "assistant"] and "tool" in roles and conv["meta"]["human_assisted"]
    assert conv["meta"]["success"] is True and any("OPERATOR MESSAGE" in m.get("content", "") for m in conv["messages"])
    out = tmp_path / "chats.jsonl"
    chats_main([str(tmp_path / "run"), "--out", str(out)])
    assert out.read_text() == ""
    chats_main([str(tmp_path / "run"), "--out", str(out), "--include-assisted"])
    assert len(out.read_text().splitlines()) == 1


def test_idle_replies_end_the_episode(tmp_path):
    agent = ToolLLMAgent(model=ScriptedChat(then="I am thinking."), max_idle_replies=2)
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(KitEnv(tmp_path), agent, trace=store, image_every=0).run(task())
    assert rec.outcome.status == "failure"
    assert "without a tool call" in agent.last["stopped"] and agent.last["calls"] == 2


def test_calc_allows_math_and_refuses_the_rest():
    assert calc("x = 2\ny = x * 3", outputs=["y"]) == {"y": 6}
    q = calc("quat_from_axis_angle([0, 0, 1], 90)")["value"]
    assert q == pytest.approx([0, 0, 2 ** -0.5, 2 ** -0.5], abs=1e-6)
    for bad in ("import os", "for i in range(3): pass", "().__class__", "open('x')", "lambda: 1", "2 ** 1000"):
        with pytest.raises(ToolError):
            calc(bad)


def test_context_stores_old_results_and_images_and_keeps_pairs():
    ctx = ToolContext("sys", "task", [np.zeros((4, 4, 3), np.uint8)], tail_decisions=1, image_decisions=1,
                      inline_chars=50, compact_tokens=10**9)
    big = "x" * 500
    for i in range(3):
        ctx.add({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": "look", "arguments": {}}]})
        ctx.add({"role": "tool", "tool_call_id": f"c{i}", "name": "look", "content": big,
                 "images": [np.full((4, 4, 3), i, np.uint8)]})
        ctx.acted()
    msgs = ctx.project()
    tools = [m for m in msgs if m["role"] == "tool"]
    assert all("stored" in m["content"] for m in tools)        # all older than the tail
    assert not any(m.get("images") for m in msgs)
    ref = json.loads(tools[0]["content"])["stored"]
    assert ctx.read_artifact({"ref": ref})["text"] == big
    img_ref = [r for r, a in ctx.artifacts.items() if a["kind"] == "image"][0]
    assert ctx.read_artifact({"ref": img_ref}).images[0].shape == (4, 4, 3)


def test_context_compacts_under_pressure_without_splitting_calls():
    seen = []

    def summarize(msgs):
        seen.append(msgs)
        return '{"progress": ["grasped"], "confirmed": [], "ruled_out": [], "attempts": [], "constraints": [], ' \
               '"avoid": [], "next_checks": ["locate again"]}'

    ctx = ToolContext("sys", "task", tail_decisions=1, inline_chars=10**6, compact_tokens=300, summarize=summarize)
    for i in range(6):
        ctx.add({"role": "assistant", "content": "y" * 200, "tool_calls": [{"id": f"c{i}", "name": "wait",
                                                                             "arguments": {"steps": 1}}]})
        ctx.add({"role": "tool", "tool_call_id": f"c{i}", "name": "wait", "content": "z" * 200})
        ctx.acted()
    msgs = ctx.project()
    assert seen and ctx.stats["summaries"] >= 1
    assert "SUMMARY OF YOUR EARLIER WORK" in msgs[2]["content"] and "locate again" in msgs[2]["content"]
    ids = {c["id"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    assert all(m["tool_call_id"] in ids for m in msgs if m["role"] == "tool")
    assert msgs[3]["role"] == "assistant"


def test_inbox_reads_only_new_complete_lines(tmp_path):
    p = tmp_path / "in.jsonl"
    post(p, "before")
    box = Inbox(p)
    assert box.poll() == []
    post(p, "one")
    with open(p, "a") as f:
        f.write('{"message": "half')                      # not finished yet
    assert box.poll() == ["one"]
    with open(p, "a") as f:
        f.write(' done"}\n')
    assert box.poll() == ["half done"]
