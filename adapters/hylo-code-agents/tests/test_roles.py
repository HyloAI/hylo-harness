"""Role campaigns (RFC-0008 part 9) on the stage world, with a scripted worker: a dev task is solved, its calls
become a recipe skill keyed by the failure signature, a model-free replay verifies it, the held-out task runs last."""

import json

import pytest

from hylo.codeagents.roles import Ledger, RoleCampaign, goal_atoms, recipe_from
from hylo.core.registry import registry
from hylo.core.tooling import Help, Surface, ToolDef, obj
from hylo.reasoners.llm.chat import ScriptedChat


def world_available(tmp_path):
    try:
        s = registry.build("surfaces", {"type": "mujoco.world", "params": {"root": str(tmp_path / "probe"),
                                                                            "stage": "p"}})
        from hylo.envs.mujoco.packs import pack

        pack("panda").model_path()
        return s is not None
    except Exception:  # noqa: BLE001
        return False


def stage(cube_x):
    return {"name": "x", "entities": [
        {"id": "table", "shape": {"type": "box", "size": [0.4, 0.4, 0.2]}, "body": {"dynamic": False},
         "pose": {"pos": [0.5, 0, 0.2]}},
        {"id": "arm", "robot": {"pack": "panda"}, "pose": {"pos": [0, 0, 0.4]}},
        {"id": "cube", "shape": {"type": "box", "size": [0.02, 0.02, 0.02]}, "body": {"mass": 0.05},
         "pose": {"pos": [cube_x, 0, 0.42]}}]}


GOAL = {"op": "lifted", "entity": "cube", "min_z": 0.5}


def worker_script(cube_x):
    plan = [("tcp", {"position": [cube_x, 0, 0.53], "quat_wxyz": [0, 1, 0, 0]}),
            ("tcp", {"position": [cube_x, 0, 0.43], "quat_wxyz": [0, 1, 0, 0]}),
            ("gripper", {"command": "close"}),
            ("tcp", {"position": [cube_x, 0, 0.6], "quat_wxyz": [0, 1, 0, 0]})]

    def fn(msgs):
        n = sum(1 for m in msgs if m["role"] == "assistant")
        if n == 0:
            return {"tool_calls": [{"name": "world_state", "arguments": {}}]}
        gen = next(json.loads(m["content"])["generation"] for m in reversed(msgs)
                   if m["role"] == "tool" and m.get("name") == "world_state")
        if n <= len(plan):
            kind, args = plan[n - 1]
            return {"tool_calls": [{"name": "act", "arguments": {"actions": [
                {"robot": "arm", "kind": kind, "expected_generation": gen, **args}]}}]}
        if n == len(plan) + 1:
            return {"tool_calls": [{"name": "check", "arguments": {"goal": GOAL}}]}
        return "Lifted."
    return fn


def test_role_campaign_proposes_verifies_and_evaluates(tmp_path):
    if not world_available(tmp_path):
        pytest.skip("the mujoco.world surface or the Panda model is not available")
    spec = {"name": "lift", "max_tool_calls": 40, "tasks": [
        {"id": "dev_lift", "partition": "dev", "stage": stage(0.5), "goal": GOAL, "request": "lift the cube"},
        {"id": "held_lift", "partition": "heldout", "stage": stage(0.45), "goal": GOAL, "request": "lift the cube"}]}

    class Router:                                         # one scripted model per task, chosen by the request
        native_tools, vision, name = True, False, "scripted:router"

        def __init__(self):
            self.dev, self.held = ScriptedChat(worker_script(0.5)), ScriptedChat(worker_script(0.45))

        def chat(self, msgs, **kw):
            first = next(m["content"] for m in msgs if m["role"] == "user")
            held = "held_lift" in first
            return (self.held if held else self.dev).chat(msgs, **kw)

    st = RoleCampaign(spec, tmp_path / "out", Router()).run()
    assert st["tasks"]["dev_lift"]["state"] == "done" and st["tasks"]["held_lift"]["state"] == "done", st["events"]
    skills = {s["id"]: s for s in st["skills"]}
    sk = skills["dev_lift-recipe@1"]
    assert sk["status"] == "verified" and sk["kind"] == "recipe" and sk["trigger"]["reason_codes"] == ["NOT_LIFTED"]
    assert [x["tool"] for x in sk["body"]["steps"]] == ["act"] * 4
    held_run = st["tasks"]["held_lift"]["runs"][0]
    assert held_run["skills_offered"] and held_run["start_reasons"] == ["NOT_LIFTED"]
    ledger = (tmp_path / "out" / "ledger" / "dev_lift" / "w1.jsonl").read_text().splitlines()
    assert len(ledger) >= 6 and all("args_sha" in json.loads(x) for x in ledger)
    states = [e["state"] for e in st["events"] if e.get("task") == "dev_lift" and "state" in e]
    assert states == ["working", "verifying", "done"]


def test_ledger_budget_and_recipe_extraction(tmp_path):
    s = Surface([ToolDef("a", "act", Help("a"), obj({"actions": {"type": "array", "description": "x"}}),
                         lambda a: {"receipts": [{"status": "completed"}]}),
                 ToolDef("r", "read", Help("r"), obj(), lambda a: {})])
    led = Ledger(s, tmp_path / "l.jsonl", max_calls=2)
    led.call("a", {"actions": [{"robot": "x", "expected_generation": 3, "action_id": "q"}]})
    led.call("r", {})
    assert led.call("r", {}).error["code"] == "campaign_budget_exceeded"
    rows = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in rows] == [True, True, False] and rows[0]["receipts"] == ["completed"]
    assert recipe_from(led.calls) == [{"tool": "a", "arguments": {"actions": [{"robot": "x"}]}}]
    assert goal_atoms({"op": "all", "terms": [{"op": "near"}, {"op": "not", "term": {"op": "lifted"}}]}) == 2
