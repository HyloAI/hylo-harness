"""Tool sessions over a surface, the MCP server, `hylo watch` and preflight's model probe (RFC-0007 / RFC-0008)."""

import io
import json

import numpy as np

from hylo.codeagents.mcp import MCPServer
from hylo.codeagents.preflight import probe_model
from hylo.codeagents.session import ToolSession
from hylo.codeagents.tool_agent import post
from hylo.codeagents.watch import condense, watch
from hylo.core.registry import registry
from hylo.core.seal import chains
from hylo.core.tooling import Help, Surface, ToolDef, ToolError, ToolOutput, obj
from hylo.reasoners.llm.chat import ScriptedChat


def counter_surface():
    state = {"n": 0}

    def add(a):
        if a["by"] > 5:
            raise ToolError("too_big", "add at most 5 at a time", field="by")
        state["n"] += a["by"]
        return {"n": state["n"]}

    def show(a):
        return ToolOutput({"n": state["n"]}, [np.full((4, 4, 3), state["n"], np.uint8)], ["counter"])

    return Surface([
        ToolDef("read", "read", Help("the counter"), obj(), lambda a: {"n": state["n"]}),
        ToolDef("add", "act", Help("add to the counter"), obj({"by": {"type": "integer", "description": "amount"}},
                                                              ["by"]), add),
        ToolDef("view", "read", Help("a picture of the counter"), obj(), show),
    ], name="counter"), state


def test_session_runs_tools_until_the_model_answers(tmp_path):
    surface, state = counter_surface()
    inbox = tmp_path / "in.jsonl"

    def script(msgs):
        n = sum(1 for m in msgs if m["role"] == "assistant")
        if n == 0:
            post(inbox, "please also read it back")
            return {"text": "adding", "tool_calls": [{"name": "add", "arguments": {"by": 9}},
                                                     {"name": "read", "arguments": {}}]}
        if n == 1:
            last = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
            assert last[-2]["ok"] and last[-1]["error"]["code"] == "too_big"     # reads run before acts
            return {"tool_calls": [{"name": "add", "arguments": {"by": 4}}, {"name": "add", "arguments": {"by": 3}}]}
        if n == 2:
            assert any("OPERATOR MESSAGE" in str(m.get("content")) for m in msgs)
            assert any(m["role"] == "user" and m.get("images") for m in msgs)    # the after-act view
            return {"tool_calls": [{"name": "read", "arguments": {}}]}
        return "The counter is at 7."

    s = ToolSession(ScriptedChat(script), surface, out=tmp_path / "trace", inbox=str(inbox), after_act="view")
    assert s.ask("add 9, then tell me the value") == "The counter is at 7."
    assert state["n"] == 7
    s.close()
    ch = chains(tmp_path / "trace")
    assert len(ch) == 1 and all(v["ok"] and v["chained"] == v["lines"] for v in ch.values())
    out = io.StringIO()
    watch(tmp_path / "trace", follow=False, out=out, colour=False)
    text = out.getvalue()
    assert "▶ add(by=4)" in text and "too_big" in text and "operator: please also read it back" in text


def test_mcp_server_lists_and_calls_tools():
    surface, state = counter_surface()
    srv = MCPServer(surface)
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "add", "arguments": {"by": 2}}},
             {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "add", "arguments": {"by": 8}}},
             {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "view", "arguments": {}}},
             {"jsonrpc": "2.0", "id": 6, "method": "nope"}]
    out = io.StringIO()
    srv.serve(io.StringIO("\n".join(json.dumps(x) for x in lines) + "\nnot json\n"), out)
    res = [json.loads(x) for x in out.getvalue().splitlines()]
    by = {r["id"]: r for r in res}
    assert by[1]["result"]["capabilities"]["tools"] and len(by[2]["result"]["tools"]) == 3
    assert {t["name"]: t["annotations"]["readOnlyHint"] for t in by[2]["result"]["tools"]} == \
        {"read": True, "add": False, "view": True}
    assert not by[3]["result"]["isError"] and json.loads(by[3]["result"]["content"][0]["text"])["n"] == 2
    assert by[4]["result"]["isError"] and state["n"] == 2
    assert by[5]["result"]["content"][1]["type"] == "image" and by[5]["result"]["content"][1]["mimeType"] == "image/png"
    assert by[6]["error"]["code"] == -32601 and by[None]["error"]["code"] == -32700


def test_condense_receipts_and_errors():
    s, ok = condense("grasp", json.dumps({"ok": True, "action_id": "a1", "status": "arrived", "steps": 40,
                                          "stages": [{"stage": "approach", "tcp_error_mm": 1.5}], "holding": True}))
    assert ok and "arrived" in s and "1.5 mm" in s and "holding=True" in s
    s, ok = condense("locate", json.dumps({"ok": False, "error": {"code": "not_found", "message": "nothing"}}))
    assert not ok and s.startswith("not_found")


def test_probe_model_with_a_scripted_model():
    def script(msgs):
        last = msgs[-1]
        if last["role"] == "tool":
            return "The sum is 42."
        if "add_numbers" in str(last.get("content")):
            return {"tool_calls": [{"name": "add_numbers", "arguments": {"a": 17, "b": 25}}]}
        if last.get("images"):
            return "red"
        return "READY"

    r = probe_model(ScriptedChat(script))
    assert r["status"] == "pass" and r["text"] and r["image"] and r["tool_call"] and r["tool_result"]


def test_surfaces_group_is_registered():
    names = registry.names("surfaces") if hasattr(registry, "names") else None
    if names is not None:
        assert "mujoco.world" in names
