"""Native tool calls (RFC-0007): wire translation for OpenAI-compatible and Anthropic endpoints, the scripted fake
and the text-protocol fallback."""

import json

import numpy as np

from hylo.reasoners.llm.chat import (AnthropicChat, OpenAIChat, ScriptedChat, TextTools, ToolSpec, parse_arguments,
                                     with_tools)

TOOLS = [ToolSpec(name="observe", description="look", parameters={"type": "object", "properties": {
    "camera": {"type": "string"}}, "additionalProperties": False})]
IMG = np.zeros((4, 4, 3), np.uint8)
CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "reasoning": "think",
     "tool_calls": [{"id": "c1", "name": "observe", "arguments": {"camera": "front"}},
                    {"id": "c2", "name": "observe", "arguments": {"camera": "side"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "observe", "content": "ok1", "images": [IMG]},
    {"role": "tool", "tool_call_id": "c2", "name": "observe", "content": "ok2"},
]


class Capture:
    def __init__(self, reply):
        self.reply, self.body = reply, None

    def __call__(self, url, body, headers):
        self.body = body
        return self.reply


def test_openai_wire_and_parse(monkeypatch):
    m = OpenAIChat("m", vision=True, replay_reasoning=True, price={"input": 1, "output": 1})
    cap = Capture({"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "observe", "arguments": '{"camera": "wrist"}'}},
        {"id": "y", "type": "function", "function": {"name": "observe", "arguments": "{bad"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    monkeypatch.setattr(m, "_post", cap)
    r = m.chat(CONVERSATION, tools=TOOLS)
    sent = cap.body["messages"]
    assert cap.body["tools"][0]["function"]["name"] == "observe"
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"camera": "front"})
    assert sent[2]["reasoning_content"] == "think"
    assert [x["role"] for x in sent] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert sent[5]["content"][1]["type"] == "image_url"          # tool images follow the tool messages
    assert r.tool_calls[0].arguments == {"camera": "wrist"} and r.tool_calls[1].error and r.tool_calls[1].arguments is None
    assert r.message()["tool_calls"][1]["arguments"] == "{bad"


def test_anthropic_wire_and_parse(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    m = AnthropicChat("m")
    cap = Capture({"content": [{"type": "text", "text": "hi"},
                               {"type": "tool_use", "id": "t9", "name": "observe", "input": {"camera": "a"}}],
                   "usage": {"input_tokens": 3, "output_tokens": 4}, "stop_reason": "tool_use"})
    monkeypatch.setattr(m, "_post", cap)
    r = m.chat(CONVERSATION, tools=TOOLS)
    turns = cap.body["messages"]
    assert cap.body["system"] == "sys" and cap.body["tools"][0]["input_schema"]["type"] == "object"
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert [b["type"] for b in turns[1]["content"]] == ["tool_use", "tool_use"]
    results = turns[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert results[0]["content"][1]["type"] == "image"
    assert r.text == "hi" and r.tool_calls[0].name == "observe" and r.tool_calls[0].arguments == {"camera": "a"}


def test_scripted_tool_calls():
    m = ScriptedChat([{"text": "look", "tool_calls": [{"name": "observe", "arguments": {"camera": "front"}}]}, "done"])
    r = m.chat([{"role": "user", "content": "x"}], tools=TOOLS)
    assert r.tool_calls[0].arguments == {"camera": "front"} and m.tools[0][0].name == "observe"
    assert m.chat([{"role": "user", "content": "x"}]).text == "done"


def test_text_tools_round_trip():
    seen = []

    def fn(msgs):
        seen.append(msgs)
        return 'I will look.\n```tool\n{"name": "observe", "arguments": {"camera": "front"}}\n```\n```tool\nnope\n```'

    inner = ScriptedChat(fn)
    inner.native_tools = False
    m = with_tools(inner)
    assert isinstance(m, TextTools)
    r = m.chat(CONVERSATION, tools=TOOLS)
    assert r.text == "I will look."
    assert r.tool_calls[0].arguments == {"camera": "front"} and r.tool_calls[1].error
    sent = seen[0]
    assert "## observe" in sent[0]["content"]
    assert "```tool" in sent[2]["content"] and sent[3]["role"] == "user" and sent[3]["images"]


def test_parse_arguments_variants():
    assert parse_arguments("a", "1", "").arguments == {}
    assert parse_arguments("a", "1", "[1]").error
    assert parse_arguments("a", "1", {"x": 1}).arguments == {"x": 1}
