"""Chat backends (RFC-0006 part 7): request shapes, images, accounting; CLIs against fake executables."""

import base64
import json
import os
import stat
import sys
import urllib.request

import numpy as np
import pytest

from hylo.reasoners.llm.chat import (AnthropicChat, ChatError, ClaudeCodeChat, CodexChat, OpenAIChat, Price,
                                     ScriptedChat, Usage, build_chat, png_bytes)

IMG = np.zeros((4, 4, 3), np.uint8)
CONV = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "what is this?", "images": [IMG]}]


def test_openai_compatible_request_images_and_cost(monkeypatch):
    seen = {}

    def post(self, url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"model": "deepseek-chat", "choices": [{"message": {"content": "a square", "reasoning_content": "hmm"},
                                                        "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_cache_hit_tokens": 400}}

    monkeypatch.setattr(OpenAIChat, "_post", post)
    monkeypatch.setenv("TEST_KEY", "sk-test")
    m = build_chat({"backend": "openai", "model": "deepseek-chat", "base_url": "https://api.example.com/v1",
                    "api_key_env": "TEST_KEY", "vision": True, "price": {"input": 0.3, "output": 1.2, "cached_input": 0.03}})
    r = m.chat(CONV, max_tokens=50)
    assert seen["url"] == "https://api.example.com/v1/chat/completions" and seen["headers"]["Authorization"] == "Bearer sk-test"
    parts = seen["body"]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"} and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["body"]["max_tokens"] == 50 and seen["body"]["temperature"] == 0.0
    assert r.text == "a square" and r.reasoning == "hmm" and r.tokens_in == 1000 and r.cached_tokens == 400
    assert r.cost_usd == pytest.approx((600 * 0.3 + 400 * 0.03 + 200 * 1.2) / 1e6)
    assert m.name == "openai:deepseek-chat"
    monkeypatch.delenv("TEST_KEY")
    with pytest.raises(ChatError, match="TEST_KEY"):
        m.chat(CONV)


def test_ollama_shorthand_is_local_and_free_and_text_models_get_no_images(monkeypatch):
    seen = {}
    monkeypatch.setattr(OpenAIChat, "_post", lambda self, url, body, headers: seen.update(body=body, url=url) or {
        "choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    m = build_chat("ollama:gpt-oss:20b")
    r = m.chat(CONV)
    assert seen["url"] == "http://localhost:11434/v1/chat/completions" and m.name == "ollama:gpt-oss:20b"
    assert seen["body"]["messages"][1]["content"] == "what is this?"          # no vision: the image is dropped
    assert r.cost_usd == 0.0 and r.backend == "ollama"


def test_anthropic_messages_request(monkeypatch):
    seen = {}

    def post(self, url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"model": "m", "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 90}}

    monkeypatch.setattr(AnthropicChat, "_post", post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    m = AnthropicChat(model="m", price={"input": 1.0, "output": 5.0, "cached_input": 0.1})
    r = m.chat([*CONV, {"role": "assistant", "content": "a"}, {"role": "user", "content": "b"},
                {"role": "user", "content": "c"}])
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "k"
    assert body["system"] == "be brief" and [t["role"] for t in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][0]["content"][0]["type"] == "image" and len(body["messages"][2]["content"]) == 2
    assert r.text == "hello" and r.tokens_in == 100 and r.cost_usd == pytest.approx((10 * 1 + 90 * 0.1 + 3 * 5) / 1e6)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ChatError, match="ANTHROPIC_API_KEY"):
        m.chat(CONV)


def test_http_errors_become_chat_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ChatError, match="connection refused"):
        OpenAIChat(model="x", retries=0).chat(CONV)


def _fake_cli(tmp_path, name, script):
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{script}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_claude_code_cli_backend(tmp_path):
    binary = _fake_cli(tmp_path, "claude", """
import json, sys
prompt = sys.stdin.read()
assert "-p" in sys.argv and "json" in sys.argv
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "```python\\nprint(1)\\n```",
                  "total_cost_usd": 0.0123, "usage": {"input_tokens": 7, "output_tokens": 9,
                  "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}, "prompt_len": len(prompt)}))
""")
    m = build_chat({"backend": "claude-code", "binary": binary, "model": "sonnet"})
    r = m.chat(CONV)
    assert r.text.startswith("```python") and r.cost_usd == 0.0123 and r.tokens_in == 107 and r.tokens_out == 9
    assert m.vision is False and ClaudeCodeChat.available(binary)
    bad = _fake_cli(tmp_path, "claude_bad", "import sys; print('nope', file=sys.stderr); sys.exit(2)")
    with pytest.raises(ChatError, match="nope"):
        ClaudeCodeChat(binary=bad).chat(CONV)


def test_codex_cli_backend(tmp_path):
    binary = _fake_cli(tmp_path, "codex", """
import sys
args = sys.argv
prompt = sys.stdin.read()
assert args[1] == "exec" and args[-1] == "-" and "=== USER ===" in prompt and "image(s) omitted" in prompt
open(args[args.index("-o") + 1], "w").write("the reply")
print("tokens used: 1,234", file=sys.stderr)
""")
    r = CodexChat(binary=binary).chat(CONV)
    assert r.text == "the reply" and r.tokens_out == 1234 and r.backend == "codex"
    assert not CodexChat.available(str(tmp_path / "missing"))
    with pytest.raises(ChatError, match="not found"):
        CodexChat(binary=str(tmp_path / "missing")).chat(CONV)


def test_scripted_model_and_usage_totals():
    m = ScriptedChat(["one", "two"], then="again", price={"input": 1.0, "output": 2.0})
    u = Usage()
    for _ in range(4):
        u.add(m.chat([{"role": "user", "content": "x" * 400}]))
    assert [c[-1]["content"][:1] for c in m.calls] == ["x"] * 4
    assert u.calls == 4 and u.tokens_in == 400 and u.tokens_out == 2           # len("again") // 4 == 1, twice
    assert u.cost_usd == pytest.approx((400 * 1.0 + 2 * 2.0) / 1e6)
    assert build_chat(m) is m
    with pytest.raises(ChatError):
        ScriptedChat([]).chat([])


def test_png_and_price_helpers():
    assert png_bytes(IMG)[:8] == b"\x89PNG\r\n\x1a\n" and png_bytes(b"raw") == b"raw"
    assert png_bytes(np.ones((2, 2, 3)) * 0.5)[:4] == b"\x89PNG"
    assert Price(input=2.0, output=4.0).cost(1_000_000, 500_000) == 4.0


@pytest.mark.skipif(os.environ.get("HYLO_TEST_LLM") != "1", reason="set HYLO_TEST_LLM=1 to call the local Ollama server")
def test_live_ollama_vision_model_sees_an_image():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :, 0] = 230
    m = build_chat({"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True, "reasoning_effort": "none",
                    "max_tokens": 20})
    r = m.chat([{"role": "user", "content": "One word: what colour is this image?", "images": [img]}])
    assert "red" in r.text.lower() and r.tokens_in and r.latency_s > 0
    assert base64.b64encode(png_bytes(img))
    assert json.dumps(r.accounting())


def test_a_harmony_tool_channel_reply_is_recovered_as_a_code_block(monkeypatch):
    import json

    from hylo.reasoners.llm.chat import ChatError, build_chat

    body = json.dumps({"error": {"message": "error parsing tool call: raw='view = robot.look()\nprint(\"ok\")\n', "
                                            "err=invalid character 'v' looking for beginning of value"}})

    def post(self, url, b, headers):
        raise ChatError(f"{url}: HTTP 500: {body[:300]}", detail=body)

    monkeypatch.setattr(OpenAIChat, "_post", post)
    reply = build_chat("ollama:gpt-oss:20b").chat([{"role": "user", "content": "go"}])
    assert reply.text == '```python\nview = robot.look()\nprint("ok")\n```'
    assert reply.stop_reason == "tool_call_recovered" and reply.cost_usd == 0.0
