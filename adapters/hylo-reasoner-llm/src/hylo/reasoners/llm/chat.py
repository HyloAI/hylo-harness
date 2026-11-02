"""Chat models for tool-using agents (RFC-0006 part 7): one interface, images, per-call accounting.

    model = build_chat({"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True})
    reply = model.chat([{"role": "system", "content": "..."},
                        {"role": "user", "content": "...", "images": [rgb_array]}])
    reply.text, reply.tokens_in, reply.tokens_out, reply.latency_s, reply.cost_usd

Backends (the `backend` key of a spec):
  openai       any OpenAI-compatible /chat/completions endpoint (vLLM, DeepSeek, OpenRouter, ...); the key is read
               from the environment variable named by `api_key_env`; `price` gives USD per million tokens
  ollama       the same against a local Ollama server (http://localhost:11434/v1), price 0
  anthropic    the Anthropic Messages API (key from ANTHROPIC_API_KEY unless `api_key_env` says otherwise)
  claude_code  the Claude Code CLI as a subprocess (`claude -p --output-format json`, prompt on stdin)
  codex        the Codex CLI as a subprocess (`codex exec ... -`, prompt on stdin)
  scripted     a deterministic fake for tests: canned replies, or a function of the conversation

Messages are dicts {"role": "system" | "user" | "assistant", "content": str, "images": [...]}; an image is an
(H, W, 3) uint8 array or PNG bytes, sent only to models with `vision`. Keys never appear in configs or traces.
Standard library only (and Pillow for PNG encoding).

Native tool calls (RFC-0007): `chat(messages, tools=[ToolSpec | {"name", "description", "parameters"}])`. The reply's
`tool_calls` are `ToolCall(id, name, arguments)`; a call whose arguments are not a JSON object keeps them in `raw`
with `error` set, so the agent can answer it instead of crashing. The conversation stays in one neutral shape and
each backend translates it at the wire:

    {"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}], "reasoning": "..."}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "<result text>", "images": [...]}

OpenAI-compatible endpoints get `tools` as functions (images of tool results follow in a user message, since tool
messages carry text only there); `replay_reasoning` sends an assistant turn's reasoning back as
`reasoning_content` (needed by endpoints that continue a reasoning chain across tool calls). Anthropic gets
`tool_use` / `tool_result` blocks. Backends without tool support (the coding-agent CLIs) are wrapped by
`TextTools`, which describes the tools in the system message and parses fenced `tool` blocks from the text.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from hylo.core.contract import Contract

Message = Mapping[str, Any]


class ChatError(RuntimeError):
    """The model could not be reached or returned nothing usable. `detail`: the server's full error body, if any."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


# Ollama serves Harmony-format models (gpt-oss) with a parser that turns a reply on the tool channel into a tool call;
# a model that writes its Python there (it was trained with a python tool) gets HTTP 500 "error parsing tool call:
# raw='<the code>'". The code is the reply: it is recovered as a fenced block instead of failing the turn.
TOOL_CALL_RAW = re.compile(r"error parsing tool call: raw='(.*)', err=", re.S)


def recover_tool_call(detail: str) -> str | None:
    """The code of a Harmony tool-channel reply from the server's error body (JSON, or its message), else None."""
    text = detail or ""
    try:
        body = json.loads(text)
        err = body.get("error") if isinstance(body, dict) else None
        text = str(err.get("message", "") if isinstance(err, dict) else err or text)
    except ValueError:
        pass
    m = TOOL_CALL_RAW.search(text)
    return m.group(1) if m else None


class ToolSpec(Contract):
    """One tool as a model sees it: a name, a description and a JSON Schema for its arguments (an object)."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(Contract):
    id: str
    name: str
    arguments: dict[str, Any] | None = None   # None when the model's arguments were not a JSON object
    raw: str = ""                             # the arguments as the model wrote them
    error: str | None = None                  # why `arguments` could not be read

    def wire(self) -> dict[str, Any]:
        """The neutral form kept in an assistant message."""
        return {"id": self.id, "name": self.name,
                "arguments": self.arguments if self.arguments is not None else self.raw}


def _tool_specs(tools: Sequence[Any] | None) -> list[ToolSpec]:
    return [t if isinstance(t, ToolSpec) else ToolSpec.model_validate(t) for t in tools or ()]


def parse_arguments(name: str, call_id: str, raw: Any) -> ToolCall:
    if isinstance(raw, Mapping):
        return ToolCall(id=call_id, name=name, arguments=dict(raw), raw=json.dumps(raw))
    text = str(raw or "").strip() or "{}"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, raw=text, error=f"arguments are not valid JSON ({exc.msg} at {exc.pos})")
    if not isinstance(value, dict):
        return ToolCall(id=call_id, name=name, raw=text, error="arguments must be a JSON object")
    return ToolCall(id=call_id, name=name, arguments=value, raw=text)


class ChatReply(Contract):
    text: str
    model: str
    backend: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_tokens: int | None = None
    latency_s: float = 0.0
    cost_usd: float | None = None          # None: the backend has no price configured
    stop_reason: str | None = None
    reasoning: str | None = None           # hidden reasoning some endpoints return separately (kept out of prompts)
    tool_calls: tuple[ToolCall, ...] = ()

    def message(self) -> dict[str, Any]:
        """This reply as an assistant message of the neutral conversation."""
        m: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            m["tool_calls"] = [c.wire() for c in self.tool_calls]
        if self.reasoning:
            m["reasoning"] = self.reasoning
        return m

    def accounting(self) -> dict[str, Any]:
        return {"model": self.model, "backend": self.backend, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "cached_tokens": self.cached_tokens,
                "latency_s": round(self.latency_s, 3), "cost_usd": self.cost_usd}


class Price(Contract):
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cached_input: float | None = None      # cache hits, when the endpoint reports them (default: the input price)

    def cost(self, tokens_in: int | None, tokens_out: int | None, cached: int | None = None) -> float:
        tin, tout, hit = int(tokens_in or 0), int(tokens_out or 0), int(cached or 0)
        rate_hit = self.input if self.cached_input is None else self.cached_input
        return round(((tin - hit) * self.input + hit * rate_hit + tout * self.output) / 1e6, 8)


@runtime_checkable
class ChatModel(Protocol):
    name: str                              # "<backend>:<model>", recorded with every call
    vision: bool

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None) -> ChatReply: ...


def supports_tools(model: Any) -> bool:
    return bool(getattr(model, "native_tools", False))


@dataclass
class Usage:
    """Running totals over calls (a budget reads these)."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    latency_s: float = 0.0

    def add(self, reply: ChatReply) -> None:
        self.calls += 1
        self.tokens_in += int(reply.tokens_in or 0)
        self.tokens_out += int(reply.tokens_out or 0)
        self.latency_s += reply.latency_s
        if reply.cost_usd is None:
            self.cost_known = False
        else:
            self.cost_usd += reply.cost_usd

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6) if self.cost_known else None,
                "cost_usd_known_part": round(self.cost_usd, 6), "latency_s": round(self.latency_s, 2)}


def png_bytes(image: Any) -> bytes:
    """PNG bytes of an (H, W, 3|4) or (H, W) uint8 array (floats in [0, 1] are scaled); bytes pass through."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.size and float(np.nanmax(arr)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _b64(image: Any) -> str:
    return base64.b64encode(png_bytes(image)).decode()


def _http_json(url: str, body: dict[str, Any], headers: Mapping[str, str], timeout_s: float, retries: int) -> dict[str, Any]:
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            full = exc.read().decode(errors="replace")
            detail = full[:300]
            last = ChatError(f"{url}: HTTP {exc.code}: {detail}", detail=full)
            if exc.code not in (408, 429, 500, 502, 503, 504, 529) or TOOL_CALL_RAW.search(full):
                break                                    # a deterministic failure: retrying repeats it
        except Exception as exc:  # noqa: BLE001 - connection errors, timeouts
            last = ChatError(f"{url}: {exc!r}")
        if attempt < retries:
            time.sleep(min(2.0 ** attempt, 10.0))
    raise last or ChatError(url)


# ---------------------------------------------------------------- OpenAI-compatible (Ollama, vLLM, DeepSeek, ...)

class OpenAIChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key_env: str | None = None,
                 vision: bool = False, temperature: float | None = 0.0, max_tokens: int = 4096,
                 timeout_s: float = 600.0, price: Mapping[str, float] | Price | None = None,
                 reasoning_effort: str | None = None, extra_body: Mapping[str, Any] | None = None,
                 retries: int = 2, backend: str = "openai", replay_reasoning: bool = False) -> None:
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.replay_reasoning, self.native_tools = replay_reasoning, True
        self.vision, self.temperature, self.max_tokens, self.timeout_s = vision, temperature, max_tokens, timeout_s
        self.price = Price.model_validate(price) if isinstance(price, Mapping) else price
        self.reasoning_effort, self.extra_body, self.retries = reasoning_effort, dict(extra_body or {}), retries
        self.backend = backend
        self.name = f"{backend}:{model}"

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return _http_json(url, body, headers, self.timeout_s, self.retries)

    def _messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending: list[tuple[str, list[Any]]] = []      # images of tool results, sent after the tool messages
        for m in messages:
            if m["role"] != "tool" and pending:
                out.append(self._tool_images(pending))
                pending = []
            images = list(m.get("images") or []) if self.vision else []
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": str(m.get("tool_call_id", "")),
                            "content": str(m.get("content", ""))})
                if images:
                    pending.append((str(m.get("name") or m.get("tool_call_id")), images))
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                a: dict[str, Any] = {"role": "assistant", "content": str(m.get("content") or "") or None,
                                     "tool_calls": [{"id": c["id"], "type": "function", "function": {
                                         "name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str)
                                         else json.dumps(c["arguments"])}} for c in m["tool_calls"]]}
                if self.replay_reasoning and m.get("reasoning"):
                    a["reasoning_content"] = m["reasoning"]
                out.append(a)
                continue
            if images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": str(m.get("content", ""))}]
                parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(i)}} for i in images]
                out.append({"role": m["role"], "content": parts})
            else:
                a = {"role": m["role"], "content": str(m.get("content", ""))}
                if self.replay_reasoning and m["role"] == "assistant" and m.get("reasoning"):
                    a["reasoning_content"] = m["reasoning"]
                out.append(a)
        if pending:
            out.append(self._tool_images(pending))
        return out

    def _tool_images(self, pending: list[tuple[str, list[Any]]]) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for name, images in pending:
            parts.append({"type": "text", "text": f"[{len(images)} image(s) returned by {name}]"})
            parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(i)}} for i in images]
        return {"role": "user", "content": parts}

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None,
             tools: Sequence[Any] | None = None) -> ChatReply:
        body: dict[str, Any] = {"model": self.model, "messages": self._messages(messages),
                                "max_tokens": int(max_tokens or self.max_tokens), **self.extra_body}
        specs = _tool_specs(tools)
        if specs:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.parameters}} for t in specs]
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if self.api_key_env and not key:
            raise ChatError(f"{self.name}: environment variable {self.api_key_env} is not set")
        t0 = time.perf_counter()
        try:
            out = self._post(self.base_url + "/chat/completions", body, {"Authorization": f"Bearer {key}"} if key else {})
        except ChatError as exc:
            code = recover_tool_call(exc.detail)
            if code is None or specs:
                raise
            return ChatReply(text=f"```python\n{code.strip()}\n```", model=self.model, backend=self.backend,
                             latency_s=time.perf_counter() - t0, cost_usd=0.0 if self.price is not None else None,
                             stop_reason="tool_call_recovered")
        latency = time.perf_counter() - t0
        try:
            choice = out["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"{self.name}: unexpected response {str(out)[:300]}") from exc
        usage = out.get("usage") or {}
        tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get("prompt_cache_hit_tokens")
        calls = tuple(parse_arguments(str((c.get("function") or {}).get("name", "")), str(c.get("id") or f"call_{i}"),
                                      (c.get("function") or {}).get("arguments"))
                      for i, c in enumerate(msg.get("tool_calls") or []))
        return ChatReply(text=msg.get("content") or "", model=str(out.get("model") or self.model), backend=self.backend,
                         tokens_in=tin, tokens_out=tout, cached_tokens=cached, latency_s=latency,
                         cost_usd=self.price.cost(tin, tout, cached) if self.price is not None else None,
                         stop_reason=choice.get("finish_reason"),
                         reasoning=msg.get("reasoning") or msg.get("reasoning_content") or None, tool_calls=calls)


# ---------------------------------------------------------------- Anthropic Messages API

class AnthropicChat:
    def __init__(self, model: str, base_url: str = "https://api.anthropic.com", api_key_env: str = "ANTHROPIC_API_KEY",
                 vision: bool = True, temperature: float | None = None, max_tokens: int = 4096,
                 timeout_s: float = 600.0, price: Mapping[str, float] | Price | None = None,
                 api_version: str = "2023-06-01", retries: int = 2) -> None:
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.vision, self.temperature, self.max_tokens, self.timeout_s = vision, temperature, max_tokens, timeout_s
        self.price = Price.model_validate(price) if isinstance(price, Mapping) else price
        self.api_version, self.retries = api_version, retries
        self.name = f"anthropic:{model}"
        self.native_tools = True

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return _http_json(url, body, headers, self.timeout_s, self.retries)

    def _body(self, messages: Sequence[Message], max_tokens: int) -> dict[str, Any]:
        system = "\n\n".join(str(m.get("content", "")) for m in messages if m["role"] == "system")
        turns: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            blocks: list[dict[str, Any]] = []
            role = "user" if m["role"] == "tool" else m["role"]
            images = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(i)}}
                      for i in (m.get("images") or [])] if self.vision and role == "user" else []
            if m["role"] == "tool":
                blocks.append({"type": "tool_result", "tool_use_id": str(m.get("tool_call_id", "")),
                               "content": [{"type": "text", "text": str(m.get("content", "")) or "(empty)"}, *images]})
            else:
                blocks += images
                if str(m.get("content") or "") or not m.get("tool_calls"):
                    blocks.append({"type": "text", "text": str(m.get("content", "")) or "(empty)"})
                for c in m.get("tool_calls") or []:
                    args = c["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_unparsed": args}
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                                   "input": args if isinstance(args, dict) else {"_value": args}})
            if turns and turns[-1]["role"] == role:                # the API wants alternating roles
                prev = turns[-1]["content"]
                # tool results must come first in their user turn
                turns[-1]["content"] = ([b for b in prev + blocks if b["type"] == "tool_result"]
                                        + [b for b in prev + blocks if b["type"] != "tool_result"])
            else:
                turns.append({"role": role, "content": blocks})
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": turns}
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None,
             tools: Sequence[Any] | None = None) -> ChatReply:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ChatError(f"{self.name}: environment variable {self.api_key_env} is not set")
        body = self._body(messages, int(max_tokens or self.max_tokens))
        specs = _tool_specs(tools)
        if specs:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                             for t in specs]
        t0 = time.perf_counter()
        out = self._post(self.base_url + "/v1/messages", body,
                         {"x-api-key": key, "anthropic-version": self.api_version})
        latency = time.perf_counter() - t0
        text = "".join(b.get("text", "") for b in out.get("content") or [] if b.get("type") == "text")
        usage = out.get("usage") or {}
        tin, tout, cached = usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cache_read_input_tokens")
        if tin is not None and cached:
            tin = int(tin) + int(cached)                      # input_tokens excludes cache reads
        return ChatReply(text=text, model=str(out.get("model") or self.model), backend="anthropic", tokens_in=tin,
                         tokens_out=tout, cached_tokens=cached, latency_s=latency,
                         cost_usd=self.price.cost(tin, tout, cached) if self.price is not None else None,
                         stop_reason=out.get("stop_reason"),
                         tool_calls=tuple(parse_arguments(str(b.get("name", "")), str(b.get("id", "")), b.get("input"))
                                          for b in out.get("content") or [] if b.get("type") == "tool_use"))


# ---------------------------------------------------------------- coding-agent CLIs

def flatten(messages: Sequence[Message]) -> str:
    """One prompt for a stateless CLI call: the conversation so far, then the request to answer the last turn."""
    parts = []
    for m in messages:
        n = len(m.get("images") or [])
        note = f"\n[{n} image(s) omitted: this model receives text only]" if n else ""
        parts.append(f"=== {m['role'].upper()} ===\n{m.get('content', '')}{note}")
    parts.append("=== INSTRUCTIONS ===\nYou are the ASSISTANT in the conversation above. Write only the assistant's next "
                 "message, in the format the SYSTEM message asks for. Do not run tools or edit files.")
    return "\n\n".join(parts)


class _CLIChat:
    tool = ""
    env_var = ""

    def __init__(self, model: str | None = None, binary: str | None = None, timeout_s: float = 900.0,
                 extra_args: Sequence[str] = (), workdir: str | None = None) -> None:
        self.model, self.timeout_s, self.extra_args, self.workdir = model, timeout_s, list(extra_args), workdir
        self.binary = binary or os.environ.get(self.env_var) or shutil.which(self.tool)
        self.vision = False
        self.name = f"{self.tool}:{model or 'default'}"

    @classmethod
    def available(cls, binary: str | None = None) -> bool:
        b = binary or os.environ.get(cls.env_var) or shutil.which(cls.tool)
        return bool(b) and Path(b).exists()

    def _run(self, cmd: list[str], prompt: str, cwd: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired as exc:
            raise ChatError(f"{self.name}: no reply within {self.timeout_s:.0f} s") from exc
        except OSError as exc:
            raise ChatError(f"{self.name}: cannot run {cmd[0]}: {exc}") from exc

    def _binary(self) -> str:
        if not self.binary or not Path(self.binary).exists():
            raise ChatError(f"{self.tool} CLI not found (install it, or set {self.env_var} or `binary`)")
        return self.binary


class ClaudeCodeChat(_CLIChat):
    """`claude -p --output-format json`, the conversation flattened into one prompt on stdin, in a scratch directory."""

    tool, env_var = "claude", "HYLO_CLAUDE_BIN"

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None) -> ChatReply:
        cmd = [self._binary(), "-p", "--output-format", "json", *(["--model", self.model] if self.model else []),
               *self.extra_args]
        with tempfile.TemporaryDirectory(prefix="hylo-claude-") as tmp:
            t0 = time.perf_counter()
            proc = self._run(cmd, flatten(messages), self.workdir or tmp)
            latency = time.perf_counter() - t0
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except json.JSONDecodeError:
            out = {}
        if not out or out.get("is_error") or proc.returncode != 0:
            raise ChatError(f"{self.name}: exit {proc.returncode}: {(out.get('result') or proc.stderr or proc.stdout)[:300]}")
        usage = out.get("usage") or {}
        cached = usage.get("cache_read_input_tokens")
        tin = usage.get("input_tokens")
        if tin is not None:
            tin = int(tin) + int(cached or 0) + int(usage.get("cache_creation_input_tokens") or 0)
        return ChatReply(text=str(out.get("result") or ""), model=self.model or "claude-code", backend="claude_code",
                         tokens_in=tin, tokens_out=usage.get("output_tokens"), cached_tokens=cached, latency_s=latency,
                         cost_usd=out.get("total_cost_usd"), stop_reason=out.get("subtype"))


class CodexChat(_CLIChat):
    """`codex exec --skip-git-repo-check -s read-only -o FILE -`: prompt on stdin, the last message from FILE."""

    tool, env_var = "codex", "HYLO_CODEX_BIN"

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None) -> ChatReply:
        with tempfile.TemporaryDirectory(prefix="hylo-codex-") as tmp:
            out_file = Path(tmp) / "reply.md"
            cwd = self.workdir or tmp
            cmd = [self._binary(), "exec", "--skip-git-repo-check", "-s", "read-only", "-C", cwd, "-o", str(out_file),
                   *(["-m", self.model] if self.model else []), *self.extra_args, "-"]
            t0 = time.perf_counter()
            proc = self._run(cmd, flatten(messages), cwd)
            latency = time.perf_counter() - t0
            text = out_file.read_text() if out_file.exists() else ""
        if proc.returncode != 0 or not text.strip():
            raise ChatError(f"{self.name}: exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")
        m = re.search(r"tokens used[:\s]+([\d,]+)", proc.stderr + proc.stdout, re.I)
        total = int(m.group(1).replace(",", "")) if m else None
        return ChatReply(text=text, model=self.model or "codex", backend="codex", tokens_in=None, tokens_out=total,
                         latency_s=latency, cost_usd=None)


# ---------------------------------------------------------------- deterministic fake

class ScriptedChat:
    """Replies from a list (in order) or from `fn(messages) -> reply`. A reply is a string (text), or a dict
    {"text": ..., "tool_calls": [{"name", "arguments"} ...]} to script native tool calls. Token counts are
    len(text) // 4, so tests can exercise budgets; `calls` keeps every conversation it was shown and `tools` the tool
    lists it was offered."""

    native_tools = True

    def __init__(self, replies: Sequence[Any] | Callable[[list[Message]], Any] = (), *, then: Any = None,
                 model: str = "scripted", vision: bool = True, price: Mapping[str, float] | Price | None = None) -> None:
        self.fn = replies if callable(replies) else None
        self.replies = [] if callable(replies) else list(replies)
        self.then, self.model, self.vision = then, model, vision
        self.price = Price.model_validate(price) if isinstance(price, Mapping) else price or Price()
        self.calls: list[list[Message]] = []
        self.tools: list[list[ToolSpec]] = []
        self.name = f"scripted:{model}"
        self._n = 0

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None,
             tools: Sequence[Any] | None = None) -> ChatReply:
        msgs = [dict(m) for m in messages]
        self.calls.append(msgs)
        self.tools.append(_tool_specs(tools))
        if self.fn is not None:
            reply = self.fn(msgs)
        elif self.replies:
            reply = self.replies.pop(0)
        elif self.then is not None:
            reply = self.then
        else:
            raise ChatError("scripted model: out of replies")
        if isinstance(reply, ChatReply):
            return reply
        text, calls = (reply, []) if isinstance(reply, str) else (str(reply.get("text", "")), reply.get("tool_calls") or [])
        parsed = []
        for c in calls:
            self._n += 1
            parsed.append(parse_arguments(c["name"], str(c.get("id") or f"call_{self._n}"), c.get("arguments", {})))
        tin = sum(len(str(m.get("content", ""))) for m in msgs) // 4
        tout = (len(text) + sum(len(c.raw) for c in parsed)) // 4
        return ChatReply(text=text, model=self.model, backend="scripted", tokens_in=tin, tokens_out=tout,
                         latency_s=0.0, cost_usd=self.price.cost(tin, tout), tool_calls=tuple(parsed),
                         stop_reason="tool_calls" if parsed else "stop")


# ---------------------------------------------------------------- tools as text, for models without native tools

TOOL_BLOCK = re.compile(r"```tool[ \t]*\n(.*?)```", re.S)


class TextTools:
    """Gives a model without native tool calls the same interface: the tools are listed in the system message, the
    model writes each call as a fenced block

        ```tool
        {"name": "observe", "arguments": {...}}
        ```

    and tool results come back as user messages. The wrapped model's accounting is passed through."""

    native_tools = True

    def __init__(self, model: ChatModel) -> None:
        self.inner, self.name, self.vision = model, f"{model.name}+text-tools", getattr(model, "vision", False)
        self._n = 0

    @staticmethod
    def manual(specs: Sequence[ToolSpec]) -> str:
        lines = ["TOOLS. Call a tool by writing a fenced block with the language tag `tool` holding one JSON object "
                 '{"name": <tool>, "arguments": {...}}. Several blocks in one reply are several calls. The results '
                 "come back in the next message. Available tools:"]
        for t in specs:
            lines.append(f"\n## {t.name}\n{t.description}\nArguments (JSON Schema): {json.dumps(t.parameters)}")
        return "\n".join(lines)

    def chat(self, messages: Sequence[Message], *, max_tokens: int | None = None,
             tools: Sequence[Any] | None = None) -> ChatReply:
        specs = _tool_specs(tools)
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "user", "content": f"[result of {m.get('name', '')} {m.get('tool_call_id', '')}]\n"
                                                       f"{m.get('content', '')}", "images": m.get("images") or []})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = "\n".join("```tool\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}) + "\n```"
                                   for c in m["tool_calls"])
                out.append({"role": "assistant", "content": (str(m.get("content") or "") + "\n" + blocks).strip()})
            else:
                out.append(dict(m))
        if specs:
            manual = self.manual(specs)
            if out and out[0]["role"] == "system":
                out[0] = {**out[0], "content": f"{out[0]['content']}\n\n{manual}"}
            else:
                out.insert(0, {"role": "system", "content": manual})
        reply = self.inner.chat(out, max_tokens=max_tokens)
        calls = []
        for block in TOOL_BLOCK.findall(reply.text):
            self._n += 1
            try:
                obj = json.loads(block)
            except json.JSONDecodeError as exc:
                calls.append(ToolCall(id=f"text_{self._n}", name="", raw=block.strip(),
                                      error=f"the tool block is not valid JSON ({exc.msg})"))
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                calls.append(ToolCall(id=f"text_{self._n}", name="", raw=block.strip(),
                                      error='a tool block needs {"name": ..., "arguments": {...}}'))
                continue
            calls.append(parse_arguments(obj["name"], f"text_{self._n}", obj.get("arguments", {})))
        text = TOOL_BLOCK.sub("", reply.text).strip()
        return reply.model_copy(update={"text": text, "tool_calls": tuple(calls)})


def with_tools(model: ChatModel) -> ChatModel:
    """`model` itself when it calls tools natively, else wrapped in TextTools."""
    return model if supports_tools(model) else TextTools(model)


# ---------------------------------------------------------------- factory

BACKENDS: dict[str, Callable[..., ChatModel]] = {
    "openai": OpenAIChat, "anthropic": AnthropicChat, "claude_code": ClaudeCodeChat, "codex": CodexChat,
    "scripted": ScriptedChat,
}


def build_chat(spec: Mapping[str, Any] | str | ChatModel) -> ChatModel:
    """A chat model from a spec: {"backend": ..., **params}, a shorthand string ("ollama:qwen3.5:35b-a3b",
    "anthropic:<model>", "openai:<model>", "claude_code", "codex[:<model>]"), or a ChatModel (returned as is)."""
    if not isinstance(spec, (Mapping, str)):
        return spec
    if isinstance(spec, str):
        backend, _, model = spec.partition(":")
        spec = {"backend": backend, **({"model": model} if model else {})}
    params = dict(spec)
    backend = str(params.pop("backend", None) or params.pop("type", "ollama")).replace("-", "_")
    if backend == "ollama":
        params.setdefault("base_url", "http://localhost:11434/v1")
        params.setdefault("price", {"input": 0.0, "output": 0.0})
        return OpenAIChat(backend="ollama", **params)
    if backend not in BACKENDS:
        raise ValueError(f"unknown chat backend {backend!r}; known: ollama, {', '.join(BACKENDS)}")
    return BACKENDS[backend](**params)


def describe(model: ChatModel) -> str:
    return f"{model.name}{' (vision)' if getattr(model, 'vision', False) else ''}"


__all__ = ["AnthropicChat", "BACKENDS", "ChatError", "ChatModel", "ChatReply", "ClaudeCodeChat", "CodexChat",
           "Message", "OpenAIChat", "Price", "ScriptedChat", "TextTools", "ToolCall", "ToolSpec", "Usage", "build_chat",
           "describe", "flatten", "parse_arguments", "png_bytes", "supports_tools", "with_tools"]
