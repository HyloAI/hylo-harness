"""Minimal JSON-mode chat client for Ollama (native API) and OpenAI-compatible endpoints.

Standard library only. Responses are cached on disk by prompt hash so that evaluations are
reproducible and re-runs do not hit the model; every result reports whether it was cached.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

from hylo.core.contract import Contract


class LLMResult(Contract):
    content: dict[str, Any]
    raw: str
    model: str
    latency_s: float
    cached: bool
    prompt_hash: str
    tokens_in: int | None = None
    tokens_out: int | None = None


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise LLMError(f"model did not return JSON: {text[:200]!r}") from None
        return json.loads(m.group(0))


class LLMClient:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434",
                 api: Literal["ollama", "openai"] = "ollama", api_key_env: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 300, timeout_s: float = 120.0,
                 cache_dir: str | None = None, think: bool = False) -> None:
        self.model, self.base_url, self.api = model, base_url.rstrip("/"), api
        self.api_key_env, self.temperature, self.max_tokens = api_key_env, temperature, max_tokens
        self.timeout_s, self.think = timeout_s, think
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _hash(self, messages: list[dict[str, Any]], images: list[bytes]) -> str:
        blob = json.dumps({"model": self.model, "api": self.api, "t": self.temperature, "messages": messages,
                           "images": [hashlib.sha256(i).hexdigest() for i in images]}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.api} request to {self.base_url}{path} failed: {exc!r}") from exc

    def chat_json(self, messages: list[dict[str, Any]], images: list[bytes] | None = None) -> LLMResult:
        images = images or []
        h = self._hash(messages, images)
        if self.cache and (self.cache / f"{h}.json").exists():
            rec = json.loads((self.cache / f"{h}.json").read_text())
            return LLMResult(**{**rec, "cached": True, "latency_s": 0.0})
        t0 = time.perf_counter()
        if self.api == "ollama":
            msgs = [dict(m) for m in messages]
            if images:
                msgs[-1]["images"] = [base64.b64encode(i).decode() for i in images]
            out = self._post("/api/chat", {"model": self.model, "messages": msgs, "stream": False,
                                           "think": self.think, "format": "json",
                                           "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}, {})
            raw, tin, tout = out["message"]["content"], out.get("prompt_eval_count"), out.get("eval_count")
        else:
            msgs = [dict(m) for m in messages]
            if images:
                parts = [{"type": "text", "text": msgs[-1]["content"]}] + [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
                    for i in images]
                msgs[-1]["content"] = parts
            key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
            out = self._post("/v1/chat/completions",
                             {"model": self.model, "messages": msgs, "temperature": self.temperature,
                              "max_tokens": self.max_tokens, "response_format": {"type": "json_object"}},
                             {"Authorization": f"Bearer {key}"} if key else {})
            raw = out["choices"][0]["message"]["content"] or ""
            tin, tout = out.get("usage", {}).get("prompt_tokens"), out.get("usage", {}).get("completion_tokens")
        result = LLMResult(content=_parse_json(raw), raw=raw, model=self.model, latency_s=time.perf_counter() - t0,
                           cached=False, prompt_hash=h, tokens_in=tin, tokens_out=tout)
        if self.cache:
            (self.cache / f"{h}.json").write_text(result.model_dump_json())
        return result
