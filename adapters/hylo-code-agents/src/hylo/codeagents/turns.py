"""One conversation of a code-writing model: it writes a block, the sandbox runs it, the feedback goes back.

Shared by the `code.llm` agent (one episode) and research sessions (many episodes). Accounting per call (model,
tokens, latency, cost) and running totals; every turn becomes one record handed to `emit` (the agent writes it to
the trace as `agent.code.turn`, a research session to its log).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np
from pydantic import Field

from hylo.core.contract import Contract
from hylo.reasoners.llm.chat import ChatError, ChatModel, Usage

from .prompt import extract_code, feedback
from .sandbox import BlockResult, Sandbox


class Budget(Contract):
    turns: int = Field(default=30, ge=1)
    tokens: int | None = None          # model tokens in + out
    cost_usd: float | None = None      # applies when the backend reports a price
    steps: int | None = None           # robot control steps (code.llm)
    resets: int | None = None          # research sessions
    submits: int | None = None         # research sessions

    @classmethod
    def parse(cls, text: str | Mapping[str, Any] | None) -> "Budget":
        """From a mapping or "turns=40,cost=2.5,submits=5" ("cost" and "cost_usd" are the same)."""
        if text is None:
            return cls()
        if isinstance(text, Mapping):
            return cls.model_validate(dict(text))
        items = dict(kv.split("=", 1) for kv in str(text).replace(" ", "").split(",") if kv)
        if "cost" in items:
            items["cost_usd"] = items.pop("cost")
        return cls.model_validate({k: (float(v) if k == "cost_usd" else int(v)) for k, v in items.items()})


def prompt_digest(messages: list[Mapping[str, Any]]) -> str:
    blob = json.dumps([{"role": m["role"], "content": m.get("content", ""), "images": len(m.get("images") or [])}
                       for m in messages], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class CodeLoop:
    """Messages, the sandbox, pending images and the usage totals of one model working on one problem."""

    def __init__(self, model: ChatModel, sandbox: Sandbox, *, system: str, first: str,
                 emit: Callable[[dict[str, Any]], None], status: Callable[[], Mapping[str, Any] | None],
                 reply_tokens: int | None = None, max_history_turns: int = 24, max_feedback: int = 8000,
                 max_context_chars: int | None = 80_000) -> None:
        self.model, self.sandbox, self.emit, self.status = model, sandbox, emit, status
        self.vision = bool(getattr(model, "vision", False))
        self.reply_tokens, self.max_history_turns, self.max_feedback = reply_tokens, max_history_turns, max_feedback
        self.max_context_chars = max_context_chars     # older turns are dropped to keep the prompt under this
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": first}]
        self.usage = Usage()
        self.turn = 0
        self.images: list[dict[str, Any]] = []         # from robot.show, for the next feedback
        self.done: str | None = None                   # why the loop stopped
        self.model_errors = 0
        self.last_code: str | None = None              # to notice a block repeated verbatim

    def show(self, image: Any, note: str = "") -> None:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.size and float(np.nanmax(arr)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        self.images.append({"image": arr, "note": str(note)[:200]})

    def _history(self) -> list[dict[str, Any]]:
        """System + first message + the most recent turns (at most max_history_turns, and as many as fit in
        max_context_chars: a served model with a fixed context window would otherwise silently drop the start of
        the prompt, the system message first); images only on the latest message."""
        head, rest = self.messages[:2], self.messages[2:]
        keep = rest[-2 * self.max_history_turns:]
        if self.max_context_chars:
            size = sum(len(str(m.get("content", ""))) for m in head) + 200
            n = 0
            for m in reversed(keep):
                size += len(str(m.get("content", "")))
                if size > self.max_context_chars and n >= 2:
                    break
                n += 1
            keep = keep[len(keep) - (n - n % 2):] if n < len(keep) else keep
        if len(keep) < len(rest):
            dropped = (len(rest) - len(keep)) // 2
            head = [*head, {"role": "user", "content": f"({dropped} earlier turns omitted; their variables still exist)"},
                    {"role": "assistant", "content": "Understood."}]
        msgs = [*head, *keep]
        # images only on the latest message (and on the first while it is the latest's predecessor: the opening view)
        last = len(msgs) - 1
        return [m if i == last or (i == 1 and last <= 3) else {k: v for k, v in m.items() if k != "images"}
                for i, m in enumerate(msgs)]

    def exceeded(self, budget: Budget, steps: int = 0) -> str | None:
        if self.turn >= budget.turns:
            return f"budget: {budget.turns} turns used"
        if budget.tokens is not None and self.usage.tokens >= budget.tokens:
            return f"budget: {self.usage.tokens} model tokens used (limit {budget.tokens})"
        if budget.cost_usd is not None and self.usage.cost_usd >= budget.cost_usd:
            return f"budget: ${self.usage.cost_usd:.4f} spent (limit ${budget.cost_usd})"
        if budget.steps is not None and steps >= budget.steps:
            return f"budget: {steps} robot steps used (limit {budget.steps})"
        return None

    def step(self, remaining: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One turn: ask the model, run its block, append the feedback. Returns the turn record (also emitted)."""
        self.turn += 1
        msgs = self._history()
        digest = prompt_digest(msgs)
        try:
            reply = self.model.chat(msgs, max_tokens=self.reply_tokens)
        except ChatError as exc:
            self.model_errors += 1
            rec = {"turn": self.turn, "prompt_digest": digest, "model_error": str(exc)[:500],
                   "usage": self.usage.as_dict(), **dict(extra or {})}
            if self.model_errors >= 3:
                self.done = f"model_error: {str(exc)[:200]}"
            self.emit(rec)
            return rec
        self.model_errors = 0
        self.usage.add(reply)
        code, n_blocks = extract_code(reply.text)
        res = self.sandbox.run(code) if code is not None else BlockResult()
        images, self.images = self.images, []
        try:
            status = self.status()
        except Exception as exc:  # noqa: BLE001 - a status failure must not end the session
            status = {"error": f"status unavailable: {exc!r}"[:200]}
        text = feedback(code=code, n_blocks=n_blocks, output=res.output, error=res.error, status=status,
                        images=images, vision=self.vision, remaining=remaining, seconds=res.seconds)
        same = code is not None and code.strip() == (self.last_code or "").strip()
        self.last_code = code
        if same:                                         # a deterministic model can loop on one block forever
            text = ("This block is the same as your previous one, so it did the same thing again: change the "
                    "approach (different goals, poses or verbs), or reset to another case.\n" + text)
        if reply.stop_reason in ("length", "max_tokens"):     # cut off at the reply limit: say so, or it looks like
            text = ("Your reply hit the reply token limit and was cut off"   # a missing fence or a broken program
                    + (" before the block was closed, so nothing ran" if code is None else "")
                    + ": write a shorter reply (less prose, a smaller block).\n" + text)
        if len(text) > self.max_feedback:
            text = text[: self.max_feedback - 40] + "\n... (feedback cut)"
        self.messages.append({"role": "assistant", "content": reply.text})
        user: dict[str, Any] = {"role": "user", "content": text}
        if images and self.vision:
            user["images"] = [i["image"] for i in images]
        self.messages.append(user)
        rec = {"turn": self.turn, "prompt_digest": digest, "reply": reply.text, "code": code, "blocks": n_blocks,
               "output": res.output, "error": res.error, "timed_out": res.timed_out, "exec_s": res.seconds,
               "images": [{"note": i["note"], "image": i["image"]} for i in images],
               **reply.accounting(), "stop_reason": reply.stop_reason,
               "reasoning": (reply.reasoning or "")[-4000:] or None,     # hidden reasoning: traced, never re-sent
               "usage": self.usage.as_dict(), **dict(extra or {})}
        if self.turn == 1:
            rec["prompt"] = {"system": self.messages[0]["content"], "first": self.messages[1]["content"]}
        self.emit(rec)
        return rec


__all__ = ["Budget", "CodeLoop", "prompt_digest"]
