"""Tool sessions (RFC-0008 part 7): a chat model works with any registered tool surface (group `hylo.surfaces`) on
a request, outside an episode: build a scene, run robots in it, record data.

    hylo world chat "put a Go1 on the floor next to a table with a red cube and walk it 1 m forward" \\
        [--stage lab] [--root runs/stages] [--model ollama:qwen3.5:35b-a3b] [--vision] [--out runs/world-chat]
    hylo world chat                  (no request: read requests from the terminal, one session that remembers)
    hylo world tools                 the tool manual of the surface

The loop is the one of `tools.llm` without an environment: every model call sends the projected conversation
(tool_context) and every tool; read calls of a reply run first, then act calls in order (each tool checks its own
preconditions: stage revisions, world generations); after any act the `after_act` read tool (the world's free
view) is called so a vision model sees the result. Events go to a hash-chained trace (`hylo watch OUT` follows it):
agent.tool.start / turn / result / operator / summary. An inbox file lets a person steer the session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hylo.core.events import Event
from hylo.core.registry import registry
from hylo.core.tooling import Surface, ToolError, ToolReply
from hylo.core.trace import TraceStore
from hylo.reasoners.llm.chat import ChatError, ChatModel, Usage, build_chat, describe, with_tools

from .tool_agent import Inbox
from .tool_context import ToolContext, estimate_tokens

WORLD_SYSTEM = """You work in a 3D robotics workspace through tools: a stage (the scene as entities, edited by
patches with revisions) and a MuJoCo world built from it (robots from packs, cameras, people), which you run with
actions and check with predicates.

How to work:
- Read before you write: stage_show gives the revision stage_edit needs; world_state gives the generation every
  robot action needs. A conflict or a stale refusal means: read again, then retry.
- Edits change the stage, not the running world: call world_build after editing, then act.
- Build scenes in metres, +z up, the floor at z = 0. A box's size is its half extents. A table top at height h with
  half thickness t has its centre at z = h - t. Put objects on surfaces (centre z = surface + half height).
- Robots come from packs (packs tool): check a pack's controls before choosing actions. Arms move their tool point
  with tcp actions; grasp from above: move above the object, descend, close the gripper, lift.
- Look at the result (view or capture) and check goals with check: a receipt says what the robot did, a check says
  whether the goal holds. Report what you measured.
- Do one thing at a time and keep replies short; finish with a summary of what exists and what was verified."""


class ToolSession:
    def __init__(self, model: ChatModel, surface: Surface, *, system: str = WORLD_SYSTEM, out: str | Path = "runs/session",
                 inbox: str | None = None, after_act: str | None = "view", max_calls: int = 60,
                 reply_tokens: int = 4096, max_cost_usd: float | None = None,
                 context: Mapping[str, Any] | None = None, emit: Callable[[str, Mapping[str, Any]], None] | None = None,
                 printer: Callable[[str], None] | None = None) -> None:
        self.model, self.surface = with_tools(model), surface
        self.after_act = after_act if after_act in surface else None
        self.max_calls, self.reply_tokens, self.max_cost_usd = max_calls, reply_tokens, max_cost_usd
        self.usage = Usage()
        self.calls = 0
        self.inbox = Inbox(inbox) if inbox else None
        self.id = "session-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.store = TraceStore(out)
        self.seq = 0
        self.print = printer or (lambda s: None)
        self._emit_extra = emit
        self.ctx = ToolContext(system, "", summarize=self._summarize,
                               **{"tail_decisions": 4, "image_decisions": 1, "inline_chars": 2500,
                                  "compact_tokens": 60_000, **dict(context or {})})
        self.ctx.head = self.ctx.head[:1]               # requests are added as they come
        surface.add(self.ctx.tool()) if "read_artifact" not in surface else None
        self.specs = surface.specs()
        self.emit("agent.tool.start", {"model": describe(self.model), "surface": surface.digest(),
                                       "tools": surface.names(), "system": system,
                                       "prompt": {"bundle": hashlib.sha256(system.encode()).hexdigest()[:16]}})

    def emit(self, type_: str, payload: Mapping[str, Any]) -> None:
        self.seq += 1
        self.store.write(Event(type=type_, episode_id=self.id, seq=self.seq, source="session", payload=dict(payload)))
        if self._emit_extra is not None:
            self._emit_extra(type_, payload)

    def _summarize(self, msgs: list[dict[str, Any]]) -> str:
        r = self.model.chat(msgs, max_tokens=1500)
        self.usage.add(r)
        return r.text

    def ask(self, request: str, max_calls: int | None = None) -> str:
        """Work on one request until the model answers without tool calls (or a budget runs out); its last words."""
        self.ctx.add({"role": "user", "content": request})
        self.emit("agent.tool.operator", {"message": request, "kind": "request"})
        limit = self.calls + (max_calls or self.max_calls)
        answer, errors = "", 0
        while self.calls < limit:
            if self.max_cost_usd is not None and self.usage.cost_usd >= self.max_cost_usd:
                answer = f"(stopped: ${self.usage.cost_usd:.3f} spent)"
                break
            for text in self.inbox.poll() if self.inbox else []:
                self.ctx.add({"role": "user", "content": f"OPERATOR MESSAGE: {text}"})
                self.emit("agent.tool.operator", {"message": text})
            msgs = self.ctx.project()
            try:
                reply = self.model.chat(msgs, tools=self.specs, max_tokens=self.reply_tokens)
            except ChatError as exc:
                errors += 1
                self.calls += 1
                self.emit("agent.tool.turn", {"call": self.calls, "model_error": str(exc)[:500]})
                if errors >= 3:
                    answer = f"(model error: {exc})"
                    break
                continue
            errors = 0
            self.calls += 1
            self.usage.add(reply)
            self.ctx.add(reply.message())
            self.emit("agent.tool.turn", {"call": self.calls, "messages": len(msgs), "est_tokens": estimate_tokens(msgs),
                                          "text": reply.text[:4000], "tool_calls": [c.wire() for c in reply.tool_calls],
                                          **reply.accounting(), "stop_reason": reply.stop_reason,
                                          "usage": self.usage.as_dict()})
            if reply.text.strip():
                self.print(f"◆ {reply.text.strip()[:600]}")
            if not reply.tool_calls:
                answer = reply.text
                break
            order = sorted(reply.tool_calls, key=lambda c: self.surface.kind(c.name) != "read")   # reads first
            acted = False
            for c in order:
                kind = self.surface.kind(c.name)
                if c.error:
                    res = ToolReply(c.name, False, error=ToolError("bad_json", c.error).as_dict())
                else:
                    res = self.surface.call(c.name, c.arguments)
                acted |= kind == "act" and res.ok
                text = res.text()
                msg: dict[str, Any] = {"role": "tool", "tool_call_id": c.id, "name": c.name, "content": text}
                if res.images and getattr(self.model, "vision", False):
                    msg["images"] = res.images
                self.ctx.add(msg)
                self.emit("agent.tool.result", {"call": self.calls, "id": c.id, "name": c.name, "kind": kind,
                                                "arguments": c.arguments, "ok": res.ok, "result": text[:6000],
                                                "images": [{"note": n, "image": im} for n, im in
                                                           zip(res.image_notes or [""] * len(res.images), res.images)]})
                self.print(f"  {'▶' if kind == 'act' else '·'} {c.name} -> {'ok' if res.ok else 'ERROR'} "
                           f"{text[:160]}")
            if acted:
                self.ctx.acted()
                if self.after_act:
                    v = self.surface.call(self.after_act, {})
                    if v.ok and v.images:
                        self.emit("agent.tool.view", {"call": self.calls, "image": v.images[0]})
                        if getattr(self.model, "vision", False):
                            self.ctx.add({"role": "user", "content": "The world after your last actions (free view).",
                                          "images": v.images})
        self.emit("agent.tool.summary", {"calls": self.calls, "usage": self.usage.as_dict(), "answer": answer[:4000],
                                         "context": dict(self.ctx.stats)})
        return answer

    def close(self) -> None:
        self.store.close()


def _surface(name: str, params: Mapping[str, Any]) -> Surface:
    return registry.build("surfaces", {"type": name, "params": dict(params)})


def _model(spec: str, vision: bool) -> ChatModel:
    backend, _, name = spec.partition(":")
    params: dict[str, Any] = {"backend": backend, **({"model": name} if name else {})}
    if backend in ("ollama", "openai", "anthropic"):
        params.update(vision=vision, timeout_s=900)
    return build_chat(params)


def main(argv: Sequence[str] | None = None) -> int:
    """a chat model builds and runs a robot world (chat), or list the world tools (tools)"""
    ap = argparse.ArgumentParser(prog="hylo world", description="a chat model builds and runs a robot world")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "tools"):
        p = sub.add_parser(name)
        p.add_argument("--surface", default="mujoco.world")
        p.add_argument("--stage", default="lab")
        p.add_argument("--root", default="runs/stages")
        if name == "chat":
            p.add_argument("request", nargs="?")
            p.add_argument("--model", default="ollama:qwen3.5:35b-a3b")
            p.add_argument("--vision", action="store_true")
            p.add_argument("--out", default="runs/world-chat")
            p.add_argument("--inbox")
            p.add_argument("--max-calls", type=int, default=60)
            p.add_argument("--max-cost", type=float)
    a = ap.parse_args(argv)
    surface = _surface(a.surface, {"root": a.root, "stage": a.stage})
    if a.cmd == "tools":
        print(surface.manual())
        return 0
    s = ToolSession(_model(a.model, a.vision), surface, out=a.out, inbox=a.inbox, max_calls=a.max_calls,
                    max_cost_usd=a.max_cost, printer=print)
    print(f"session {s.id}: trace in {a.out} (hylo watch {a.out})", file=sys.stderr)
    try:
        if a.request:
            print(s.ask(a.request))
        else:
            for line in sys.stdin:
                if line.strip():
                    print(s.ask(line.strip()))
    finally:
        s.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["ToolSession", "WORLD_SYSTEM", "main"]
