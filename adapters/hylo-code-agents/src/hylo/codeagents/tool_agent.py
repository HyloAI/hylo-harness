"""`tools.llm` (RFC-0007): a chat model operates the robot through typed tools with native tool calls.

    agent: {type: tools.llm, params: {model: {backend: ollama, model: qwen3.5:35b-a3b, vision: true},
            max_decisions: 40, max_calls: 150, inbox: runs/steer/inbox.jsonl}}

Each model call sends the projected conversation (tool_context) and every tool of the surface (robot_surface plus
read_artifact). Read calls of one reply run in order; an act call must be alone in its reply, runs through the
kernel and the SafetyShield, and the next message carries its receipt and fresh images. `finish` ends the episode;
the task's success check stays the verdict and is never shown.

Budgets: decisions (act calls that moved or could have moved the robot; refusals that moved nothing do not count),
model calls, calls without an action, tokens, cost, wall time. Replies without a tool call get a reminder; after
`max_idle_replies` in a row the episode ends.

Steering: with `inbox`, a JSON-lines file a person (the Studio, `hylo watch --say`) appends {"message": ...} to,
each new message is shown to the model before its next call as operator guidance. A run that received any is
marked `human_assisted` in its summary event and must not be counted as autonomous.

Trace events: agent.tool.start (surface and prompt digests), agent.tool.turn (per model call: projection stats,
reply, tool calls, usage), agent.tool.result (per call), agent.tool.proposal / agent.tool.receipt (per act),
agent.tool.operator, agent.tool.finish, agent.tool.summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from hylo.core.tooling import ToolError, ToolReply
from hylo.reasoners.llm.chat import ChatError, ChatModel, Usage, build_chat, describe, with_tools

from .agents import _CodeAgentBase, make_tools, tools_ref
from .bridge import EpisodeBridge, code_agent_spec
from .robot_surface import RobotSurface
from .tool_context import ToolContext, estimate_tokens
from .tool_prompt import bundle, task_message


class Inbox:
    """Operator messages: a JSON-lines file; `poll()` returns the messages appended since the last poll (those
    present when the inbox was opened are skipped: they were meant for an earlier run)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.is_file() else 0

    def poll(self) -> list[str]:
        if not self.path.is_file():
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1                       # only complete lines
        self.offset += end
        out = []
        for line in data[:end].decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                text = msg.get("message") if isinstance(msg, dict) else str(msg)
            except json.JSONDecodeError:
                text = line
            if text:
                out.append(str(text)[:2000])
        return out


def post(path: str | os.PathLike[str], message: str, sender: str = "operator") -> None:
    """Append one operator message to an inbox file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"message": message, "from": sender, "time": round(time.time(), 3)}) + "\n")


class ToolLLMAgent(_CodeAgentBase):
    name = "tools.llm"

    def __init__(self, model: str | Mapping[str, Any] | ChatModel = "ollama:qwen3.5:35b-a3b",
                 tools: str | Mapping[str, Any] | None = None, max_decisions: int = 40, max_calls: int = 150,
                 calls_per_decision: int = 25, max_tokens: int | None = None, max_cost_usd: float | None = None,
                 wall_s: float | None = None, reply_tokens: int = 4096, max_idle_replies: int = 4,
                 inbox: str | None = None, contract: str | None = None, overlay_camera: str | None = None,
                 observe_cameras: list[str] | None = None, context: Mapping[str, Any] | None = None,
                 extra_prompt: str = "", act_wait_s: float = 21600.0) -> None:
        super().__init__(tools, act_wait_s)
        self.model = with_tools(build_chat(model))
        self.max_decisions, self.max_calls, self.calls_per_decision = max_decisions, max_calls, calls_per_decision
        self.max_tokens, self.max_cost_usd, self.wall_s = max_tokens, max_cost_usd, wall_s
        self.reply_tokens, self.max_idle_replies = reply_tokens, max_idle_replies
        self.inbox_path, self.contract, self.overlay_camera = inbox, contract, overlay_camera
        self.observe_cameras = observe_cameras
        self.context_params = dict(context or {})
        self.extra_prompt = extra_prompt
        self.join_s = 60.0
        self.last: dict[str, Any] = {}

    def spec(self):
        budget = {"decisions": self.max_decisions, "calls": self.max_calls, "tokens": self.max_tokens,
                  "cost_usd": self.max_cost_usd, "wall_s": self.wall_s}
        comps = {"model": describe(self.model), "tools": tools_ref(self.tools),
                 "budget": json.dumps({k: v for k, v in budget.items() if v is not None}, sort_keys=True),
                 "context": json.dumps(self.context_params, sort_keys=True)}
        if self.contract:
            comps["contract"] = "sha256:" + hashlib.sha256(self.contract.encode()).hexdigest()[:16]
        if self.extra_prompt:
            comps["extra_prompt"] = "sha256:" + hashlib.sha256(self.extra_prompt.encode()).hexdigest()[:16]
        return code_agent_spec(self.name, comps)

    # ------------------------------------------------------------------ the episode
    def _observation(self, rs: RobotSurface, title: str) -> dict[str, Any]:
        view, _ = rs.view()
        names = [c for c in (self.observe_cameras or list(view.cams)) if c in view.cams]
        state = rs.robot_state({})
        msg: dict[str, Any] = {"role": "user", "content": f"{title} (step {view.step}). Images: {', '.join(names)}. "
                                                         f"Robot: {json.dumps(_compact_state(state))}"}
        if getattr(self.model, "vision", False):
            msg["images"] = [view.cams[n].rgb for n in names]
        return msg

    def _program(self, bridge: EpisodeBridge) -> str | None:
        ctx_env = self.ctx.env_spec
        state = {"outcome": None}

        def finish(outcome: str, reason: str) -> None:
            state["outcome"] = outcome
            bridge.finish(f"finish({outcome}): {reason}")

        tools = make_tools(self.tools, ctx_env, bridge, None)
        rs = RobotSurface(tools, emit=self.emit, finish=finish, overlay_camera=self.overlay_camera)
        system, digests = bundle(ctx_env.robot, ctx_env.name, ctx_env.control_hz, self.extra_prompt)
        usage = Usage()

        def summarize(msgs: list[dict[str, Any]]) -> str:
            r = self.model.chat(msgs, max_tokens=1500)
            usage.add(r)
            self.emit("agent.tool.summary_call", {**r.accounting(), "text": r.text[:4000]})
            return r.text

        obs = bridge.observation()
        instruction = obs.instruction or self.ctx.task.spec.instruction
        first = self._observation(rs, "OPENING OBSERVATION")
        task = task_message(instruction, self.contract) + "\n\n" + first["content"]
        cp = {"tail_decisions": 3, "image_decisions": 1, "inline_chars": 1500, "compact_tokens": 48_000,
              **self.context_params}
        ctx = ToolContext(system, task, first.get("images"), summarize=summarize, **cp)
        surface = rs.surface
        surface.add(ctx.tool())
        specs = surface.specs()
        inbox = Inbox(self.inbox_path) if self.inbox_path else None
        self.emit("agent.tool.start", {"model": describe(self.model), "surface": surface.digest(),
                                       "tools": surface.names(), "prompt": digests, "instruction": instruction,
                                       "system": system, "task": task, "context": cp})
        t0 = time.monotonic()
        calls = idle = since_act = operator = 0
        last_failure: tuple[str, str] | None = None
        refused_key, refused_n = None, 0
        errors = 0
        stop: str | None = None
        while not bridge.ended and state["outcome"] is None and not stop:
            stop = self._exceeded(rs, usage, calls, since_act, t0)
            if stop:
                break
            for text in inbox.poll() if inbox else []:
                operator += 1
                ctx.add({"role": "user", "content": f"OPERATOR MESSAGE (a person supervising this run): {text}"},
                        operator=True)
                self.emit("agent.tool.operator", {"message": text, "step": bridge.step})
            msgs = ctx.project()
            msgs.append({"role": "user", "content": "Left: " + json.dumps(self._left(rs, usage, calls, t0))})
            digest = hashlib.sha256(json.dumps([{k: v for k, v in m.items() if k != "images"} for m in msgs],
                                               default=str, sort_keys=True).encode()).hexdigest()[:16]
            try:
                reply = self.model.chat(msgs, tools=specs, max_tokens=self.reply_tokens)
            except ChatError as exc:
                errors += 1
                calls += 1
                self.emit("agent.tool.turn", {"call": calls, "request": digest, "model_error": str(exc)[:500]})
                if errors >= 3:
                    stop = f"model_error: {str(exc)[:200]}"
                    break
                continue
            errors = 0
            calls += 1
            since_act += 1
            usage.add(reply)
            ctx.add(reply.message())
            self.emit("agent.tool.turn", {
                "call": calls, "request": digest, "messages": len(msgs),
                "images": sum(len(m.get("images") or []) for m in msgs),
                "est_tokens": estimate_tokens(msgs), "text": reply.text[:4000],
                "reasoning": (reply.reasoning or "")[-4000:] or None,
                "tool_calls": [c.wire() for c in reply.tool_calls], **reply.accounting(),
                "stop_reason": reply.stop_reason, "usage": usage.as_dict(), "step": bridge.step,
                "decisions": rs.decisions})
            if not reply.tool_calls:
                idle += 1
                cut = reply.stop_reason in ("length", "max_tokens")
                ctx.add({"role": "user", "content": (
                    "Your reply was cut off at the reply limit; keep the text short and call a tool."
                    if cut else "Your reply had no tool call. Continue the task by calling a tool (or finish).")})
                if idle >= self.max_idle_replies:
                    stop = f"{idle} replies in a row without a tool call"
                    break
                continue
            idle = 0
            n = len(reply.tool_calls)
            acted = ended = False
            for c in reply.tool_calls:
                kind = surface.kind(c.name)
                if c.error:
                    res = ToolReply(c.name, False, error=ToolError("bad_json", c.error).as_dict())
                elif kind in ("act", "end") and n > 1:
                    res = ToolReply(c.name, False, error=ToolError(
                        "act_not_alone", f"{c.name} changes the world and must be the only call in its reply; it was "
                                         f"one of {n}. The read calls ran; send {c.name} again on its own.").as_dict())
                else:
                    res = surface.call(c.name, c.arguments)
                    if kind == "act" and res.ok and (res.result or {}).get("counted_as_decision", True):
                        acted = True                       # refusals that moved nothing are not an action
                    if kind == "end" and res.ok:
                        ended = True
                key = (c.name, json.dumps(c.arguments, sort_keys=True, default=str))
                payload = res.payload()
                if kind == "act" and res.ok and (res.result or {}).get("status") == "refused":
                    refused_n = refused_n + 1 if refused_key == key else 1
                    refused_key = key
                    if refused_n >= 2:
                        payload["repeated"] = (f"this exact action was refused {refused_n} times in a row: change the "
                                               "target, measure again, or try another approach")
                    if refused_n >= 3:
                        stop = f"the same action was refused {refused_n} times in a row"
                elif kind == "act":
                    refused_key, refused_n = None, 0
                if not res.ok:
                    code = (res.error or {}).get("code", "")
                    if last_failure == (key[0] + key[1], code):
                        payload["repeated"] = ("this exact call failed the same way last time: change the arguments "
                                               "or the approach")
                    last_failure = (key[0] + key[1], code)
                text = json.dumps(payload, separators=(",", ":"), default=str)
                tm: dict[str, Any] = {"role": "tool", "tool_call_id": c.id, "name": c.name, "content": text}
                if res.images and getattr(self.model, "vision", False):
                    tm["images"] = res.images
                ctx.add(tm)
                self.emit("agent.tool.result", {"call": calls, "id": c.id, "name": c.name, "kind": kind,
                                                "arguments": c.arguments, "ok": res.ok, "result": text[:6000],
                                                "images": [{"note": nt, "image": im} for nt, im in
                                                           zip(res.image_notes or [""] * len(res.images), res.images)],
                                                "step": bridge.step})
            if ended or bridge.ended or stop:
                break
            if acted:
                since_act = 0
                ctx.acted()
                ctx.add(self._observation(rs, "OBSERVATION after your action"))
        self.last = {"calls": calls, "decisions": rs.decisions, "refused": rs.refused, "usage": usage.as_dict(),
                     "outcome": rs.outcome, "stopped": stop or ("finish" if state["outcome"] else "episode ended"),
                     "operator_messages": operator, "human_assisted": operator > 0, "steps": bridge.step,
                     "context": dict(ctx.stats), "issues": rs.issues, "wall_s": round(time.monotonic() - t0, 1)}
        self.emit("agent.tool.summary", self.last)
        return stop

    def _left(self, rs: RobotSurface, usage: Usage, calls: int, t0: float) -> dict[str, Any]:
        left: dict[str, Any] = {"decisions": self.max_decisions - rs.decisions, "model_calls": self.max_calls - calls}
        st = rs.t.executor.status()
        left["robot_steps"] = int(st.get("max_steps") or 0) - int(st.get("step") or 0)
        if self.wall_s:
            left["seconds"] = int(self.wall_s - (time.monotonic() - t0))
        if self.max_cost_usd is not None:
            left["cost_usd"] = round(self.max_cost_usd - usage.cost_usd, 4)
        return left

    def _exceeded(self, rs: RobotSurface, usage: Usage, calls: int, since_act: int, t0: float) -> str | None:
        if rs.decisions >= self.max_decisions:
            return f"budget: {rs.decisions} decisions"
        if calls >= self.max_calls:
            return f"budget: {calls} model calls"
        if since_act >= self.calls_per_decision:
            return f"budget: {since_act} model calls without an action"
        if self.max_tokens is not None and usage.tokens >= self.max_tokens:
            return f"budget: {usage.tokens} tokens"
        if self.max_cost_usd is not None and usage.cost_usd >= self.max_cost_usd:
            return f"budget: ${usage.cost_usd:.4f}"
        if self.wall_s is not None and time.monotonic() - t0 >= self.wall_s:
            return f"budget: {self.wall_s:.0f} s wall time"
        return None


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"steps_left": state.get("steps_left")}
    for arm, a in (state.get("arms") or {}).items():
        out[arm] = {"tcp_position": a["tcp"]["position"], "tcp_quat_xyzw": a["tcp"]["quat_xyzw"],
                    "hand_opening": a["hand"]["opening"], "holding": a["hand"]["holding"]}
    from hylo.core.tooling import jsonable

    return jsonable(out)


__all__ = ["Inbox", "ToolLLMAgent", "post"]
