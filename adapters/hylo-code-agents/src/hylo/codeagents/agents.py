"""Code-writing agents (RFC-0006 parts 4 and 5).

`code.llm`: a chat model writes one Python block per turn against the toolkit (`robot`), in a persistent restricted
namespace; it sees what the block printed, errors, the robot's status and images it asked for, never the task's
score. It ends the episode with finish() or when a budget runs out.

    agent: {type: code.llm, params: {model: {backend: ollama, model: qwen3.5:35b-a3b, vision: true},
            max_turns: 30, max_cost_usd: 1.0, block_timeout_s: 60,
            skills: {type: skill_library, params: {path: runs/skills/manip, frozen: true}}}}

`code.program`: runs a frozen solve(robot) (from a sealed program artifact, or source in params for experiments)
with the same toolkit and no model: free to run and reproducible.

    agent: {type: code.program, params: {artifact: artifact:pick-place-sub2@1a2b3c4d}}

Both drive the robot through EpisodeBridge, so every motion is an ActionChunk the kernel executes step by step
through the SafetyShield. `tools` names the toolkit factory (default hylo.manip.tools:Tools; tests use
hylo.codeagents.fakes:FakeTools); it is called as factory(robot_spec, executor, control_hz=..., show=..., **params).
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from typing import Any, Callable, Mapping

from hylo.core.agent import AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.errors import AgentError
from hylo.core.registry import registry
from hylo.core.types import ActionChunk, EnvSpec, Observation, StepResult
from hylo.reasoners.llm.chat import ChatModel, build_chat, describe

from .bridge import EpisodeBridge, TimedExecutor, code_agent_spec
from .prompt import api_reference, code_llm_system, robot_description, scene_preview, task_message
from .sandbox import CodeRejected, Sandbox, defines, validate
from .skills import load_program_skills, read_program
from .turns import Budget, CodeLoop

DEFAULT_TOOLS = "hylo.manip.tools:Tools"


def load_ref(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def tools_ref(tools: str | Mapping[str, Any] | None) -> str:
    if tools is None:
        return DEFAULT_TOOLS
    return tools if isinstance(tools, str) else str(tools.get("type", DEFAULT_TOOLS))


def tool_status(tools: Any, fallback: Any) -> dict[str, Any]:
    """The robot's status for feedback; with the reason the last plan() failed when the toolkit says (robot.why())."""
    try:
        st = dict(tools.status())
    except NotImplementedError:
        st = dict(fallback.status())
    why = getattr(tools, "why", None)
    if callable(why):
        try:
            reason = why()
        except Exception:  # noqa: BLE001
            reason = ""
        if reason:
            st["last_plan_failure"] = str(reason)[:500]
    return st


def make_tools(tools: str | Mapping[str, Any] | None, env_spec: EnvSpec, executor: Any,
               show: Callable[[Any, str], None] | None) -> Any:
    if env_spec.robot is None:
        raise AgentError(f"{env_spec.name} publishes no RobotSpec (EnvSpec.robot): code agents need one")
    params = dict(tools.get("params") or {}) if isinstance(tools, Mapping) else {}
    return load_ref(tools_ref(tools))(env_spec.robot, executor, control_hz=env_spec.control_hz, show=show, **params)


class _CodeAgentBase:
    """Episode plumbing shared by both agents: a bridge per episode and one program thread."""

    name = "code"
    join_s = 30.0

    def __init__(self, tools: str | Mapping[str, Any] | None = None, act_wait_s: float = 1800.0) -> None:
        self.tools, self.act_wait_s = tools, act_wait_s
        self.bridge: EpisodeBridge | None = None
        self.sandbox: Sandbox | None = None
        self._thread: threading.Thread | None = None
        self._emit_open = False

    def reset(self, ctx: EpisodeContext) -> None:
        self._stop()
        self.ctx = ctx
        if ctx.env_spec.robot is None:
            raise AgentError(f"{ctx.env_spec.name} publishes no RobotSpec (EnvSpec.robot): code agents need one")
        self.bridge = EpisodeBridge(ctx.env_spec, ctx.task, act_wait_s=self.act_wait_s)
        self.sandbox, self._thread, self._emit_open = None, None, True

    def emit(self, type_: str, payload: Mapping[str, Any]) -> None:
        if self._emit_open:                              # late events after the episode closed are dropped
            self.ctx.emit(type_, payload)

    def act(self, observation: Observation) -> ActionChunk:
        assert self.bridge is not None, "reset() first"
        if self._thread is None:                         # the program starts with the first observation
            bridge = self.bridge
            self._thread = threading.Thread(target=self._main, args=(bridge,), daemon=True, name=f"hylo-{self.name}")
            self._thread.start()
        return self.bridge.act(observation)

    def on_step(self, result: StepResult) -> None:
        if self.bridge is not None:
            self.bridge.on_step(result)

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        if self.bridge is not None:
            self.bridge.on_episode_end(outcome)
        self._join()
        self._emit_open = False

    def _join(self) -> None:
        t = self._thread
        if t is None:
            return
        t.join(timeout=self.join_s)
        if t.is_alive() and self.sandbox is not None:   # still computing after the end: stop the block
            self.sandbox.interrupt()
            t.join(timeout=5.0)

    def _stop(self) -> None:
        if self.bridge is not None:
            self.bridge.cancel()
        if self._thread is not None and self._thread.is_alive():
            if self.sandbox is not None:
                self.sandbox.interrupt()
            self._thread.join(timeout=5.0)
        self._emit_open = False

    def close(self) -> None:
        self._stop()

    def _main(self, bridge: EpisodeBridge) -> None:
        reason = "program ended"
        try:
            reason = self._program(bridge) or reason
        except Exception as exc:  # noqa: BLE001 - a broken program ends its episode, never the run
            reason = f"program error: {exc!r}"[:300]
            self.emit("agent.code.error", {"error": reason})
        finally:
            bridge.finish(reason)

    def _program(self, bridge: EpisodeBridge) -> str | None:
        raise NotImplementedError


class CodeLLMAgent(_CodeAgentBase):
    name = "code.llm"

    def __init__(self, model: str | Mapping[str, Any] | ChatModel = "ollama:qwen3.5:35b-a3b",
                 tools: str | Mapping[str, Any] | None = None, max_turns: int = 30, max_tokens: int | None = None,
                 max_cost_usd: float | None = None, max_steps: int | None = None, block_timeout_s: float = 60.0,
                 reply_tokens: int | None = None, act_wait_s: float = 21600.0, demonstration: str | None = None,
                 skills: str | Mapping[str, Any] | Any | None = None, max_skills: int = 3,
                 min_skill_similarity: float = 0.2, max_history_turns: int = 24,
                 max_context_chars: int | None = 80_000) -> None:
        super().__init__(tools, act_wait_s)
        self.model = build_chat(model)
        self.budget = Budget(turns=max_turns, tokens=max_tokens, cost_usd=max_cost_usd, steps=max_steps)
        self.block_timeout_s, self.reply_tokens, self.max_history_turns = block_timeout_s, reply_tokens, max_history_turns
        self.max_context_chars = max_context_chars
        self.demonstration = self._demo(demonstration)
        self.library = registry.build("memory", {"type": "skill_library", "params": {"path": skills, "frozen": True}}) \
            if isinstance(skills, str) else registry.build("memory", dict(skills)) if isinstance(skills, Mapping) else skills
        self.max_skills, self.min_skill_similarity = max_skills, min_skill_similarity
        self.join_s = block_timeout_s + 15.0
        self.loop: CodeLoop | None = None

    @staticmethod
    def _demo(demo: str | None) -> str | None:
        """A demonstration program: source text, or a path inside a sealed program artifact."""
        if not demo or "\n" in demo or "def " in demo:
            return demo
        return read_program(demo)[0]

    def spec(self) -> AgentSpec:
        comps = {"model": describe(self.model), "tools": tools_ref(self.tools),
                 "budget": self.budget.model_dump_json(exclude_none=True)}
        if self.library is not None:
            comps["skills"] = f"library({self.library.snapshot()})"
        if self.demonstration:
            comps["demonstration"] = "sha256:" + hashlib.sha256(self.demonstration.encode()).hexdigest()[:16]
        return code_agent_spec(self.name, comps)

    def _remaining(self, loop: CodeLoop, bridge: EpisodeBridge) -> dict[str, Any]:
        b = self.budget
        left: dict[str, Any] = {"turns": b.turns - loop.turn,
                                "robot_steps": bridge.task.spec.max_steps - bridge.step}
        if b.steps is not None:
            left["robot_steps"] = min(left["robot_steps"], b.steps - bridge.step)
        if b.tokens is not None:
            left["model_tokens"] = b.tokens - loop.usage.tokens
        if b.cost_usd is not None:
            left["cost_usd"] = round(b.cost_usd - loop.usage.cost_usd, 4)
        return left

    def _program(self, bridge: EpisodeBridge) -> str | None:
        ctx = self.ctx
        holder: dict[str, CodeLoop] = {}
        state = {"finished": False}

        def show(image: Any, note: str = "") -> None:
            holder["loop"].show(image, note)

        def finish(reason: str = "") -> dict[str, Any]:
            state["finished"] = True
            out = bridge.finish(str(reason)[:200] or "finish() called")
            return {"ended": True, "steps": out.get("steps")}         # the outcome itself is not shown

        sandbox = self.sandbox = Sandbox({}, block_timeout_s=self.block_timeout_s)
        tools = make_tools(self.tools, ctx.env_spec, TimedExecutor(bridge, sandbox), show)
        skills_ns, skills_text, used = None, "", []
        obs = bridge.observation()
        instruction = obs.instruction or ctx.task.spec.instruction     # the episode's own words, not the template
        if self.library is not None:
            skills_ns, skills_text, used = load_program_skills(
                self.library, instruction, lambda src, entry: sandbox.define(src, entry),
                k=self.max_skills, min_score=self.min_skill_similarity)
            self.emit("agent.code.skills", {"retrieved": used})
        names = {"robot": tools, "finish": finish}
        if skills_ns is not None and vars(skills_ns):
            names["skills"] = skills_ns
        sandbox.add(**names)                            # skill programs see the same robot and finish
        system = code_llm_system(robot=robot_description(ctx.env_spec.robot, ctx.env_spec), api=api_reference(tools),
                                 block_s=self.block_timeout_s, vision=bool(getattr(self.model, "vision", False)),
                                 skills=skills_text, demonstration=self.demonstration)

        def status() -> dict[str, Any]:
            return tool_status(tools, bridge)

        def remaining() -> dict[str, Any]:
            return self._remaining(holder["loop"], bridge)

        holder["loop"] = loop = self.loop = CodeLoop(
            self.model, sandbox, system=system, first="", emit=lambda rec: self.emit("agent.code.turn", rec),
            status=status, reply_tokens=self.reply_tokens, max_history_turns=self.max_history_turns,
            max_context_chars=self.max_context_chars)
        scene, images = scene_preview(tools, loop.vision)
        if skills_text:                                  # next to the task too: small models skim a long system prompt
            names_ = ", ".join(f"skills.{u['skill'].split('@')[0]}(robot)" for u in used if "skipped" not in u)
            scene = (scene + "\n" if scene else "") + (f"Verified skills that may fit this instruction: {names_} "
                                                         "(see the system message); try the fitting one first.")
        loop.messages[1]["content"] = task_message(instruction, status(), remaining(), scene)
        if images:
            loop.messages[1]["images"] = images
        reason = None
        while not bridge.ended and not state["finished"]:
            reason = loop.exceeded(self.budget, bridge.step) or loop.done
            if reason:
                break
            bridge.provenance = {"source": self.name, "turn": loop.turn + 1}
            loop.step(remaining(), extra={"step": bridge.step})
        self.emit("agent.code.summary", {"turns": loop.turn, "usage": loop.usage.as_dict(), "finished": state["finished"],
                                         "stopped": reason or ("finish() called" if state["finished"] else "episode ended"),
                                         "steps": bridge.step})
        return reason


class CodeProgramAgent(_CodeAgentBase):
    name = "code.program"

    def __init__(self, artifact: str | None = None, source: str | None = None, entry: str = "solve",
                 tools: str | Mapping[str, Any] | None = None, timeout_s: float = 300.0, act_wait_s: float = 600.0,
                 kwargs: Mapping[str, Any] | None = None, record_output: int = 4000) -> None:
        super().__init__(tools, act_wait_s)
        if (artifact is None) == (source is None):
            raise ValueError("code.program needs exactly one of `artifact` (a sealed program) or `source`")
        if artifact is not None:
            self.source, m, meta = read_program(artifact)
            self.digest, entry = m.digest, meta.get("entry", entry)
        else:
            self.source = str(source)
            self.digest = "sha256:" + hashlib.sha256(self.source.encode()).hexdigest()[:16]
        tree = validate(self.source)
        if not defines(tree, entry):
            raise CodeRejected(f"the program defines no top-level {entry}(robot)")
        self.entry, self.timeout_s, self.kwargs, self.record_output = entry, timeout_s, dict(kwargs or {}), record_output
        self.join_s = 30.0
        self.last: dict[str, Any] = {}

    def spec(self) -> AgentSpec:
        return code_agent_spec(self.name, {"program": self.digest, "tools": tools_ref(self.tools)})

    def _program(self, bridge: EpisodeBridge) -> str | None:
        state = {"finished": ""}

        def finish(reason: str = "") -> dict[str, Any]:
            state["finished"] = str(reason)[:200] or "finish() called"
            out = bridge.finish(state["finished"])
            return {"ended": True, "steps": out.get("steps")}

        sandbox = self.sandbox = Sandbox({}, block_timeout_s=self.timeout_s, max_output=self.record_output)
        tools = make_tools(self.tools, self.ctx.env_spec, TimedExecutor(bridge, sandbox), None)
        sandbox.add(robot=tools, finish=finish)
        res = sandbox.call(lambda: sandbox.define(self.source, self.entry, fresh=False)(tools, **self.kwargs))
        self.last = {"output": res.output, "error": res.error, "seconds": res.seconds, "timed_out": res.timed_out,
                     "steps": bridge.step}
        self.emit("agent.code.program", {**self.last, "program": self.digest})
        if res.error:
            return f"program error: {res.error.splitlines()[-1][:200]}"
        return state["finished"] or f"{self.entry} returned"


__all__ = ["CodeLLMAgent", "CodeProgramAgent", "DEFAULT_TOOLS", "load_ref", "make_tools", "tool_status", "tools_ref"]
