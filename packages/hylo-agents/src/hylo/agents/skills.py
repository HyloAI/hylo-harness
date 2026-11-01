"""SkillAgent (RFC-0003 M3): apply verified skills from a SkillLibrary to the agent it wraps, per episode.

At reset the library is matched against the task (instruction pattern, task tags, env). The best verified skill
of each kind is applied for this episode only and undone at its end:
- steering: a partial SteeringConfig merged into the nearest SteeredAgent in the wrapped chain;
- recovery: trigger rules appended to that SteeredAgent's rules (the steering cap of 4 rules still holds);
- plan: subgoal templates, filled with the trigger's slot bindings, given to the nearest SubgoalAgent.
Weights skills are not swapped per episode (a policy reload); they are applied as config by the improvement loop.
Each application is a `skill.applied` trace event, so outcomes can be attributed to the skill.

    agent: {type: skills, params: {skills: {type: skill_library, params: {path: ..., frozen: true}},
            inner: {type: planner.subgoal, params: {inner: {type: lerobot.steered, params: {...}}}}}}
"""

from __future__ import annotations

from typing import Any

from hylo.core.agent import Agent, AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.registry import registry
from hylo.core.types import ActionChunk, Observation, StepResult

from .planner import Plan, Subgoal, SubgoalAgent
from .steering import SteeredAgent, SteeringConfig, TriggerRule


def _find(agent: Any, cls: type) -> Any | None:
    seen = 0
    while agent is not None and seen < 8:
        if isinstance(agent, cls):
            return agent
        agent, seen = getattr(agent, "inner", None), seen + 1
    return None


class TemplatePlanner:
    """One-episode planner: subgoals from a plan skill; completion checks delegate to the original planner."""

    def __init__(self, templates: list[str], binds: dict[str, str], fallback: Any, skill_id: str,
                 max_steps: int = 260) -> None:
        self.templates, self.binds, self.fallback, self.max_steps = templates, binds, fallback, max_steps
        self.name = f"skill.plan[{skill_id}]"

    def plan(self, instruction: str, observation: Observation) -> Plan:
        goals = [t.format(**self.binds) for t in self.templates]
        return Plan(subgoals=[Subgoal(instruction=g, max_steps=self.max_steps) for g in goals],
                    rationale=f"from verified skill {self.name}", meta={"bindings": self.binds})

    def done(self, subgoal: Subgoal, observation: Observation) -> tuple[bool, str, dict[str, Any]]:
        return self.fallback.done(subgoal, observation)


class SkillAgent:
    def __init__(self, inner: Agent | dict[str, Any], skills: Any, kinds: tuple[str, ...] = ("steering", "recovery", "plan")) -> None:
        self.inner = registry.build("agents", inner) if isinstance(inner, dict) else inner
        self.library = registry.build("memory", skills) if isinstance(skills, dict) else skills
        self.kinds = tuple(kinds)
        self._undo: list[tuple[Any, str, Any]] = []

    def spec(self) -> AgentSpec:
        base = self.inner.spec()
        verified = len(self.library.all("verified"))
        return base.model_copy(update={"name": base.name + "+skills", "components": {
            **base.components, "skills": f"library({self.library.snapshot()},verified={verified},kinds={','.join(self.kinds)})"}})

    def _apply(self, ctx: EpisodeContext) -> list[dict[str, Any]]:
        spec = ctx.task.spec
        env_name = getattr(ctx.env_spec, "name", None)
        applied, used_kinds = [], set()
        for skill, binds in self.library.match(spec.instruction, task_tags=tuple(spec.tags), env=env_name, kinds=self.kinds):
            if skill.kind in used_kinds:
                continue                                       # best match per kind only
            sid = f"{skill.name}@{skill.version}"
            if skill.kind in ("steering", "recovery"):
                steered = _find(self.inner, SteeredAgent)
                if steered is None:
                    continue
                cfg = steered.cfg
                if skill.kind == "steering":
                    new = SteeringConfig.model_validate({**cfg.model_dump(), **dict(skill.body.get("steering", {}))})
                else:
                    rules = [*cfg.rules, *(TriggerRule.model_validate(r) for r in skill.body.get("rules", []))][:4]
                    new = cfg.model_copy(update={"rules": rules})
                self._undo.append((steered, "cfg", cfg))
                steered.cfg = new
            elif skill.kind == "plan":
                sub = _find(self.inner, SubgoalAgent)
                if sub is None or not skill.body.get("subgoals"):
                    continue
                try:
                    planner = TemplatePlanner(list(skill.body["subgoals"]), binds, sub.planner, sid)
                    planner.plan(spec.instruction, None)          # fail early on a missing slot
                except (KeyError, IndexError, ValueError):
                    continue
                self._undo.append((sub, "planner", sub.planner))
                sub.planner = planner
            else:
                continue
            used_kinds.add(skill.kind)
            applied.append({"skill": sid, "kind": skill.kind, "bindings": binds, "description": skill.description[:200]})
        return applied

    def reset(self, ctx: EpisodeContext) -> None:
        self._restore()
        self.ctx = ctx
        self.applied = self._apply(ctx)                          # before inner.reset so the episode starts configured
        self.inner.reset(ctx)
        for a in self.applied:
            ctx.emit("skill.applied", a)

    def _restore(self) -> None:
        for obj, attr, value in reversed(self._undo):
            setattr(obj, attr, value)
        self._undo = []

    def act(self, observation: Observation) -> ActionChunk:
        return self.inner.act(observation)

    def on_step(self, result: StepResult) -> None:
        self.inner.on_step(result)

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        self.inner.on_episode_end(outcome)
        self.ctx.emit("x.skills.summary", {"applied": [a["skill"] for a in self.applied], "status": outcome.status})
        self._restore()

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
