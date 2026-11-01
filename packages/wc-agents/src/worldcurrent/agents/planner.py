"""Subgoal planning around any Agent (RFC-0002 L2).

SubgoalAgent asks a Planner to decompose the task instruction into short sequential sub-instructions, feeds the
current one to the wrapped agent (typically a VLA) in place of the original instruction, and advances when the
planner judges it done or its step budget runs out. Plans and switches are trace events, so failures can be
attributed to planning vs execution. The planner's prompt and check cadence are config, i.e. evolvable.
"""

from __future__ import annotations

from typing import Any


from worldcurrent.core.agent import Agent, AgentSpec, EpisodeContext, EpisodeOutcome
from worldcurrent.core.reasoning import Plan, Planner, Subgoal  # noqa: F401 - the contract, re-exported
from worldcurrent.core.registry import registry
from worldcurrent.core.types import ActionChunk, Observation, StepResult


class IdentityPlanner:
    """Baseline: one subgoal, the original instruction (SubgoalAgent then behaves like the wrapped agent)."""

    name = "planner.identity@0.1"

    def plan(self, instruction: str, observation: Observation) -> Plan:
        return Plan(subgoals=[Subgoal(instruction=instruction, max_steps=10_000)], rationale="no decomposition")

    def done(self, subgoal: Subgoal, observation: Observation) -> tuple[bool, str, dict[str, Any]]:
        return False, "never switches", {}


class SubgoalAgent:
    def __init__(self, inner: Agent | dict[str, Any], planner: Planner | dict[str, Any] | None = None,
                 check_every: int = 1, final_instruction: str = "original") -> None:
        self.inner = registry.build("agents", inner) if isinstance(inner, dict) else inner
        self.planner = (registry.build("planners", planner) if isinstance(planner, dict)
                        else planner or IdentityPlanner())
        self.check_every = max(1, int(check_every))       # ask "done?" every N chunks
        self.final_instruction = final_instruction          # after the last subgoal: "original" | "last"

    def spec(self) -> AgentSpec:
        base = self.inner.spec()
        return base.model_copy(update={"name": base.name + "+subgoals",
                                       "components": {**base.components, "planner": self.planner.name,
                                                      "planner_check_every": str(self.check_every)}})

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.inner.reset(ctx)
        self.plan_: Plan | None = None
        self.index, self.steps_in_goal, self.chunks = 0, 0, 0

    def _emit(self, kind: str, **payload: Any) -> None:
        self.ctx.emit("agent.decision", {"kind": kind, "component": self.planner.name, "trigger": kind, **payload})

    def act(self, observation: Observation) -> ActionChunk:
        if self.plan_ is None:
            self.plan_ = self.planner.plan(observation.instruction or "", observation)
            self._emit("plan", output=[g.instruction for g in self.plan_.subgoals], rationale=self.plan_.rationale,
                       inputs={"instruction": observation.instruction, "meta": self.plan_.meta})
        elif self.index < len(self.plan_.subgoals):
            goal = self.plan_.subgoals[self.index]
            switch, why, meta = False, "", {}
            if self.steps_in_goal >= goal.max_steps:
                switch, why = True, f"step budget {goal.max_steps} exhausted"
            elif self.chunks % self.check_every == 0:
                switch, why, meta = self.planner.done(goal, observation)
            if switch:
                self.index += 1
                self.steps_in_goal = 0
                nxt = self.plan_.subgoals[self.index].instruction if self.index < len(self.plan_.subgoals) else None
                self._emit("subgoal_switch", output=nxt, rationale=why, inputs={"finished": goal.instruction, "meta": meta})
        self.chunks += 1
        return self.inner.act(observation.model_copy(update={"instruction": self.current_instruction(observation)}))

    def current_instruction(self, observation: Observation) -> str:
        assert self.plan_ is not None
        if self.index < len(self.plan_.subgoals):
            return self.plan_.subgoals[self.index].instruction
        return (observation.instruction or "") if self.final_instruction == "original" else self.plan_.subgoals[-1].instruction

    def on_step(self, result: StepResult) -> None:
        self.steps_in_goal += 1
        self.inner.on_step(result)

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        self.ctx.emit("x.planner.summary", {"subgoals_done": self.index,
                                            "subgoals": [g.instruction for g in (self.plan_.subgoals if self.plan_ else [])],
                                            "status": outcome.status})
        self.inner.on_episode_end(outcome)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
