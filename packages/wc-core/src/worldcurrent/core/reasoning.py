"""Reasoning plugin contracts, one per role (each its own plugin group and version):

- `reasoners` (wc.reasoner): the slow, event-triggered part of an agent; turns observations and memory into a
  Decision (a subgoal, a skill, an abort), with begin / deliberate / end around an episode.
- `choosers` (wc.chooser): which object an instruction refers to, among perceived or remembered objects.
- `planners` (wc.planner): decomposes an instruction into short sequential subgoals and judges when one is done.
- `reflectors` (wc.reflector): distils lessons from a campaign's diagnosis into operations on a lesson book.

Rule-based, LLM and VLM implementations satisfy the same protocols; the kernel never sees the difference. The
agents that use them (composable object reach, SubgoalAgent, the experience generator) live in wc-agents and
wc-evolve; the LLM ones in wc-reasoner-llm and wc-proposer-llm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol

from pydantic import Field

from .contract import Contract, JsonAny
from .memory import MemoryStore, WorkingMemory
from .types import Observation

Emit = Callable[[str, Mapping[str, Any]], None]


# ---------------------------------------------------------------- reasoners
class Decision(Contract):
    kind: str                                  # "subgoal" | "skill" | "abort" | ...
    output: JsonAny
    rationale: str
    trigger: str
    component: str
    inputs: dict[str, JsonAny] = Field(default_factory=dict)


@dataclass
class ReasonerContext:
    task_id: str
    episode_id: str
    working: WorkingMemory
    memory: MemoryStore | None
    emit: Emit


class Reasoner(Protocol):
    name: str
    required_state: tuple[str, ...]     # observation.state keys this reasoner reads
    visibility_key: str                 # state key whose 0/1 change triggers re-deliberation

    def begin(self, rctx: ReasonerContext) -> None: ...
    def deliberate(self, obs: Observation, trigger: str, rctx: ReasonerContext) -> Decision: ...
    def end(self, success: bool | None, rctx: ReasonerContext) -> None: ...


# ---------------------------------------------------------------- choosers
class SceneObject(Contract):
    id: str
    rgb: tuple[float, float, float]
    position: tuple[float, float]
    source: Literal["seen", "remembered"]


class Choice(Contract):
    object_id: str | None
    rationale: str
    meta: dict[str, Any] = Field(default_factory=dict)   # model, latency_s, cached, raw output ...


class Chooser(Protocol):
    name: str

    def choose(self, instruction: str, objects: list[SceneObject], robot_xy: tuple[float, float],
               obs: Observation | None = None) -> Choice: ...   # obs gives access to camera images if any


# ---------------------------------------------------------------- planners
class Subgoal(Contract):
    instruction: str
    max_steps: int = Field(default=250, ge=1)


class Plan(Contract):
    subgoals: list[Subgoal] = Field(min_length=1, max_length=6)
    rationale: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Planner(Protocol):
    name: str

    def plan(self, instruction: str, observation: Observation) -> Plan: ...
    def done(self, subgoal: Subgoal, observation: Observation) -> tuple[bool, str, dict[str, Any]]: ...


# ---------------------------------------------------------------- reflectors
class Reflector(Protocol):
    """Reads only what a generator may read (search-split diagnostics and timelines) and returns operations
    (ADD / UPVOTE / DOWNVOTE / EDIT / RETIRE) that the lesson book applies deterministically."""

    name: str

    def reflect(self, diagnosis: Mapping[str, Any], book: Any) -> list[dict[str, Any]]: ...


__all__ = ["Choice", "Chooser", "Decision", "Emit", "Plan", "Planner", "Reasoner", "ReasonerContext", "Reflector",
           "SceneObject", "Subgoal"]
