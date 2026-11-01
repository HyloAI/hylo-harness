"""Experience contracts (RFC-0003): action memory, lessons and skills. Contracts and protocols only; stores live in
hylo-memory, consumers in hylo-agents, writers in hylo-evolve / hylo-reasoner-llm.

Every memory used in an evaluation is a frozen artifact with a content digest (recorded in the agent spec), and is
built from verified development-split experience only.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import Field

from .contract import Contract, JsonAny, NDArray
from .types import Observation


class EpisodeRef(Contract):
    run_id: str
    episode_id: str
    split: str
    judge: str = "sim"                   # who verified the outcome: "sim" (ground truth) | "vlm:<model>" | "human"


# ---------------------------------------------------------------- action memory

class ActionSegment(Contract):
    """One verified decision point: the retrieval key of the observation and the chunk the policy then produced."""

    key: NDArray                         # (D,) L2-normalised
    actions: NDArray                     # (T, A) unnormalised, as the policy emitted it
    task_id: str
    step: int = Field(ge=0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)   # step / episode length (success at 1)
    source: EpisodeRef


@runtime_checkable
class ActionMemory(Protocol):
    def add(self, segments: Sequence[ActionSegment]) -> int: ...
    def query(self, key: np.ndarray, *, task_id: str | None = None, k: int = 10) -> list[tuple[ActionSegment, float]]: ...
    def snapshot(self) -> str: ...       # content digest
    def save(self, path: str) -> str: ...
    def __len__(self) -> int: ...


class PriorDecision(Contract):
    """What the memory steerer hands the sampler: a chunk prior and the flow time to start integrating from."""

    prior: NDArray | None                # (T, A) unnormalised, or None (sample from pure noise)
    t0: float = Field(default=1.0, ge=0.0, le=1.0)          # 1.0 = the unchanged policy
    info: dict[str, JsonAny] = Field(default_factory=dict)
    # noise-space steering (RFC-0004 W3): the initial flow noise becomes sqrt(w) * noise + sqrt(1 - w) * fresh noise,
    # integrated over the full flow; the policy is unchanged. noise is in the policy's (normalised) noise space.
    noise: NDArray | None = None
    noise_weight: float = Field(default=0.0, ge=0.0, le=1.0)


PriorFn = Callable[[np.ndarray], PriorDecision]


@runtime_checkable
class PriorSampler(Protocol):
    """A ChunkSampler that can start flow sampling from x_t0 = (1 - t0) * prior + t0 * noise. `prior_fn` receives the
    retrieval key computed inside the policy's own forward pass, so retrieval costs no extra encoder call."""

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0,
               prior_fn: PriorFn | None = None) -> np.ndarray: ...
    def retrieval_key(self, observation: Observation) -> np.ndarray: ...


# ---------------------------------------------------------------- lessons (M4)

class Lesson(Contract):
    id: str
    scope: str                           # "task:<id>" | "pattern:<name>" | "global"
    kind: Literal["strategy", "pitfall"]
    title: str
    content: str
    helpful: int = 0
    harmful: int = 0
    evidence: list[EpisodeRef] = Field(default_factory=list)
    status: Literal["active", "retired"] = "active"


# ---------------------------------------------------------------- skills (M3)

class SkillTrigger(Contract):
    instruction_pattern: str | None = None   # regex with named groups, e.g. r"put both the (?P<a>.+) and the (?P<b>.+) in the basket"
    task_tags: tuple[str, ...] = ()
    env: str | None = None


class Skill(Contract):
    name: str
    version: str
    description: str                     # name + description form the index an LLM sees (progressive disclosure)
    trigger: SkillTrigger
    kind: Literal["plan", "steering", "recovery", "weights"]
    body: dict[str, JsonAny]             # plan: {"subgoals": [templates]}; steering/recovery: a config patch; weights: a ref
    status: Literal["proposed", "verified", "refuted", "deprecated"] = "proposed"
    verdicts: dict[str, int] = Field(default_factory=dict)   # usage feedback: useful / misleading / not_applicable
    evidence: dict[str, Any] = Field(default_factory=dict)   # gate statistics, archive entry, source candidate digest
