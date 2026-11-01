"""Agent contract. The kernel only knows `Agent`; reasoners, planners, policies, world models and
memory are an Agent's internal composition (hylo-agents)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from pydantic import Field

from .contract import Contract
from .task import TaskInstance
from .types import ActionChunk, ActionSemantics, EnvSpec, Observation, StepResult


class AgentSpec(Contract):
    name: str
    version: str
    required_state: tuple[str, ...] = ()        # observation.state keys the agent reads
    required_cameras: tuple[str, ...] = ()
    action_semantics: ActionSemantics | None = None
    action_dim: int | None = None
    components: dict[str, str] = Field(default_factory=dict)  # role -> "name@version"
    weights_license: str | None = None          # mandatory for learned policies


EpisodeStatus = Literal["success", "failure", "error", "safety_abort", "infra_invalid", "unlabeled"]  # unlabeled:
# nobody could judge the outcome (a robot episode before the operator's verdict, an external episode)


class EpisodeOutcome(Contract):
    episode_id: str
    task: TaskInstance
    status: EpisodeStatus
    success: bool | None
    steps: int
    termination_reason: str


@dataclass
class EpisodeContext:
    """Runtime handle given to the Agent at reset. Not serialized."""

    episode_id: str
    task: TaskInstance
    env_spec: EnvSpec
    emit: Callable[[str, Mapping[str, Any]], None]   # writes namespaced events into the trace
    seed: int = 0


@runtime_checkable
class Agent(Protocol):
    def spec(self) -> AgentSpec: ...
    def reset(self, ctx: EpisodeContext) -> None: ...
    def act(self, observation: Observation) -> ActionChunk: ...
    def on_step(self, result: StepResult) -> None: ...
    def on_episode_end(self, outcome: EpisodeOutcome) -> None: ...


class AgentBase:
    """Convenience base with no-op hooks. Not required by the protocol."""

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def on_step(self, result: StepResult) -> None:
        pass

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        pass
