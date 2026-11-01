"""Policies (plugin group "policies"): action-chunk generators that agents run, steer and compose.

An Agent is what the kernel talks to (`core.agent`). A ChunkPolicy is the model inside: given an observation it
draws action chunks, one per noise seed. Separating the two lets one policy be run, steered (seeded noise,
sample-and-select, recovery rules), temporally ensembled (ACT), shadowed or memory-steered by agents in wc-agents
without the policy knowing, and lets one agent wrap any policy by configuration:

    agent: {type: policy, params: {policy: {type: policy.flow, params: {path: artifacts/flow}}}}
    agent: {type: steered, params: {sampler: {type: lerobot, params: {repo_id: lerobot/smolvla_libero}},
                                    steering: {n_samples: 4, selector: medoid}}}

Deterministic policies (regression, retrieval, scripted) ignore the seeds and return identical samples.
Stochastic ones (diffusion, flow matching, sampled tokens) must derive their randomness from the seeds only, so
a candidate and its parent face the same noise on the same init state and an episode can be replayed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .agent import AgentSpec, EpisodeContext
from .types import Observation


@runtime_checkable
class ChunkPolicy(Protocol):
    def spec(self) -> AgentSpec: ...
    def reset(self, ctx: EpisodeContext) -> None: ...
    def default_execution_steps(self) -> int: ...

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float) -> np.ndarray:
        """(N, T, A) action chunks in the environment's units, one per seed (N = len(seeds), or 1 if None)."""
        ...


def build_policy(policy: dict[str, Any] | Any) -> ChunkPolicy:
    """A policy from a registry spec {type, params}, or an already built one."""
    if isinstance(policy, dict):
        from .registry import registry

        return registry.build("policies", policy)
    return policy


__all__ = ["ChunkPolicy", "build_policy"]
