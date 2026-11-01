"""Agents that run a policy (core.policy.ChunkPolicy) given by configuration.

- "policy": draw one chunk per decision (seeded noise, so episodes replay and paired comparisons share noise)
  and execute its first `execution_steps` actions open loop. The plain way to run any policy plugin.
- "temporal_ensemble": ACT's inference (Zhao et al. 2023): query the policy every step and execute, for the
  current step, the exponentially weighted average of every chunk predicted so far that covers it,
  w_i = exp(-m * i) with i = 0 the oldest prediction. Smoother motion at the price of one query per step.

    agent: {type: policy, params: {policy: {type: policy.bc, params: {path: artifacts/bc}}, execution_steps: 8}}
    agent: {type: temporal_ensemble, params: {policy: {type: policy.bc, params: {path: artifacts/bc}}, m: 0.1}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from worldcurrent.core.agent import AgentBase, AgentSpec, EpisodeContext
from worldcurrent.core.policy import ChunkPolicy, build_policy
from worldcurrent.core.types import ActionChunk, Observation, StepResult

from .steering import SteeredAgent


class PolicyAgent(SteeredAgent):
    """A policy executed chunk by chunk: SteeredAgent with one sample and no rules."""

    def __init__(self, policy: ChunkPolicy | dict[str, Any], execution_steps: int | None = None, seed_noise: bool = True,
                 noise_scale: float = 1.0) -> None:
        super().__init__(policy, {"n_samples": 1, "execution_steps": execution_steps, "seed_noise": seed_noise,
                                  "noise_scale": noise_scale})

    def spec(self) -> AgentSpec:
        base = self.sampler.spec()
        return base.model_copy(update={"components": {**base.components,
                                                      "runner": f"chunked(exec={self.cfg.execution_steps})"}})


class TemporalEnsembleAgent(AgentBase):
    def __init__(self, policy: ChunkPolicy | dict[str, Any], m: float = 0.01, seed_noise: bool = True,
                 noise_scale: float = 1.0) -> None:
        if m < 0:
            raise ValueError("m must be >= 0 (0 = plain average)")
        self.policy, self.m, self.seed_noise, self.noise_scale = build_policy(policy), float(m), seed_noise, noise_scale

    def spec(self) -> AgentSpec:
        base = self.policy.spec()
        return base.model_copy(update={"name": base.name + "+temporal_ensemble",
                                       "components": {**base.components, "runner": f"temporal_ensemble(m={self.m})"}})

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.policy.reset(ctx)
        self.t = 0
        self.chunks: list[tuple[int, np.ndarray]] = []          # (step the chunk was predicted at, chunk)

    def on_step(self, result: StepResult) -> None:
        self.t += 1

    def act(self, observation: Observation) -> ActionChunk:
        from .steering import _seed

        seeds = [_seed(self.ctx.task.key, self.t)] if self.seed_noise else None
        chunk = np.asarray(self.policy.sample(observation, seeds, self.noise_scale))[0]
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < len(c)] + [(self.t, chunk)]
        preds = np.stack([c[self.t - t0] for t0, c in self.chunks])          # oldest first
        w = np.exp(-self.m * np.arange(len(preds)))
        action = (w[:, None] * preds).sum(0) / w.sum()
        return ActionChunk(actions=action[None], execution_steps=1,
                           provenance={"model": self.policy.spec().components.get("policy", "policy"),
                                       "ensembled": len(preds), "seeds": seeds or []})
