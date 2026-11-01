"""CriticSteeredAgent: a SteeredAgent that executes the sampled chunk a learned value model scores highest (RFC-0002 L3).

The value model is an instrument (hylo.instruments.critic.ChunkCritic, in the hylo-instruments distribution); it is
imported when an agent is built, so hylo-agents installs without it (extra `hylo-agents[critic]`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from hylo.core.agent import EpisodeContext
from hylo.core.contract import Contract
from hylo.core.types import ActionChunk, Observation

from .steering import SteeredAgent, _seed, select_chunk

class CriticConfig(Contract):
    path: str
    beta: float = Field(default=0.5, ge=0.0, le=5.0)          # pessimism: score = mean - beta * std
    within_vote: bool = True                                   # rank only the gripper-majority chunks
    min_samples: int = Field(default=2, ge=1)                  # below this, fall back to the steering selector


class CriticSteeredAgent(SteeredAgent):
    """SteeredAgent whose sample-and-select step picks the chunk the critic scores highest."""

    def __init__(self, sampler: Any, steering: Any = None, critic: CriticConfig | dict[str, Any] | None = None) -> None:
        super().__init__(sampler, steering)
        self.critic_cfg = critic if isinstance(critic, CriticConfig) else CriticConfig.model_validate(critic)
        from hylo.instruments.critic import ChunkCritic

        self.critic = ChunkCritic.load(self.critic_cfg.path)

    def spec(self):
        base = super().spec()
        c = self.critic_cfg
        tag = f"critic({self.critic.meta.get('weights_sha256_16', '?')},beta={c.beta},vote={c.within_vote})"
        return base.model_copy(update={"components": {**base.components, "critic": tag}})

    def reset(self, ctx: EpisodeContext) -> None:
        super().reset(ctx)
        self.task_id = ctx.task.spec.id

    def _sample_and_select(self, obs: Observation) -> ActionChunk:
        c, cc = self.cfg, self.critic_cfg
        if c.n_samples < cc.min_samples:
            return super()._sample_and_select(obs)
        seeds = ([_seed(self.key, self.chunk_index, i, self.resample_round) for i in range(c.n_samples)]
                 if c.seed_noise else None)
        chunks = self.sampler.sample(obs, seeds, self.noise_scale)
        exec_steps = min(c.execution_steps or self.sampler.default_execution_steps(), chunks.shape[1])
        vote_idx, self.disagreement = select_chunk(chunks, c.selector, c.compare_steps)
        pool = np.arange(len(chunks))
        if cc.within_vote and len(chunks) > 1:
            grip = np.sign(chunks[:, : c.compare_steps, -1].mean(1))
            same = np.flatnonzero((grip >= 0) == (grip[vote_idx] >= 0))
            pool = same if len(same) else pool
        score, mu, sd = self.critic.score_chunks(obs, chunks[pool], step=self.steps, task_id=self.task_id,
                                                 execution_steps=exec_steps, beta=cc.beta)
        idx = int(pool[int(np.argmax(score))])
        self.chunk_index += 1
        return ActionChunk(actions=chunks[idx], execution_steps=exec_steps,
                           provenance={"model": self.sampler.spec().components.get("policy", "policy"),
                                       "n_samples": c.n_samples, "selected": idx, "selector": "critic",
                                       "vote_choice": vote_idx, "disagreement": round(self.disagreement, 5),
                                       "critic_mean": [round(float(v), 4) for v in mu],
                                       "critic_std": [round(float(v), 4) for v in sd],
                                       "candidates": [int(i) for i in pool], "noise_scale": self.noise_scale,
                                       "seeds": seeds or [], "chunk_index": self.chunk_index - 1})
