"""Noise-space steering (RFC-0004 W3): an offline, retrieval-based counterpart of DSRL for a frozen flow VLA.

DSRL learns a policy over the flow policy's initial noise and leaves the VLA untouched. Offline, from our own
successes: every successful chunk was produced from a known seed, so its exact initial noise is recoverable. The
bank (memory_build --store noise) keys those noises by the policy's own visual tokens. At decision time the agent
retrieves the nearest same-task noises, averages them (similarity-weighted), rescales the mean to unit variance and
hands it to the sampler, which starts the full flow from sqrt(w) * prior noise + sqrt(1 - w) * fresh noise. With
w = 0 this is exactly the policy; nothing is retrieved below `sim_floor`.

Unlike an action prior, the chunk is still produced by the policy's velocity field for the *current* observation:
the noise only biases which mode the flow lands in.

    agent: {type: remote, params: {ref: worldcurrent.policies.lerobot.experience_agent:make_noise_agent, params: {
        repo_id: lerobot/smolvla_libero, memory: {type: action_bank, params: {path: "artifact:l10-noise-search-v1@2687768f", frozen: true}},
        noise_steer: {weight: 0.5, k: 5, sim_floor: 0.95}, steering: {n_samples: 1, seed_noise: true}}}}
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from worldcurrent.core.agent import EpisodeContext
from worldcurrent.core.contract import Contract
from worldcurrent.core.experience import ActionMemory, PriorDecision
from worldcurrent.core.registry import registry
from worldcurrent.core.types import ActionChunk, Observation

from .steering import SteeredAgent, _seed, select_chunk


class NoiseSteerConfig(Contract):
    weight: float = Field(default=0.5, ge=0.0, le=1.0)       # share of the retrieved noise in the initial noise
    k: int = Field(default=5, ge=1)
    same_task: bool = True
    sim_floor: float = Field(default=0.95, ge=-1.0, le=1.0)
    temperature: float = Field(default=0.02, gt=0.0)          # softmax over cosine similarity


def noise_prior(hits: list[tuple[Any, float]], cfg: NoiseSteerConfig) -> PriorDecision:
    if not hits:
        return PriorDecision(prior=None, info={"kept": 0, "reason": "no neighbour above the similarity floor"})
    sims = np.array([s for _, s in hits])
    w = np.exp((sims - sims.max()) / cfg.temperature)
    w /= w.sum()
    mean = sum(wi * np.asarray(seg.actions, dtype=np.float64) for (seg, _), wi in zip(hits, w))
    std = float(mean.std()) or 1.0
    return PriorDecision(prior=None, noise=(mean / std).astype(np.float32), noise_weight=cfg.weight,
                         info={"kept": len(hits), "top_sim": round(float(sims.max()), 4), "weight": cfg.weight,
                               "effective_neighbours": round(float(1 / (w ** 2).sum()), 2)})


class NoiseSteeredAgent(SteeredAgent):
    def __init__(self, sampler: Any, steering: Any = None, memory: ActionMemory | dict[str, Any] | None = None,
                 noise_steer: NoiseSteerConfig | dict[str, Any] | None = None) -> None:
        super().__init__(sampler, steering)
        self.memory = registry.build("memory", memory) if isinstance(memory, dict) else memory
        self.ncfg = noise_steer if isinstance(noise_steer, NoiseSteerConfig) else NoiseSteerConfig.model_validate(noise_steer or {})
        if self.memory is None:
            raise ValueError("NoiseSteeredAgent needs a noise bank (memory_build --store noise)")
        self._last = PriorDecision(prior=None)

    def spec(self):
        base = super().spec()
        n = self.ncfg
        tag = f"noise_bank({self.memory.snapshot()},n={len(self.memory)},k={n.k},floor={n.sim_floor},w={n.weight})"
        return base.model_copy(update={"components": {**base.components, "noise_prior": tag}})

    def reset(self, ctx: EpisodeContext) -> None:
        super().reset(ctx)
        self.task_id = ctx.task.spec.id

    def _decide(self, key: np.ndarray) -> PriorDecision:
        n = self.ncfg
        hits = [(s, sim) for s, sim in self.memory.query(key, task_id=self.task_id if n.same_task else None, k=n.k)
                if sim >= n.sim_floor]
        self._last = noise_prior(hits, n)
        return self._last

    def _sample_and_select(self, obs: Observation) -> ActionChunk:
        c = self.cfg
        seeds = ([_seed(self.key, self.chunk_index, i, self.resample_round) for i in range(c.n_samples)]
                 if c.seed_noise else None)
        exec_steps = c.execution_steps or self.sampler.default_execution_steps()
        self._last = PriorDecision(prior=None, info={"reason": "sampler did not call back"})
        chunks = self.sampler.sample(obs, seeds, self.noise_scale, prior_fn=self._decide)
        idx, self.disagreement = select_chunk(chunks, c.selector, c.compare_steps)
        self.chunk_index += 1
        info = dict(self._last.info)
        self.ctx.emit("memory.retrieval", {"chunk_index": self.chunk_index - 1, "step": self.steps, "kind": "noise",
                                           "used_prior": self._last.noise is not None, **info})
        return ActionChunk(actions=chunks[idx], execution_steps=min(exec_steps, chunks.shape[1]),
                           provenance={"model": self.sampler.spec().components.get("policy", "policy"),
                                       "n_samples": c.n_samples, "selected": idx, "disagreement": round(self.disagreement, 5),
                                       "noise_scale": self.noise_scale, "seeds": seeds or [], "chunk_index": self.chunk_index - 1,
                                       "noise_weight": self.ncfg.weight if self._last.noise is not None else 0.0,
                                       "memory_kept": info.get("kept", 0)})
