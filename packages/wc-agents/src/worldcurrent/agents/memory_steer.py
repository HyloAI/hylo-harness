"""Action-memory steering for a frozen flow VLA (RFC-0003 M1; mechanism after Retrieve-then-Steer, arXiv 2605.10094).

At every decision the sampler computes a retrieval key inside its own forward pass and calls back here:
1. retrieve the top-K verified segments by cosine similarity (same task by default) above a similarity floor;
2. drop candidates whose chunk is inconsistent with the others (median pairwise DTW over the executed prefix);
3. aggregate the rest into an elite prior with softmax-over-similarity weights (weighted mean for motion,
   weighted vote for the gripper, nearest neighbour when the vote is split);
4. map retrieval confidence (similarity, agreement) to a flow start time t0: confident -> small t0 -> strong guidance;
   nothing usable -> t0 = 1, i.e. the unchanged policy.
The sampler then integrates the policy's own velocity field from x_t0 = (1 - t0) prior + t0 noise, so the current
observation still decides the action: memory biases the sample, it never replays it.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from pydantic import Field

from worldcurrent.core.agent import EpisodeContext
from worldcurrent.core.contract import Contract
from worldcurrent.core.experience import ActionMemory, ActionSegment, PriorDecision
from worldcurrent.core.registry import registry
from worldcurrent.core.types import ActionChunk, Observation

from .steering import SteeredAgent, _seed, select_chunk


class MemorySteerConfig(Contract):
    k: int = Field(default=10, ge=1, le=64)
    same_task: bool = True
    sim_floor: float = Field(default=0.9, ge=-1.0, le=1.0)
    temperature: float = Field(default=0.05, gt=0)        # softmax over similarity
    dtw_mad: float = Field(default=2.0, gt=0)             # drop if inconsistency > median + dtw_mad * MAD
    compare_steps: int | None = Field(default=None, ge=1)  # DTW over this prefix (None: execution steps)
    # SmolVLA's velocity field pulls hard towards its own action: on LIBERO a prior 0.9 std away moves the result
    # ~40% of the way at t0 = 0.1 and almost not at all from t0 = 0.4, so useful guidance lives at small t0.
    t0_min: float = Field(default=0.1, ge=0.0, le=1.0)
    t0_max: float = Field(default=0.7, ge=0.0, le=1.0)
    fixed_t0: float | None = Field(default=None, ge=0.0, le=1.0)   # ablation: ignore confidence
    dispersion_scale: float = Field(default=0.5, gt=0)    # DTW per step (action units) at which confidence halves ~
    gripper_dim: int = -1


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Multivariate dynamic time warping with Euclidean step cost, normalised by path length bound (len(a)+len(b))."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def elite_prior(cands: list[tuple[ActionSegment, float]], cfg: MemorySteerConfig, horizon: int,
                compare: int) -> PriorDecision:
    """Filter, aggregate and score retrieved candidates (already above the similarity floor)."""
    if not cands:
        return PriorDecision(prior=None, t0=1.0, info={"reason": "no candidate above the similarity floor"})
    chunks = [np.asarray(s.actions, dtype=np.float64) for s, _ in cands]
    T = min(horizon, *(len(c) for c in chunks))
    chunks = [c[:T] for c in chunks]
    sims = np.array([sim for _, sim in cands])
    keep = np.arange(len(cands))
    dispersion = 0.0
    if len(cands) >= 3:
        pre = [c[:compare] for c in chunks]
        d = np.zeros((len(pre), len(pre)))
        for i in range(len(pre)):
            for j in range(i + 1, len(pre)):
                d[i, j] = d[j, i] = dtw(pre[i], pre[j])
        score = np.array([np.median(np.delete(d[i], i)) for i in range(len(pre))])
        med, mad = np.median(score), np.median(np.abs(score - np.median(score))) + 1e-9
        keep = np.flatnonzero(score <= med + cfg.dtw_mad * mad)
        sub = d[np.ix_(keep, keep)]
        dispersion = float(sub[np.triu_indices(len(keep), 1)].mean()) if len(keep) > 1 else 0.0
    elif len(cands) == 2:
        dispersion = dtw(chunks[0][:compare], chunks[1][:compare])
    w = np.exp((sims[keep] - sims[keep].max()) / cfg.temperature)
    w /= w.sum()
    stack = np.stack([chunks[i] for i in keep])                     # (K', T, A)
    prior = np.einsum("k,kta->ta", w, stack)
    g = cfg.gripper_dim
    vote = np.einsum("k,kt->t", w, np.sign(stack[:, :, g]))
    nearest = stack[int(np.argmax(w)), :, g]
    prior[:, g] = np.where(np.abs(vote) < 0.2, nearest, np.sign(vote) * np.abs(stack[:, :, g]).mean(0))
    s_norm = float(np.clip((np.dot(w, sims[keep]) - cfg.sim_floor) / max(1e-6, 1.0 - cfg.sim_floor), 0.0, 1.0))
    conf = s_norm * float(np.exp(-dispersion / cfg.dispersion_scale))
    t0 = cfg.fixed_t0 if cfg.fixed_t0 is not None else cfg.t0_max - (cfg.t0_max - cfg.t0_min) * conf
    info = {"candidates": len(cands), "kept": int(len(keep)), "mean_sim": round(float(np.dot(w, sims[keep])), 5),
            "dispersion": round(dispersion, 4), "confidence": round(conf, 4), "t0": round(float(t0), 4),
            "hits": [[s.source.episode_id, s.step, round(sim, 5)] for s, sim in (cands[i] for i in keep[:5])]}
    return PriorDecision(prior=prior, t0=float(t0), info=info)


class MemorySteeredAgent(SteeredAgent):
    """SteeredAgent whose samples start from an action-memory prior (needs a PriorSampler)."""

    def __init__(self, sampler: Any, steering: Any = None, memory: ActionMemory | dict[str, Any] | None = None,
                 memory_steer: MemorySteerConfig | dict[str, Any] | None = None) -> None:
        super().__init__(sampler, steering)
        self.memory = registry.build("memory", memory) if isinstance(memory, dict) else memory
        self.mcfg = memory_steer if isinstance(memory_steer, MemorySteerConfig) else MemorySteerConfig.model_validate(memory_steer or {})
        if self.memory is None:
            raise ValueError("MemorySteeredAgent needs an action memory")

    def spec(self):
        base = super().spec()
        m = self.mcfg
        t0 = f"t0={m.fixed_t0}" if m.fixed_t0 is not None else f"t0=[{m.t0_min},{m.t0_max}]"
        tag = f"bank({self.memory.snapshot()},n={len(self.memory)},k={m.k},floor={m.sim_floor},{t0})"
        return base.model_copy(update={"components": {**base.components, "action_memory": tag}})

    def reset(self, ctx: EpisodeContext) -> None:
        super().reset(ctx)
        self.task_id = ctx.task.spec.id

    def _decide(self, key: np.ndarray, exec_steps: int, horizon: int) -> PriorDecision:
        m = self.mcfg
        hits = self.memory.query(key, task_id=self.task_id if m.same_task else None, k=m.k)
        hits = [(s, sim) for s, sim in hits if sim >= m.sim_floor]
        decision = elite_prior(hits, m, horizon, m.compare_steps or exec_steps)
        self._last = decision
        return decision

    def _sample_and_select(self, obs: Observation) -> ActionChunk:
        c = self.cfg
        seeds = ([_seed(self.key, self.chunk_index, i, self.resample_round) for i in range(c.n_samples)]
                 if c.seed_noise else None)
        exec_steps = c.execution_steps or self.sampler.default_execution_steps()
        horizon = int(getattr(getattr(self.sampler, "cfg", None), "chunk_size", 50))
        self._last = PriorDecision(prior=None, info={"reason": "sampler did not call back"})
        chunks = self.sampler.sample(obs, seeds, self.noise_scale,
                                     prior_fn=lambda key: self._decide(key, exec_steps, horizon))
        idx, self.disagreement = select_chunk(chunks, c.selector, c.compare_steps)
        self.chunk_index += 1
        info = dict(self._last.info)
        self.ctx.emit("memory.retrieval", {"chunk_index": self.chunk_index - 1, "step": self.steps,
                                           "used_prior": self._last.prior is not None, **info})
        return ActionChunk(actions=chunks[idx], execution_steps=min(exec_steps, chunks.shape[1]),
                           provenance={"model": self.sampler.spec().components.get("policy", "policy"),
                                       "n_samples": c.n_samples, "selected": idx, "disagreement": round(self.disagreement, 5),
                                       "noise_scale": self.noise_scale, "seeds": seeds or [],
                                       "chunk_index": self.chunk_index - 1, "t0": info.get("t0", 1.0),
                                       "memory_kept": info.get("kept", 0)})


def verified_decision_points(trace: Any, *, splits: tuple[str, ...] = ("search", "search_val")
                             ) -> Iterator[dict[str, Any]]:
    """Policy chunks of a successful development-split episode whose observation has camera images.

    Yields {"observation": payload, "actions": (T, A), "step", "progress", "task_id", "episode_id", "split"}.
    Only simulator-verified successes qualify (unverified memory is worse than none)."""
    start, end = trace.first("episode.start"), trace.last("episode.end")
    if start is None or end is None or end["payload"]["status"] != "success":
        return
    sp = start["payload"]
    if sp.get("split") not in splits:
        return
    total = max(1, int(end["payload"]["steps"]))
    obs, step = None, 0
    for e in trace.events:
        t, p = e["type"], e["payload"]
        if t == "observation":
            obs = p
        elif t == "step.result":
            step += 1
        elif t == "agent.chunk" and obs is not None and obs.get("images"):
            if str((p.get("provenance") or {}).get("model", "")).startswith("recovery."):
                continue
            yield {"observation": obs, "actions": np.asarray(p["actions"], dtype=np.float64), "step": step,
                   "progress": min(1.0, step / total), "task_id": sp["task_id"], "episode_id": trace.episode_id,
                   "split": sp["split"], "provenance": dict(p.get("provenance") or {})}
