"""Steering a frozen stochastic policy (flow-matching / diffusion VLAs) without touching its weights.

SteeredAgent wraps any ChunkSampler (core.policy.ChunkPolicy: a "policies" plugin, given as a spec or an object)
that can draw action chunks for given noise seeds:
- seeded noise: chunk noise is derived from (task key, chunk index, sample index, resample round), so a candidate
  and its parent face the same randomness on the same init state (common random numbers for paired tests) and
  every episode is replayable;
- sample-and-select: draw N chunks and execute their medoid ("medoid"), or keep the gripper-majority ones first
  ("gripper_vote_medoid", RoboMonkey-style vote); no extra work when N = 1;
- runtime triggers: declarative rules over cheap features (still steps, grasp attempts, sample disagreement)
  that fire recovery actions (resample with more noise, or back off and reopen the gripper).
Everything tunable is data (SteeringConfig), so the improvement loop can propose and gate it as config patches.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
from pydantic import Field

from hylo.core.agent import AgentBase, AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.contract import Contract
from hylo.core.policy import ChunkPolicy, build_policy
from hylo.core.types import ActionChunk, Observation, StepResult

Feature = Literal["still_steps", "grasp_attempts", "disagreement", "steps"]


class TriggerRule(Contract):
    name: str
    feature: Feature
    op: Literal[">=", ">", "<=", "<"] = ">="
    value: float
    action: Literal["resample", "backoff"]
    params: dict[str, float] = Field(default_factory=dict)  # resample: noise_scale; backoff: steps, dz, open_gripper
    cooldown: int = Field(default=30, ge=0)                  # steps before the rule may fire again
    max_fires: int = Field(default=3, ge=1)                  # per episode


class SteeringConfig(Contract):
    n_samples: int = Field(default=1, ge=1, le=8)            # cost grows linearly on CPU
    selector: Literal["first", "medoid", "gripper_vote_medoid"] = "first"
    noise_scale: float = Field(default=1.0, gt=0, le=3.0)
    execution_steps: int | None = Field(default=None, ge=1, le=200)  # None: the policy's own n_action_steps
    seed_noise: bool = True
    rules: list[TriggerRule] = Field(default_factory=list, max_length=4)
    compare_steps: int = Field(default=10, ge=1)             # chunk prefix used for voting / disagreement
    still_eps: float = 0.0015                                # m per step
    gripper_closed_below: float = 0.06                       # finger opening (m) counted as closed
    eef_key: str = "eef_pos"
    gripper_key: str = "gripper_qpos"


ChunkSampler = ChunkPolicy          # the contract lives in core.policy; the old name stays for existing code


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little")


def select_chunk(chunks: np.ndarray, selector: str, k: int) -> tuple[int, float]:
    """Index of the chunk to execute and the sample disagreement (mean distance to the mean prefix)."""
    if len(chunks) == 1:
        return 0, 0.0
    prefix = chunks[:, :k, :]
    disagreement = float(np.mean(np.linalg.norm(prefix - prefix.mean(0, keepdims=True), axis=-1)))
    if selector == "first":
        return 0, disagreement
    if selector == "medoid":                                        # no gripper assumption: any action space
        flat = prefix.reshape(len(chunks), -1)
        return int(np.argmin(np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(1))), disagreement
    grip = np.sign(prefix[:, :, -1].mean(1))
    majority = 1.0 if (grip >= 0).sum() >= (grip < 0).sum() else -1.0
    idx = np.flatnonzero((grip >= 0) == (majority >= 0)) if majority else np.arange(len(chunks))
    flat = prefix[idx, :, :-1].reshape(len(idx), -1)
    dist = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(1)
    return int(idx[int(np.argmin(dist))]), disagreement


class SteeredAgent(AgentBase):
    def __init__(self, sampler: ChunkSampler | dict[str, Any], steering: SteeringConfig | dict[str, Any] | None = None) -> None:
        self.sampler = build_policy(sampler)
        self.cfg = steering if isinstance(steering, SteeringConfig) else SteeringConfig.model_validate(steering or {})

    def spec(self) -> AgentSpec:
        base = self.sampler.spec()
        c = self.cfg
        tag = f"steer(n={c.n_samples},sel={c.selector},noise={c.noise_scale},exec={c.execution_steps},rules={len(c.rules)})"
        return base.model_copy(update={"name": base.name + "+steered", "components": {**base.components, "steering": tag}})

    # -- episode state -------------------------------------------------------------------------------
    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.sampler.reset(ctx)
        self.key = ctx.task.key
        self.chunk_index, self.resample_round, self.steps = 0, 0, 0
        self.still, self.grasps, self.disagreement = 0, 0, 0.0
        self.prev_eef: np.ndarray | None = None
        self.was_closed = False
        self.fires: dict[str, int] = {}
        self.last_fire: dict[str, int] = {}
        self.noise_scale = self.cfg.noise_scale

    def _update_features(self, obs: Observation) -> None:
        s = obs.state
        if self.cfg.eef_key in s:
            eef = np.asarray(s[self.cfg.eef_key], dtype=float)
            if self.prev_eef is not None:
                self.still = self.still + 1 if np.linalg.norm(eef - self.prev_eef) < self.cfg.still_eps else 0
            self.prev_eef = eef
        if self.cfg.gripper_key in s:
            g = np.asarray(s[self.cfg.gripper_key], dtype=float)
            closed = float(g[0] - g[-1]) < self.cfg.gripper_closed_below if g.size > 1 else float(g[0]) < self.cfg.gripper_closed_below
            if closed and not self.was_closed:
                self.grasps += 1
            self.was_closed = closed

    def on_step(self, result: StepResult) -> None:
        self.steps += 1
        self._update_features(result.observation)

    def features(self) -> dict[str, float]:
        return {"still_steps": self.still, "grasp_attempts": self.grasps, "disagreement": self.disagreement,
                "steps": self.steps}

    def _check_rules(self) -> TriggerRule | None:
        f = self.features()
        ops = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less}
        for r in self.cfg.rules:
            if self.fires.get(r.name, 0) >= r.max_fires:
                continue
            if self.steps - self.last_fire.get(r.name, -10**9) < r.cooldown:
                continue
            if ops[r.op](f[r.feature], r.value):
                self.fires[r.name] = self.fires.get(r.name, 0) + 1
                self.last_fire[r.name] = self.steps
                return r
        return None

    # -- acting ---------------------------------------------------------------------------------------
    def act(self, observation: Observation) -> ActionChunk:
        if self.steps == 0:
            self._update_features(observation)
        rule = self._check_rules()
        if rule is not None:
            self.ctx.emit("agent.decision", {"kind": "recovery", "rule": rule.name, "action": rule.action,
                                             "features": self.features(), "params": dict(rule.params),
                                             "rationale": f"{rule.feature} {rule.op} {rule.value}", "trigger": "runtime_rule",
                                             "component": "steering"})
            self.resample_round += 1
            if rule.action == "resample":
                self.noise_scale = float(rule.params.get("noise_scale", self.cfg.noise_scale))
            elif rule.action == "backoff":
                self.still = 0
                return self._backoff_chunk(rule)
        return self._sample_and_select(observation)

    def _backoff_chunk(self, rule: TriggerRule) -> ActionChunk:
        steps = int(rule.params.get("steps", 8))
        a = np.zeros((steps, self.spec().action_dim or 7))
        a[:, 2] = float(rule.params.get("dz", 0.5))                    # move up (eef delta z)
        a[:, -1] = -1.0 if rule.params.get("open_gripper", 1.0) else (1.0 if self.was_closed else -1.0)
        return ActionChunk(actions=a, execution_steps=steps, provenance={"model": "recovery.backoff", "rule": rule.name})

    def _sample_and_select(self, obs: Observation) -> ActionChunk:
        c = self.cfg
        seeds = ([_seed(self.key, self.chunk_index, i, self.resample_round) for i in range(c.n_samples)]
                 if c.seed_noise else None)
        chunks = self.sampler.sample(obs, seeds, self.noise_scale)
        idx, self.disagreement = select_chunk(chunks, c.selector, c.compare_steps)
        self.chunk_index += 1
        exec_steps = c.execution_steps or self.sampler.default_execution_steps()
        return ActionChunk(actions=chunks[idx], execution_steps=min(exec_steps, chunks.shape[1]),
                           provenance={"model": self.sampler.spec().components.get("policy", "policy"),
                                       "n_samples": c.n_samples, "selected": idx, "disagreement": round(self.disagreement, 5),
                                       "noise_scale": self.noise_scale, "seeds": seeds or [],
                                       "chunk_index": self.chunk_index - 1})

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        if self.fires:
            self.ctx.emit("x.steering.summary", {"fires": dict(self.fires), "features": self.features(),
                                                 "status": outcome.status})
