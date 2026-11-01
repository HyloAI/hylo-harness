"""policy.mpc: model-predictive control with a learned world model (the planning half of TD-MPC / PETS, without a
learned value function).

Each decision runs the cross-entropy method over `horizon`-step action sequences in the world model (an
artifact of the "world_model.state" learner): sample sequences from a Gaussian (seeded), roll them out with the
ensemble, keep the best `elites`, refit, repeat. The objective is either the model's learned success probability
("success"), or a task cost given as data, {"reach": ["tip_pos", "target_pos"]}: the distance between two state
entries. `uncertainty_penalty` > 0 adds the ensemble's disagreement to the cost, keeping plans where the model has
data (off by default: it slowed reaching on MuJoCo planar reach).

Train the world model on data that explores: expert runs alone cover too few actions, and the planner then
exploits the model's errors. On MuJoCo planar reach (40 held-out episodes): expert-only data 0.88 success, expert +
random-action runs 0.95 in 44 steps, with these defaults; 64 samples x 3 iterations, re-planning every 2 steps:
0.42 / 0.75.

    policy: {type: policy.mpc, params: {world_model: artifacts/wm, objective: {reach: [tip_pos, target_pos]}}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hylo.agents.world_model import StateWorldModel
from hylo.core.agent import AgentSpec, EpisodeContext
from hylo.core.types import Observation

from .base import _vec


class MPCPolicy:
    def __init__(self, world_model: str, objective: str | dict[str, Any] = "success", horizon: int = 10,
                 population: int = 128, elites: int = 8, iterations: int = 5, execution_steps: int = 1,
                 uncertainty_penalty: float = 0.0, action_low: float = -1.0, action_high: float = 1.0) -> None:
        self.model = StateWorldModel.load(world_model)
        self.objective, self.horizon, self.population, self.elites = objective, horizon, population, elites
        self.iterations, self.exec_steps, self.penalty = iterations, execution_steps, uncertainty_penalty
        env_spec = self.model.meta.get("env_spec") or {}
        action = env_spec.get("action") or {}
        self.action_dim = int((action.get("shape") or [None])[0] or 0) or None
        self.low = np.asarray(action.get("low") or [action_low] * (self.action_dim or 1), dtype=float)
        self.high = np.asarray(action.get("high") or [action_high] * (self.action_dim or 1), dtype=float)
        self.semantics = action.get("semantics")
        if self.action_dim is None:
            raise ValueError("the world model has no env_spec: build it with params.env set (world_model.state learner)")
        idx, i = {}, 0
        for k, n in zip(self.model.keys, self.model.dims):
            idx[k], i = slice(i, i + n), i + n
        self.idx = idx

    def spec(self) -> AgentSpec:
        from hylo.core.types import ActionSemantics

        return AgentSpec(name="policy.mpc", version="0.1", required_state=tuple(self.model.keys),
                         action_semantics=ActionSemantics.model_validate(self.semantics) if self.semantics else None,
                         action_dim=self.action_dim,
                         components={"policy": f"cem_mpc(h={self.horizon},n={self.population},obj={self.objective})"},
                         weights_license="self-trained from development runs")

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def default_execution_steps(self) -> int:
        return min(self.exec_steps, self.horizon)

    def _cost(self, s: np.ndarray) -> np.ndarray:
        if isinstance(self.objective, dict) and "reach" in self.objective:
            a, b = self.objective["reach"]
            return np.linalg.norm(s[:, self.idx[a]] - s[:, self.idx[b]], axis=1)
        return -self.model.p_done_batch(s)

    def _plan(self, s0: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
        A, H = self.action_dim, self.horizon
        mu, sd = np.zeros((H, A)), np.full((H, A), 0.5 * (self.high - self.low) * noise_scale)
        for _ in range(self.iterations):
            seq = np.clip(mu + sd * rng.standard_normal((self.population, H, A)), self.low, self.high)
            s, cost = np.repeat(s0[None], self.population, axis=0), np.zeros(self.population)
            for t in range(H):
                s, unc = self.model.predict_batch(s, seq[:, t])
                cost += self._cost(s) + self.penalty * unc
            elite = seq[np.argsort(cost)[: self.elites]]
            mu, sd = elite.mean(0), elite.std(0) + 1e-3
        return mu

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        s0 = _vec(observation.state, self.model.keys)
        rngs = [np.random.default_rng(s) for s in seeds] if seeds else [np.random.default_rng()]
        return np.stack([self._plan(s0, r, noise_scale) for r in rngs])
