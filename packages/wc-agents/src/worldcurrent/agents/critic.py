"""Chunk value critic (RFC-0002 L3): a small learned Q(s, chunk) that re-ranks the N chunks a frozen VLA samples.

V-GPS-style test-time steering, scaled to CPU and to the data we already have:
- data: our own traces. Each `agent.chunk` event plus the observation before it is one (state, chunk) sample;
  the target is the discounted success-to-go gamma**(steps until success), 0 if the episode failed
  (Monte Carlo, no bootstrapping: the behaviour policy is the policy being steered).
- features: proprio state, episode progress, task one-hot, a summary of the chunk's executed part
  (net translation / rotation, gripper level and flips), optionally a coarse camera thumbnail.
- model: an ensemble of small numpy MLPs; selection maximises mean - beta * std (pessimism against the
  critic's own extrapolation, the role Cal-QL's conservatism plays in V-GPS).
Weights are .npz + JSON metadata (no pickle), so the critic loads in any venv.

    agent: {type: remote, params: {ref: worldcurrent.policies.lerobot.critic_agent:make_critic_steered, params: {
        repo_id: ..., steering: {n_samples: 4, execution_steps: 25}, critic: {path: critics/l10-v1, beta: 0.5}}}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from worldcurrent.core.agent import EpisodeContext
from worldcurrent.core.contract import Contract
from worldcurrent.core.types import ActionChunk, Observation

from .steering import SteeredAgent, _seed, select_chunk

STATE_KEYS = ("eef_pos", "eef_axis_angle", "gripper_qpos")


class FeatureConfig(Contract):
    state_keys: tuple[str, ...] = STATE_KEYS
    prefix_steps: int = Field(default=10, ge=1)
    horizon: int = Field(default=520, ge=1)           # steps used to normalise progress (libero_10 max)
    tasks: tuple[str, ...] = ()                       # task-id vocabulary for the one-hot (unknown -> all zeros)
    image_camera: str | None = None                   # e.g. "agentview"; None: no image features
    image_px: int = Field(default=8, ge=2, le=32)     # thumbnail side; grayscale


def chunk_features(chunk: np.ndarray, execution_steps: int, prefix_steps: int) -> np.ndarray:
    """Summary of the part of a (T, 7) chunk that will actually run: net 6-D motion over the prefix and the
    executed part, mean gripper command over each, and the number of gripper sign flips."""
    e = max(1, min(int(execution_steps), len(chunk)))
    k = max(1, min(prefix_steps, e))
    run, pre = chunk[:e], chunk[:k]
    flips = float(np.count_nonzero(np.diff(np.sign(run[:, -1])) != 0))
    return np.concatenate([pre[:, :-1].sum(0), [pre[:, -1].mean()], run[:, :-1].sum(0), [run[:, -1].mean(), flips]])


def state_features(state: dict[str, Any], step: int, task_id: str, images: dict[str, Any] | None,
                   cfg: FeatureConfig) -> np.ndarray:
    parts = [np.asarray(state[k], dtype=np.float64).reshape(-1) for k in cfg.state_keys]
    parts.append(np.array([step / cfg.horizon]))
    onehot = np.zeros(len(cfg.tasks))
    if task_id in cfg.tasks:
        onehot[cfg.tasks.index(task_id)] = 1.0
    parts.append(onehot)
    if cfg.image_camera:
        img = (images or {}).get(cfg.image_camera)
        if img is None:
            parts.append(np.zeros(cfg.image_px * cfg.image_px + 1))
        else:
            parts.append(np.append(_thumb(np.asarray(img), cfg.image_px), 1.0))   # trailing 1 = "image present"
    return np.concatenate(parts)


def _thumb(img: np.ndarray, px: int) -> np.ndarray:
    g = img[..., :3].astype(np.float64).mean(-1) / 255.0
    h, w = g.shape
    g = g[: h - h % px, : w - w % px]
    return g.reshape(px, g.shape[0] // px, px, g.shape[1] // px).mean((1, 3)).reshape(-1)


# ---------------------------------------------------------------- model

class _MLP:
    """Two hidden layers, ReLU, sigmoid output; trained with Adam on binary cross-entropy with soft targets."""

    def __init__(self, n_in: int, hidden: int, rng: np.random.Generator) -> None:
        def init(a: int, b: int) -> np.ndarray:
            return rng.normal(0, np.sqrt(2.0 / a), size=(a, b))
        self.p = {"W1": init(n_in, hidden), "b1": np.zeros(hidden), "W2": init(hidden, hidden), "b2": np.zeros(hidden),
                  "W3": init(hidden, 1) * 0.1, "b3": np.zeros(1)}

    def forward(self, x: np.ndarray, cache: bool = False):
        h1 = np.maximum(0, x @ self.p["W1"] + self.p["b1"])
        h2 = np.maximum(0, h1 @ self.p["W2"] + self.p["b2"])
        z = (h2 @ self.p["W3"] + self.p["b3"])[:, 0]
        return (z, (x, h1, h2)) if cache else z

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, *, epochs: int, lr: float, l2: float, batch: int,
            rng: np.random.Generator) -> None:
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        t = 0
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for s in range(0, len(x), batch):
                idx = order[s:s + batch]
                z, (xb, h1, h2) = self.forward(x[idx], cache=True)
                dz = (1 / (1 + np.exp(-z)) - y[idx]) * w[idx] / w[idx].sum()
                g = {"W3": h2.T @ dz[:, None], "b3": dz.sum(keepdims=True)}
                d2 = (dz[:, None] @ self.p["W3"].T) * (h2 > 0)
                g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
                d1 = (d2 @ self.p["W2"].T) * (h1 > 0)
                g["W1"], g["b1"] = xb.T @ d1, d1.sum(0)
                t += 1
                for k in self.p:
                    gk = g[k] + (l2 * self.p[k] if k.startswith("W") else 0)
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v[k] = 0.999 * v[k] + 0.001 * gk * gk
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)


class ChunkCritic:
    def __init__(self, features: FeatureConfig, mean: np.ndarray, std: np.ndarray, members: list[dict[str, np.ndarray]],
                 meta: dict[str, Any] | None = None) -> None:
        self.features, self.mean, self.std, self.members = features, mean, std, members
        self.meta = dict(meta or {})

    # -- inference ---------------------------------------------------------------------------------
    def _predict_all(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mean) / self.std
        out = []
        for p in self.members:
            h1 = np.maximum(0, x @ p["W1"] + p["b1"])
            h2 = np.maximum(0, h1 @ p["W2"] + p["b2"])
            out.append(1 / (1 + np.exp(-(h2 @ p["W3"] + p["b3"])[:, 0])))
        return np.stack(out)                                     # (members, n)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self._predict_all(np.atleast_2d(x))
        return p.mean(0), p.std(0)

    def score_chunks(self, obs: Observation, chunks: np.ndarray, *, step: int, task_id: str, execution_steps: int,
                     beta: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = state_features(obs.state, step, task_id, obs.images, self.features)
        x = np.stack([np.concatenate([s, chunk_features(c, execution_steps, self.features.prefix_steps)]) for c in chunks])
        mu, sd = self.predict(x)
        return mu - beta * sd, mu, sd

    # -- persistence (no pickle) ---------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {"mean": self.mean, "std": self.std}
        for i, p in enumerate(self.members):
            arrays.update({f"m{i}.{k}": v for k, v in p.items()})
        np.savez(path / "weights.npz", **arrays)
        digest = hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()[:16]
        (path / "critic.json").write_text(json.dumps({"schema": "wc.critic/v0.1", "features": self.features.model_dump(mode="json"),
                                                      "members": len(self.members), "weights_sha256_16": digest,
                                                      **self.meta}, indent=2))
        return digest

    @classmethod
    def load(cls, path: str | Path) -> "ChunkCritic":
        path = Path(path)
        meta = json.loads((path / "critic.json").read_text())
        if meta.get("schema") != "wc.critic/v0.1":
            raise ValueError(f"not a wc.critic/v0.1 directory: {path}")
        with np.load(path / "weights.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        members = [{k: arrays[f"m{i}.{k}"] for k in ("W1", "b1", "W2", "b2", "W3", "b3")} for i in range(meta["members"])]
        return cls(FeatureConfig.model_validate(meta.pop("features")), arrays["mean"], arrays["std"], members, meta)

    # -- training --------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, features: FeatureConfig, *, weights: np.ndarray | None = None,
            members: int = 5, hidden: int = 64, epochs: int = 60, lr: float = 1e-3, l2: float = 1e-4, batch: int = 256,
            seed: int = 0, meta: dict[str, Any] | None = None) -> "ChunkCritic":
        rng = np.random.default_rng(seed)
        mean, std = x.mean(0), x.std(0) + 1e-6
        xn = (x - mean) / std
        w = np.ones(len(x)) if weights is None else weights
        nets = []
        for _ in range(members):                       # bootstrap resampling gives the ensemble its spread
            boot = rng.integers(0, len(x), len(x))
            net = _MLP(x.shape[1], hidden, rng)
            net.fit(xn[boot], y[boot], w[boot], epochs=epochs, lr=lr, l2=l2, batch=batch, rng=rng)
            nets.append(net.p)
        return cls(features, mean, std, nets, meta)


# ---------------------------------------------------------------- agent

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
