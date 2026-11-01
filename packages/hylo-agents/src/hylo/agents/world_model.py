"""World model (RFC-0004 W2): an action-conditioned model of the environment, learned from our own traces.

StateWorldModel predicts the next proprioceptive / object state from (state, action) with an ensemble of small MLPs
(their disagreement is the model's uncertainty), and the probability that the task is complete in a state.
WorldModelEnvironment wraps it behind the Environment protocol, so the unchanged kernel can run an agent inside
the model: episodes start from initial states recorded in development runs and end when the model says the task is
complete or the step budget runs out.

It is an instrument. Its use in the improvement loop is pre-screening (protocol.prescreen): candidates the model
predicts to be clearly worse than the parent are dropped before any real rollout; the gate never sees model
results. This version models low-dimensional state, so it can host agents that act on state (MuJoCo / test
baselines, planners); a VLA acting on camera images needs a video world model (GPU), which plugs into the same
protocol later.

    env: {type: world_model, params: {path: "artifact:world_model@<digest>"}}     # trained by the "world_model.state"
                                                                        # learner (hylo.learn.world_model)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.types import Action, EnvSpec, Observation, StepResult


class _Reg:
    """Two-layer tanh MLP, linear output, MSE, Adam (numpy)."""

    def __init__(self, n_in: int, n_out: int, hidden: int, rng: np.random.Generator) -> None:
        g = lambda a, b: rng.normal(0, np.sqrt(1.0 / a), size=(a, b))  # noqa: E731
        self.p = {"W1": g(n_in, hidden), "b1": np.zeros(hidden), "W2": g(hidden, hidden), "b2": np.zeros(hidden),
                  "W3": g(hidden, n_out) * 0.1, "b3": np.zeros(n_out)}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(x @ self.p["W1"] + self.p["b1"])
        h2 = np.tanh(h1 @ self.p["W2"] + self.p["b2"])
        return h2 @ self.p["W3"] + self.p["b3"]

    def fit(self, x, y, *, epochs: int, lr: float = 2e-3, batch: int = 256, rng: np.random.Generator) -> None:
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(w) for k, w in self.p.items()}
        t = 0
        for _ in range(epochs):
            for s in range(0, len(x), batch):
                idx = rng.permutation(len(x))[:batch] if len(x) > batch else np.arange(len(x))
                xb, yb = x[idx], y[idx]
                h1 = np.tanh(xb @ self.p["W1"] + self.p["b1"])
                h2 = np.tanh(h1 @ self.p["W2"] + self.p["b2"])
                d3 = 2 * ((h2 @ self.p["W3"] + self.p["b3"]) - yb) / len(xb)
                g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
                d2 = (d3 @ self.p["W3"].T) * (1 - h2 ** 2)
                g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
                d1 = (d2 @ self.p["W2"].T) * (1 - h1 ** 2)
                g["W1"], g["b1"] = xb.T @ d1, d1.sum(0)
                t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)


def _vec(state: dict[str, Any], keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(state[k], float).reshape(-1) for k in keys])


class StateWorldModel:
    def __init__(self, keys: list[str], dims: list[int], mean: dict[str, np.ndarray], std: dict[str, np.ndarray],
                 dyn: list[dict[str, np.ndarray]], done: list[dict[str, np.ndarray]], meta: dict[str, Any]) -> None:
        self.keys, self.dims, self.mean, self.std, self.meta = keys, dims, mean, std, meta
        self._dyn = [self._wrap(p) for p in dyn]
        self._done = [self._wrap(p) for p in done]

    @staticmethod
    def _wrap(p: dict[str, np.ndarray]) -> _Reg:
        r = _Reg.__new__(_Reg)
        r.p = p
        return r

    def split(self, v: np.ndarray) -> dict[str, np.ndarray]:
        out, i = {}, 0
        for k, n in zip(self.keys, self.dims):
            out[k] = v[i:i + n].astype(np.float32)
            i += n
        return out

    def predict(self, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, float]:
        x = (np.concatenate([s, a]) - self.mean["x"]) / self.std["x"]
        outs = np.stack([m(x[None])[0] for m in self._dyn]) * self.std["dy"] + self.mean["dy"]
        return s + outs.mean(0), float(np.linalg.norm(outs.std(0)))

    def predict_batch(self, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """predict() for (B, S) states and (B, A) actions: next states (B, S) and ensemble disagreement (B,)."""
        x = (np.concatenate([s, a], axis=1) - self.mean["x"]) / self.std["x"]
        outs = np.stack([m(x) for m in self._dyn]) * self.std["dy"] + self.mean["dy"]
        return s + outs.mean(0), np.linalg.norm(outs.std(0), axis=-1)

    def p_done_batch(self, s: np.ndarray) -> np.ndarray:
        z = np.mean([m((s - self.mean["s"]) / self.std["s"])[:, 0] for m in self._done], axis=0)
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def p_done(self, s: np.ndarray) -> float:
        x = (s - self.mean["s"]) / self.std["s"]
        z = np.mean([m(x[None])[0, 0] for m in self._done])
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    # -- persistence (npz + json, no pickle) ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {f"norm.{k}": v for k, v in {**{f"mean_{k}": v for k, v in self.mean.items()},
                                               **{f"std_{k}": v for k, v in self.std.items()}}.items()}
        for name, ms in (("dyn", self._dyn), ("done", self._done)):
            for i, m in enumerate(ms):
                arrays.update({f"{name}{i}.{k}": v for k, v in m.p.items()})
        np.savez(path / "world_model.npz", **arrays)
        (path / "world_model.json").write_text(json.dumps({"schema": "hylo.world_model/v0.1", "keys": self.keys, "dims": self.dims,
                                                           "members": len(self._dyn), **self.meta}, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "StateWorldModel":
        path = Path(path)
        meta = json.loads((path / "world_model.json").read_text())
        with np.load(path / "world_model.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        mean = {k[len("norm.mean_"):]: v for k, v in a.items() if k.startswith("norm.mean_")}
        std = {k[len("norm.std_"):]: v for k, v in a.items() if k.startswith("norm.std_")}
        P = ("W1", "b1", "W2", "b2", "W3", "b3")
        dyn = [{p: a[f"dyn{i}.{p}"] for p in P} for i in range(meta["members"])]
        done = [{p: a[f"done{i}.{p}"] for p in P} for i in range(meta["members"])]
        keys, dims, members = meta.pop("keys"), meta.pop("dims"), meta.pop("members")
        del members
        return cls(keys, dims, mean, std, dyn, done, meta)


class WorldModelEnvironment:
    """Environment plugin "world_model": the kernel runs an agent inside a learned StateWorldModel."""

    def __init__(self, path: str, done_threshold: float = 0.5, init: str = "recorded") -> None:
        self.path = path
        self.model = StateWorldModel.load(path)
        if not self.model.meta.get("env_spec"):
            raise ValueError("this world model was trained without the real env's spec (learner param `env`)")
        base = EnvSpec.model_validate(self.model.meta["env_spec"])
        self._spec = base.model_copy(update={"name": f"world_model[{base.name}]",
                                             "capabilities": base.capabilities.model_copy(update={"privileged_state": False, "snapshot": False})})
        self.done_threshold, self._s, self._t = done_threshold, None, 0
        self._inits = self.model.meta.get("inits", {})
        self.uncertainty: list[float] = []

    def spec(self) -> EnvSpec:
        return self._spec

    def _obs(self, instruction: str | None) -> Observation:
        return Observation(state=self.model.split(self._s), instruction=instruction, time=float(self._t))

    def reset(self, task) -> Observation:
        by_task = self._inits.get(task.spec.id) or next(iter(self._inits.values()), {})
        if not by_task:
            raise ValueError("no recorded initial states in this world model")
        init = by_task.get(str(task.seed))
        if init is None:                                            # unseen seed: a deterministic recorded start
            seeds = sorted(by_task)
            init = by_task[seeds[int(hashlib.sha256(str(task.seed).encode()).hexdigest(), 16) % len(seeds)]]
        self._s, self._t, self._instr = np.asarray(init, float), 0, task.spec.instruction
        self.uncertainty = []
        return self._obs(self._instr)

    def step(self, action: Action) -> StepResult:
        self._s, unc = self.model.predict(self._s, np.asarray(action.data, float).reshape(-1))
        self._t += 1
        self.uncertainty.append(unc)
        p = self.model.p_done(self._s)
        done = p >= self.done_threshold
        return StepResult(observation=self._obs(self._instr), reward=1.0 if done else 0.0, terminated=done, truncated=False,
                          success=done, info={"p_done": round(p, 4), "uncertainty": round(unc, 5)})

    def close(self) -> None:
        pass
