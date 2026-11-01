"""Learners for the reference policies (RFC-0004): development runs -> demonstrations -> sealed policy artifact.

    {"learner": "policy.bc", "inputs": ["runs/demos"], "out": "artifacts/bc",
     "params": {"chunk": 8, "epochs": 150, "hidden": 128, "execution_steps": 4}}

Common params: chunk (actions per prediction, default 8), state_keys (default: every small state entry),
statuses (episodes to imitate, default ["success"]), max_episodes, execution_steps, seed. Only development splits
are read (core.artifact.development_runs), and the artifact's role is "behaviour": only the gate activates it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hylo.core.artifact import ArtifactManifest, Provenance, check_splits, development_runs, write_manifest

from . import base
from .flow import tau_features
from .nn import MLP, train
from .tokens import softmax


class _Learner:
    method = "reference"
    defaults: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"hylo.policies.reference.{self.method}@0.1"

    def fit(self, job) -> ArtifactManifest:
        p = {**self.defaults, "chunk": 8, "seed": 0, **job.params}
        sources = development_runs(job.inputs, check_splits(job.splits))
        demos = base.load_demos(sources, chunk=int(p["chunk"]), keys=p.get("state_keys"),
                                statuses=tuple(p.get("statuses", ["success"])), max_episodes=p.get("max_episodes"))
        on, an = base.Normalizer.fit(demos.obs), base.Normalizer.fit(demos.chunks.reshape(-1, demos.chunks.shape[2]))
        x = on(demos.obs)
        y = an(demos.chunks)                                             # (N, T, A) normalised
        arrays, metrics = self._fit(x, y, demos, p)
        arrays.update(obs_mean=on.mean, obs_std=on.std, act_mean=an.mean, act_std=an.std)
        base.save(job.out, self.method, demos, arrays, {k: v for k, v in p.items() if k not in ("state_keys", "statuses")})
        return write_manifest(job.out, kind="policy_params", role="behaviour", name=f"policy.{self.method}",
                              metrics={"episodes": demos.episodes, "samples": len(x), "obs_dim": int(x.shape[1]),
                                       "chunk": int(y.shape[1]), "action_dim": int(y.shape[2]), **metrics},
                              provenance=Provenance(builder=self.name, sources=[r for _, r in sources], params=p))

    def _fit(self, x: np.ndarray, y: np.ndarray, demos: base.Demos, p: dict[str, Any]) -> tuple[dict, dict]:
        raise NotImplementedError


def _net_arrays(net: MLP) -> dict[str, np.ndarray]:
    return {f"net.{k}": v for k, v in net.p.items()}


class KNNLearner(_Learner):
    """Stores the normalised observations and their chunks (in action units)."""

    method, defaults = "knn", {"k": 5, "temperature": 0.1}

    def _fit(self, x, y, demos, p):
        return {"obs": x, "chunks": demos.chunks}, {}


class BCLearner(_Learner):
    method, defaults = "bc", {"hidden": 128, "layers": 2, "epochs": 150, "lr": 1e-3}

    def _fit(self, x, y, demos, p):
        yf = y.reshape(len(y), -1)
        net = MLP([x.shape[1], *[int(p["hidden"])] * int(p["layers"]), yf.shape[1]], np.random.default_rng(p["seed"]))

        def loss_grad(net, idx, rng):
            out, acts = net.forward(x[idx])
            d = out - yf[idx]
            return float((d ** 2).mean()), net.backward(acts, 2 * d / d.size)

        loss = train(net, len(x), loss_grad, epochs=int(p["epochs"]), lr=float(p["lr"]), seed=int(p["seed"]))
        return _net_arrays(net), {"train_mse": loss}


class FlowLearner(_Learner):
    method, defaults = "flow", {"hidden": 256, "layers": 3, "epochs": 300, "lr": 1e-3, "integration_steps": 10}

    def _fit(self, x, y, demos, p):
        yf = y.reshape(len(y), -1)
        d = yf.shape[1]
        net = MLP([d + 9 + x.shape[1], *[int(p["hidden"])] * int(p["layers"]), d], np.random.default_rng(p["seed"]))

        def loss_grad(net, idx, rng):
            x1, x0 = yf[idx], rng.standard_normal((len(idx), d))
            tau = rng.random(len(idx))
            xt = (1 - tau[:, None]) * x0 + tau[:, None] * x1
            out, acts = net.forward(np.concatenate([xt, tau_features(tau), x[idx]], axis=1))
            diff = out - (x1 - x0)
            return float((diff ** 2).mean()), net.backward(acts, 2 * diff / diff.size)

        loss = train(net, len(x), loss_grad, epochs=int(p["epochs"]), lr=float(p["lr"]), seed=int(p["seed"]))
        return _net_arrays(net), {"train_flow_mse": loss}


class TokenLearner(_Learner):
    method, defaults = "tokens", {"bins": 21, "hidden": 128, "layers": 2, "epochs": 150, "lr": 1e-3}

    def _fit(self, x, y, demos, p):
        acts = demos.chunks                                              # bins are over action units
        low, high = acts.reshape(-1, acts.shape[2]).min(0), acts.reshape(-1, acts.shape[2]).max(0)
        high = np.where(high - low < 1e-6, low + 1e-6, high)
        K = int(p["bins"])
        target = np.clip(((acts - low) / (high - low) * K).astype(int), 0, K - 1)   # (N, T, A)
        T, A = acts.shape[1], acts.shape[2]
        net = MLP([x.shape[1], *[int(p["hidden"])] * int(p["layers"]), T * A * K], np.random.default_rng(p["seed"]))
        onehot = np.eye(K)

        def loss_grad(net, idx, rng):
            out, a = net.forward(x[idx])
            prob = softmax(out.reshape(len(idx), T, A, K))
            t = onehot[target[idx]]
            loss = float(-(t * np.log(prob + 1e-12)).sum(-1).mean())
            return loss, net.backward(a, ((prob - t) / (len(idx) * T * A)).reshape(len(idx), -1))

        loss = train(net, len(x), loss_grad, epochs=int(p["epochs"]), lr=float(p["lr"]), seed=int(p["seed"]))
        acc = float((net(x).reshape(len(x), T, A, K).argmax(-1) == target).mean())
        return {**_net_arrays(net), "low": low, "high": high}, {"train_ce": loss, "train_bin_accuracy": acc}
