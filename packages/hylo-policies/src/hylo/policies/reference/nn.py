"""A small multilayer perceptron with manual gradients and Adam (numpy). Enough for low-dimensional state."""

from __future__ import annotations

import numpy as np


class MLP:
    """tanh hidden layers, linear output. Parameters are a flat dict of arrays (saved with np.savez)."""

    def __init__(self, sizes: list[int], rng: np.random.Generator | None = None) -> None:
        self.n = len(sizes) - 1
        rng = rng or np.random.default_rng(0)
        self.p = {}
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(1.0 / a) * (0.1 if i == self.n - 1 else 1.0)
            self.p[f"W{i}"], self.p[f"b{i}"] = rng.normal(0, scale, (a, b)), np.zeros(b)

    @classmethod
    def from_params(cls, params: dict[str, np.ndarray]) -> "MLP":
        net = cls.__new__(cls)
        net.p, net.n = params, sum(1 for k in params if k.startswith("W"))
        return net

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        for i in range(self.n):
            z = acts[-1] @ self.p[f"W{i}"] + self.p[f"b{i}"]
            acts.append(np.tanh(z) if i < self.n - 1 else z)
        return acts[-1], acts

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, acts: list[np.ndarray], dout: np.ndarray) -> dict[str, np.ndarray]:
        g, d = {}, dout
        for i in reversed(range(self.n)):
            g[f"W{i}"], g[f"b{i}"] = acts[i].T @ d, d.sum(0)
            if i:
                d = (d @ self.p[f"W{i}"].T) * (1 - acts[i] ** 2)
        return g


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3) -> None:
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            params[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def train(net: MLP, n: int, loss_grad, *, epochs: int, batch: int = 256, lr: float = 1e-3, seed: int = 0) -> float:
    """Minibatch Adam. `loss_grad(net, idx, rng)` returns (loss, grads) for the samples `idx`. Returns the last
    epoch's mean loss."""
    rng, opt = np.random.default_rng(seed), Adam(net.p, lr)
    last = float("nan")
    for _ in range(epochs):
        order, losses = rng.permutation(n), []
        for s in range(0, n, batch):
            loss, grads = loss_grad(net, order[s:s + batch], rng)
            opt.step(net.p, grads)
            losses.append(loss)
        last = float(np.mean(losses))
    return last
