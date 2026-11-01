"""policy.tokens: tokenized actions. Each action dimension at each chunk step is one of `bins` uniform bins over
the demonstrated range (RT-1 and OpenVLA discretize to 256), predicted by a categorical head and sampled.

The heads are independent given the observation (RT-1 style), not decoded autoregressively as in OpenVLA or
pi0-FAST. `noise_scale` is the sampling temperature; greedy=True takes the most likely bin.
"""

from __future__ import annotations

import numpy as np

from .base import ReferencePolicy
from .nn import MLP


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class TokenPolicy(ReferencePolicy):
    method = "tokens"

    def __init__(self, path: str, execution_steps: int | None = None, greedy: bool = False) -> None:
        super().__init__(path, execution_steps)
        self.net = MLP.from_params({k[4:]: v for k, v in self.a.items() if k.startswith("net.")})
        self.bins, self.greedy = int(self.meta["hyper"]["bins"]), greedy
        self.low, self.high = self.a["low"], self.a["high"]

    def centers(self) -> np.ndarray:                                  # (A, bins)
        e = np.linspace(0, 1, self.bins + 1)
        return self.low[:, None] + (self.high - self.low)[:, None] * ((e[:-1] + e[1:]) / 2)[None]

    def _sample(self, x, rngs, noise_scale):
        logits = self.net(x[None])[0].reshape(self.chunk, self.action_dim, self.bins)
        centers = self.centers()
        if self.greedy:
            idx = logits.argmax(-1)
            return np.repeat(np.take_along_axis(centers[None], idx[..., None], -1)[..., 0][None], len(rngs), axis=0)
        p = softmax(logits / max(noise_scale, 1e-3))
        out = []
        for r in rngs:
            u = r.random(p.shape[:2])[..., None]
            idx = (p.cumsum(-1) < u).sum(-1).clip(0, self.bins - 1)
            out.append(np.take_along_axis(centers[None], idx[..., None], -1)[..., 0])
        return np.stack(out)
