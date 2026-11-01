"""policy.knn: retrieval (VINN, Pari et al. 2021): the k demonstration observations nearest to the current one,
their action chunks averaged with softmax(-distance / temperature) weights. No training beyond storing the data."""

from __future__ import annotations

import numpy as np

from .base import ReferencePolicy


class KNNPolicy(ReferencePolicy):
    method = "knn"

    def __init__(self, path: str, execution_steps: int | None = None, k: int | None = None, temperature: float | None = None) -> None:
        super().__init__(path, execution_steps)
        h = self.meta["hyper"]
        self.k, self.temperature = int(k or h.get("k", 5)), float(temperature or h.get("temperature", 0.1))

    def _sample(self, x, rngs, noise_scale):
        d = np.linalg.norm(self.a["obs"] - x, axis=1)
        idx = np.argsort(d)[: self.k]
        w = np.exp(-(d[idx] - d[idx].min()) / self.temperature)
        chunk = np.tensordot(w / w.sum(), self.a["chunks"][idx], axes=1)
        return np.repeat(chunk[None], len(rngs), axis=0)
