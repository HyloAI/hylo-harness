"""policy.bc: regression behaviour cloning with action chunking: an MLP from the observation to the next T
actions, trained with mean squared error (the parametric baseline of VINN's "BC-Rep"; chunked like ACT)."""

from __future__ import annotations

import numpy as np

from .base import ReferencePolicy
from .nn import MLP


class BCPolicy(ReferencePolicy):
    method = "bc"

    def __init__(self, path: str, execution_steps: int | None = None) -> None:
        super().__init__(path, execution_steps)
        self.net = MLP.from_params({k[4:]: v for k, v in self.a.items() if k.startswith("net.")})

    def _sample(self, x, rngs, noise_scale):
        z = self.net(x[None])[0].reshape(self.chunk, self.action_dim)
        return np.repeat(self.act_norm.inverse(z)[None], len(rngs), axis=0)
