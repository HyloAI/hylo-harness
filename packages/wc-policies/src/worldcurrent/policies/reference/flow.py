"""policy.flow: a flow-matching action head (the objective of pi0's and SmolVLA's action experts).

Training: for a demonstration chunk x1 (normalised) and noise x0 ~ N(0, I), x_tau = (1 - tau) x0 + tau x1, and an
MLP v(x_tau, tau, obs) regresses the velocity x1 - x0. Sampling: start from seeded noise (times noise_scale) and
integrate dx/dtau = v from tau = 0 to 1 with Euler steps. Different seeds give different, plausible chunks, so
the policy can be steered (seeded noise, sample-and-select) like the large flow-matching VLAs.
"""

from __future__ import annotations

import numpy as np

from .base import ReferencePolicy
from .nn import MLP


def tau_features(tau: np.ndarray) -> np.ndarray:
    f = np.arange(1, 5)[None] * np.pi * tau[:, None]
    return np.concatenate([np.sin(f), np.cos(f), tau[:, None]], axis=1)          # (B, 9)


class FlowPolicy(ReferencePolicy):
    method = "flow"

    def __init__(self, path: str, execution_steps: int | None = None, steps: int | None = None) -> None:
        super().__init__(path, execution_steps)
        self.net = MLP.from_params({k[4:]: v for k, v in self.a.items() if k.startswith("net.")})
        self.steps = int(steps or self.meta["hyper"].get("integration_steps", 10))

    def _sample(self, x, rngs, noise_scale):
        d = self.chunk * self.action_dim
        z = np.stack([r.standard_normal(d) for r in rngs]) * noise_scale
        obs = np.repeat(x[None], len(rngs), axis=0)
        for i in range(self.steps):
            tau = np.full(len(rngs), i / self.steps)
            z = z + self.net(np.concatenate([z, tau_features(tau), obs], axis=1)) / self.steps
        return self.act_norm.inverse(z.reshape(len(rngs), self.chunk, self.action_dim))
