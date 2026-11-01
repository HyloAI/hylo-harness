"""LeRobot flow policies with experience memory (RFC-0003 M1).

PriorLeRobotSampler extends LeRobotAgent with the PriorSampler protocol:
- the retrieval key is the policy's own visual tokens (SigLIP embeddings of each real camera, 2x2 grid-pooled,
  concatenated, L2-normalised), captured from the forward pass the policy runs anyway;
- `prior_fn(key)` is called between the prefix computation and the denoising loop; if it returns a prior, the
  Euler integration starts at x_t0 = (1 - t0) * normalise(prior) + t0 * noise and runs t0 -> 0 with the policy's
  unchanged velocity field. With no prior (or t0 = 1) the result is bit-identical to LeRobotAgent.sample.
Implemented by wrapping `embed_image` and `euler_integrate` for the duration of one call; LeRobot is not modified.

    agent: {type: remote, params: {ref: worldcurrent.policies.lerobot.experience_agent:make_experience_agent,
            params: {repo_id: lerobot/smolvla_libero, memory: {type: action_bank, params: {path: ...}}, ...}}}
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from worldcurrent.core.experience import PriorFn
from worldcurrent.core.types import Observation

from .agent import LeRobotAgent

_PATCH_LOCK = threading.Lock()          # the integrator patch is module-global


class _KeyOnly(Exception):
    def __init__(self, key: np.ndarray) -> None:
        self.key = key


class PriorLeRobotSampler(LeRobotAgent):
    def __init__(self, *args: Any, key_cameras: int = 2, key_grid: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.key_cameras, self.key_grid = key_cameras, key_grid
        norm = getattr(self.cfg, "normalization_mapping", {}).get("ACTION")
        if str(getattr(norm, "value", norm)) != "MEAN_STD":
            raise ValueError(f"prior sampling needs MEAN_STD action normalisation, policy has {norm}")
        from safetensors.numpy import load_file

        from .selftrain import pretrained_dir

        files = sorted(pretrained_dir(self.repo_id).glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
        stats = load_file(str(files[0]))
        self._act_mean = stats["action.mean"].astype(np.float32)
        self._act_std = stats["action.std"].astype(np.float32)

    # -- key ----------------------------------------------------------------------------------------
    def _pool(self, emb) -> np.ndarray:
        """(B, n, D) image tokens on a sqrt(n) x sqrt(n) grid -> (B, g*g*D) averaged over g x g cells."""
        b, n, d = emb.shape
        side = int(round(n ** 0.5))
        if side * side != n or side % self.key_grid:
            return emb.float().mean(1).cpu().numpy()
        grid = emb.float().reshape(b, self.key_grid, side // self.key_grid, self.key_grid, side // self.key_grid, d)
        return grid.mean((2, 4)).reshape(b, -1).cpu().numpy()

    def retrieval_key(self, observation: Observation) -> np.ndarray:
        def stop(key: np.ndarray):
            raise _KeyOnly(key)
        try:
            self.sample(observation, [0], prior_fn=stop)
        except _KeyOnly as k:
            return k.key
        raise RuntimeError("the policy forward pass never reached the integrator")

    # -- sampling -----------------------------------------------------------------------------------
    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0,
               prior_fn: PriorFn | None = None) -> np.ndarray:
        if prior_fn is None:
            return super().sample(observation, seeds, noise_scale)
        from lerobot.policies.smolvla import modeling_smolvla as ms

        torch = self.torch
        vlm = self.policy.model.vlm_with_expert
        captured: list = []
        original_embed, original_euler = vlm.embed_image, ms.euler_integrate

        def embed_image(img):
            out = original_embed(img)
            captured.append(out)
            return out

        def euler_from_prior(denoise_fn, noise, num_steps, *, rtc_enabled=False, **kw):
            if rtc_enabled:
                raise RuntimeError("prior sampling is not combined with real-time chunking")
            key = np.concatenate([self._pool(e[:1]) for e in captured[: self.key_cameras]], axis=1)[0]
            key = key / (np.linalg.norm(key) + 1e-12)
            decision = prior_fn(key)
            t0 = float(decision.t0)
            if decision.noise is not None and decision.noise_weight > 0:          # noise-space steering
                w = float(decision.noise_weight)
                eps = torch.from_numpy(np.asarray(decision.noise, dtype=np.float32)).to(noise.device, noise.dtype)
                eps = eps[: noise.shape[1], : noise.shape[2]].unsqueeze(0)
                return original_euler(denoise_fn, (w ** 0.5) * eps + ((1 - w) ** 0.5) * noise, num_steps,
                                      rtc_enabled=rtc_enabled, **kw)
            if decision.prior is None or t0 >= 1.0 - 1e-6:
                return original_euler(denoise_fn, noise, num_steps, rtc_enabled=rtc_enabled, **kw)
            T, D = noise.shape[1], noise.shape[2]
            p = np.asarray(decision.prior, dtype=np.float32)
            p = np.concatenate([p, np.repeat(p[-1:], max(0, T - len(p)), 0)])[:T]          # pad by holding the last action
            pn = np.zeros((T, D), dtype=np.float32)
            a = min(p.shape[1], len(self._act_mean))
            pn[:, :a] = (p[:, :a] - self._act_mean[:a]) / self._act_std[:a]
            prior_t = torch.from_numpy(pn).to(noise.device, noise.dtype).unsqueeze(0)
            x = (1.0 - t0) * prior_t + t0 * noise
            steps = max(1, int(round(t0 * num_steps)))
            dt = -t0 / steps
            for i in range(steps):
                tt = torch.tensor(t0 + i * dt, dtype=torch.float32, device=noise.device).expand(noise.shape[0])
                x = x + dt * denoise_fn(x, tt)
            return x

        with _PATCH_LOCK:
            vlm.embed_image = embed_image
            ms.euler_integrate = euler_from_prior
            try:
                return super().sample(observation, seeds, noise_scale)
            finally:
                ms.euler_integrate = original_euler
                del vlm.embed_image                              # back to the bound method


def seed_noise(seed: int, shape: tuple[int, int], noise_scale: float = 1.0) -> np.ndarray:
    """The exact initial noise LeRobotAgent.sample draws for a seed (torch's generator, so this needs torch)."""
    import torch

    return (torch.randn(shape, generator=torch.Generator().manual_seed(int(seed))) * float(noise_scale)).numpy()


def make_noise_agent(memory: dict[str, Any], noise_steer: dict[str, Any] | None = None,
                     steering: dict[str, Any] | None = None, **lerobot_kwargs: Any):
    """LeRobot policy + SteeredAgent + noise prior retrieved from successful episodes (worldcurrent.agents.noise_steer)."""
    from worldcurrent.agents.noise_steer import NoiseSteeredAgent

    return NoiseSteeredAgent(PriorLeRobotSampler(**lerobot_kwargs), steering, memory, noise_steer)


def make_experience_agent(memory: dict[str, Any], memory_steer: dict[str, Any] | None = None,
                          steering: dict[str, Any] | None = None, **lerobot_kwargs: Any):
    """LeRobot policy + SteeredAgent + action-memory prior (worldcurrent.agents.memory_steer)."""
    from worldcurrent.agents.memory_steer import MemorySteeredAgent

    return MemorySteeredAgent(PriorLeRobotSampler(**lerobot_kwargs), steering, memory, memory_steer)
