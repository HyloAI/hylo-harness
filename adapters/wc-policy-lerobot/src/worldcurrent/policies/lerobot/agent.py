"""A LeRobot policy (SmolVLA, pi0, pi0.5, ...) as a worldcurrent Agent.

Meant to run behind an AgentServer in a venv that has `lerobot` + torch; the main process talks to it
through RemoteAgent. Observation keys are mapped explicitly (no positional guessing): camera names to
the policy's image features, and an ordered list of state keys concatenated into observation.state.
The policy returns a whole action chunk; the kernel's ChunkExecutor runs `n_action_steps` of it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from worldcurrent.core.agent import AgentBase, AgentSpec, EpisodeContext
from worldcurrent.core.types import ActionChunk, ActionSemantics, Observation

LIBERO_CAMERAS = {"agentview": "observation.images.camera1", "wrist": "observation.images.camera2"}
LIBERO_STATE = ("eef_pos", "eef_axis_angle", "gripper_qpos")


class LeRobotAgent(AgentBase):
    def __init__(self, repo_id: str = "lerobot/smolvla_libero", device: str = "cpu",
                 camera_map: dict[str, str] | None = None, state_keys: list[str] | tuple[str, ...] = LIBERO_STATE,
                 empty_cameras: int | None = 1, n_action_steps: int | None = None, num_threads: int | None = None,
                 weights_license: str | None = None,
                 action_semantics: dict[str, Any] | None = None) -> None:
        import torch

        if num_threads:
            torch.set_num_threads(num_threads)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        t0 = time.perf_counter()
        cfg = PreTrainedConfig.from_pretrained(repo_id)
        cfg.device = device
        if empty_cameras is not None and hasattr(cfg, "empty_cameras"):
            cfg.empty_cameras = empty_cameras   # LeRobot's LIBERO CI evaluates smolvla_libero with one empty camera
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(repo_id, config=cfg).to(device).eval()
        overrides = {"device_processor": {"device": device}}
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=repo_id, preprocessor_overrides=overrides,
                                                       postprocessor_overrides=overrides)
        self.torch, self.cfg, self.repo_id, self.device = torch, cfg, repo_id, device
        self.camera_map = dict(camera_map or LIBERO_CAMERAS)
        self.state_keys = tuple(state_keys)
        self.weights_license = weights_license or f"per checkpoint: see the model card of {repo_id}"
        self.semantics = ActionSemantics(**(action_semantics or {"control_mode": "eef_delta_pose", "frame": "base",
                                                                 "rotation_repr": "axis_angle",
                                                                 "gripper": "binary_close_positive"}))
        self.load_s = time.perf_counter() - t0

    def spec(self) -> AgentSpec:
        return AgentSpec(name=f"lerobot.{self.cfg.type}", version="0.1", required_state=self.state_keys,
                         required_cameras=tuple(self.camera_map), action_semantics=self.semantics,
                         action_dim=int(self.cfg.output_features["action"].shape[0]),
                         components={"policy": f"{self.repo_id}@{self.cfg.type}",
                                     "execution_steps": str(self.cfg.n_action_steps)},
                         weights_license=self.weights_license)

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.policy.reset()
        for proc in (self.pre, self.post):
            if hasattr(proc, "reset"):
                proc.reset()

    # -- ChunkSampler (used by worldcurrent.agents.steering.SteeredAgent) -------------------------------
    def default_execution_steps(self) -> int:
        return int(self.cfg.n_action_steps)

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        """(N, T, A) unnormalized chunks; one batched forward pass, noise drawn from the given seeds."""
        torch = self.torch
        n = len(seeds) if seeds else 1
        batch: dict[str, Any] = {
            key: torch.from_numpy(np.ascontiguousarray(observation.images[cam])).permute(2, 0, 1).float().div(255.0)
            .unsqueeze(0).expand(n, -1, -1, -1).contiguous()
            for cam, key in self.camera_map.items()}
        state = np.concatenate([np.asarray(observation.state[k], dtype=np.float32).reshape(-1) for k in self.state_keys])
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).expand(n, -1).contiguous()
        batch["task"] = [observation.instruction or ""] * n
        noise = None
        if seeds is not None or noise_scale != 1.0:
            shape = (int(self.cfg.chunk_size), int(getattr(self.cfg, "max_action_dim", 32)))
            gens = [torch.Generator().manual_seed(int(s)) for s in (seeds or [int(torch.randint(0, 2**31 - 1, (1,)))])]
            noise = torch.stack([torch.randn(shape, generator=g) for g in gens]) * float(noise_scale)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch), noise=noise)   # (N, T, A), normalized
            actions = self.post(chunk)                                                # unnormalized
        return actions.detach().cpu().numpy().astype(np.float64)

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        actions = self.sample(observation, None)[0]
        return ActionChunk(actions=actions, execution_steps=int(self.cfg.n_action_steps),
                           provenance={"model": self.repo_id, "latency_s": round(time.perf_counter() - t0, 3)})


def make_steered(steering: dict[str, Any] | None = None, **lerobot_kwargs: Any):
    """A LeRobot policy wrapped in SteeredAgent (seeded noise, sample-and-select, runtime recovery rules)."""
    from worldcurrent.agents.steering import SteeredAgent

    return SteeredAgent(LeRobotAgent(**lerobot_kwargs), steering)
