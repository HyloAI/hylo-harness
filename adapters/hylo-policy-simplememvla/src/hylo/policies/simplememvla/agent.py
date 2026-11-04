"""Optional native Hylo agent for the released RoboMME SimpleMemVLA checkpoint."""
from __future__ import annotations

import hashlib

import numpy as np

from hylo.core.agent import AgentSpec
from hylo.core.types import ActionChunk, ActionSemantics


class SimpleMemVLAAgent:
    def __init__(self, checkpoint: str | None = None, *, runtime=None, camera_map=None,
                 joint_key="joint_pos", gripper_key="gripper_pos", execution_steps: int = 16,
                 weights_license="MIT (released SimpleMemVLA RoboMME checkpoint)"):
        if execution_steps < 1:
            raise ValueError("execution_steps must be positive")
        if runtime is None:
            if checkpoint is None:
                raise ValueError("checkpoint or injected runtime required")
            from .runtime import SimpleMemVLARuntime
            runtime = SimpleMemVLARuntime(checkpoint)
        self.runtime = runtime
        self.camera_map = dict(camera_map or {"front": "front", "wrist": "wrist"})
        self.image_keys = tuple(runtime.config.image_keys)
        if set(self.camera_map) != {k.split(".")[-1] for k in self.image_keys}:
            raise ValueError("camera_map must map every checkpoint camera tag to a Hylo camera")
        if runtime.config.state_dim != 8 or runtime.config.action_dim != 8:
            raise ValueError("this agent maps the released RoboMME seven-joint, one-gripper interface only")
        self.joint_key, self.gripper_key = joint_key, gripper_key
        self.execution_steps = min(execution_steps, runtime.config.action_horizon)
        self.weights_license = weights_license
        self.session = None

    def spec(self):
        return AgentSpec(name="simplememvla", version="0.1", required_state=(self.joint_key, self.gripper_key),
                         required_cameras=tuple(self.camera_map.values()), action_dim=8,
                         action_semantics=ActionSemantics(control_mode="joint_pos"),
                         weights_license=self.weights_license,
                         components={"memory": "simplememvla.video/v1", "policy": "frozen-simplememvla"})

    def reset(self, ctx):
        semantics = ctx.env_spec.action.semantics
        if (ctx.env_spec.action.shape != (8,) or semantics is None or semantics.control_mode != "joint_pos"
                or not np.isclose(ctx.env_spec.control_hz, self.runtime.config.control_frequency_hz)):
            raise ValueError("SimpleMemVLA requires matching RoboMME joint-position action layout and control rate")
        if self.session is not None:
            self.session.reset()
        self.session = self.runtime.new_session()
        self.session.reset()
        self.session.memory.episode_id = ctx.episode_id
        self._base_seed = ctx.seed
        self._step, self._observed_step, self._calls = 0, -1, 0

    def _images(self, observation):
        return {key: observation.images[self.camera_map[key.split(".")[-1]]] for key in self.image_keys}

    def _observe_current(self, observation):
        if self.session is None:
            raise RuntimeError("reset before using the agent")
        if self._observed_step != self._step:
            self.session.observe(self._images(observation))
            self._observed_step = self._step

    def observe_history(self, observations):
        """Optional reset demonstration prefix, excluding the current act() observation.

        Environment adapters must deliver all prefix frames in chronological native ticks.
        Only RGB is ingested; reward, task IDs, oracle state and success are not forwarded.
        """
        if self.session is None or self._observed_step >= 0:
            raise RuntimeError("demonstration prefix is accepted only after reset, before the first observation")
        for observation in observations:
            self.session.observe(self._images(observation))

    def act(self, observation):
        self._observe_current(observation)
        joints = np.asarray(observation.state[self.joint_key], dtype=np.float32).reshape(-1)
        gripper = np.asarray(observation.state[self.gripper_key], dtype=np.float32).reshape(-1)
        if joints.shape != (7,) or gripper.size not in (1, 2):
            raise ValueError("expected seven arm joints and one/two finger positions")
        state = np.concatenate([joints, gripper[:1]])
        # Stable per-decision seeds, independent of other episodes sharing this runtime.
        seed = int.from_bytes(hashlib.blake2b(f"{self._base_seed}:{self._calls}".encode(), digest_size=8).digest(), "little") % (2**63)
        actions, subtask = self.runtime.infer_batch([self.session], [observation.instruction], [state], seed=seed)[0]
        if actions.shape != (self.runtime.config.action_horizon, 8) or not np.isfinite(actions).all():
            raise RuntimeError("invalid SimpleMemVLA action chunk")
        self._calls += 1
        return ActionChunk(actions=np.asarray(actions, dtype=np.float64), execution_steps=self.execution_steps,
                           provenance={"source": "simplememvla", "memory": "simplememvla.video", "subtask": subtask,
                                       "seed": seed, "history": self.session.memory.snapshot()})

    def on_step(self, result):
        self._step += 1
        self._observe_current(result.observation)

    def on_episode_end(self, outcome):
        if self.session is not None:
            self.session.reset()
            self.session = None
