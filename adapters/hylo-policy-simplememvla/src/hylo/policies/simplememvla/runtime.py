"""Shared frozen upstream model, with separate Hylo memory sessions per episode."""
from contextlib import nullcontext
from pathlib import Path
import threading

import numpy as np

from .bridge import memory_policy_type

_INFERENCE_LOCK = threading.RLock()


class SimpleMemVLARuntime:
    def __init__(self, checkpoint: str, *, compute_dtype="bfloat16", attention="sdpa",
                 num_denoising_steps: int = 10, temperature: float = 1.0, max_subtask_tokens: int = 64):
        if not Path(checkpoint).is_dir():
            raise ValueError("checkpoint must be an existing local directory; no implicit download")
        if num_denoising_steps < 1 or max_subtask_tokens < 1 or not np.isfinite(temperature) or temperature < 0:
            raise ValueError("invalid inference parameters")
        try:
            import torch
            from transformers import AutoProcessor
            from robomme_sim.policy import RoboMMEPolicy, get_vla
            from robomme_sim.batched_policy import BatchedEvalPolicy
        except ImportError as exc:
            raise ImportError("SimpleMemVLA runtime requires the pinned upstream source on PYTHONPATH and its inference dependencies; see adapter README") from exc
        self.model, self.unnormalize, self.normalize = get_vla(checkpoint, compute_dtype, attention)
        self.model.requires_grad_(False)
        self.model.eval()
        self.config = self.model.config
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                                       padding_side="right", model_max_length=8192)
        self._policy_type = memory_policy_type(RoboMMEPolicy)
        self.num_denoising_steps, self.temperature, self.max_subtask_tokens = num_denoising_steps, temperature, max_subtask_tokens
        self.batched = BatchedEvalPolicy(self.model, self.processor, self.unnormalize, self.normalize,
                                        num_denoising_steps=num_denoising_steps, eval_temperature=temperature,
                                        max_subtask_tokens=max_subtask_tokens)
        self.device = next(self.model.parameters()).device
        self.torch = torch

    def new_session(self):
        return self._policy_type(vla=self.model, processor=self.processor, unnormalize_action=self.unnormalize,
                                 normalize_state=self.normalize, num_denoising_steps=self.num_denoising_steps,
                                 temperature=self.temperature, max_subtask_tokens=self.max_subtask_tokens)

    def infer_batch(self, sessions, instructions, states, *, seed: int | None = None):
        """Return one (action array, subtask) per session; preserve input row order.

        seed=None retains native batch RNG semantics. An explicit seed isolates one request's
        noise using fork_rng. Different batch compositions need not draw identical action noise.
        """
        if not sessions or not (len(sessions) == len(instructions) == len(states)):
            raise ValueError("sessions, instructions and states must have matching nonempty lengths")
        if len({id(s) for s in sessions}) != len(sessions) or any(s.vla is not self.model for s in sessions):
            raise ValueError("each row needs a distinct session belonging to this runtime")
        if any(not isinstance(text, str) or not text.strip() for text in instructions):
            raise ValueError("nonempty task instructions required")
        arrays = [np.asarray(state, dtype=np.float32) for state in states]
        if any(a.shape != (self.config.state_dim,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("state shape/values do not match the checkpoint")
        torch = self.torch
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with _INFERENCE_LOCK, torch.random.fork_rng(devices=devices) if seed is not None else nullcontext():
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
            state_tensors = [self.normalize.normalize(torch.from_numpy(a)).unsqueeze(0).to(self.device)
                             if self.normalize is not None else None for a in arrays]
            inputs = [session._prepare_inputs(text) for session, text in zip(sessions, instructions)]
            outputs = self.batched.generate_batch(inputs, state_tensors)
        if len(outputs) != len(sessions) or any(
                a.shape != (self.config.action_horizon, self.config.action_dim) or not np.isfinite(a).all()
                for a, _ in outputs):
            raise RuntimeError("model returned invalid action chunks")
        return outputs
