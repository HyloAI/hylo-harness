"""Bounded visual retrieval over an episode's observed RGB frames.

Keep demonstration coverage and diverse rollout evidence separately. This is an image-only
heuristic, not a detector of task-relevant events; the lossless evaluation trace remains the
source of truth when a retrieval budget omits a cue.
"""
from __future__ import annotations

import numpy as np


class FrameEvidence:
    def __init__(self, *, demo_budget: int = 32, rollout_budget: int = 16, max_frames: int = 10000):
        if min(demo_budget, rollout_budget, max_frames) < 1:
            raise ValueError("frame budgets must be positive")
        self.demo_budget, self.rollout_budget, self.max_frames = demo_budget, rollout_budget, max_frames
        self.reset()

    def reset(self):
        self._frames = {}
        self._features = {}
        self._demo = []
        self._rollout = []

    def append(self, frame_id: int, rgb: np.ndarray, *, demonstration: bool):
        if frame_id < 0 or (self._frames and frame_id <= max(self._frames)):
            raise ValueError("frame IDs must increase")
        if len(self._frames) >= self.max_frames:
            raise OverflowError("frame evidence capacity exhausted")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("expected uint8 RGB image")
        self._frames[frame_id] = image.copy()
        ys = np.linspace(0, image.shape[0] - 1, 24).astype(int)
        xs = np.linspace(0, image.shape[1] - 1, 24).astype(int)
        self._features[frame_id] = image[ys[:, None], xs].astype(np.float32).ravel() / 255
        (self._demo if demonstration else self._rollout).append(frame_id)

    def retrieve(self, *, before: int):
        """Only past frames; chronological output with bounded demo and rollout allocations."""
        demo = [i for i in self._demo if i < before]
        if len(demo) > self.demo_budget:
            demo = [demo[i] for i in np.linspace(0, len(demo) - 1, self.demo_budget).astype(int)]
        candidates = [i for i in self._rollout if i < before]
        if len(candidates) <= self.rollout_budget:
            chosen = candidates
        else:
            features = np.stack([self._features[i] for i in candidates])
            selected = [0]
            if self.rollout_budget > 1:
                selected.append(len(candidates) - 1)
            distance = np.full(len(candidates), np.inf)
            for index in selected:
                distance = np.minimum(distance, np.mean((features - features[index]) ** 2, axis=1))
            while len(selected) < self.rollout_budget:
                distance[selected] = -1
                index = int(np.argmax(distance))
                selected.append(index)
                distance = np.minimum(distance, np.mean((features - features[index]) ** 2, axis=1))
            chosen = [candidates[i] for i in selected]
        ids = sorted(demo + chosen)
        return ids, [self._frames[i].copy() for i in ids]
