"""SimpleMemVLA's frame-indexed video memory, independent of other Hylo memories.

Sampling and padding reproduce OpenBMB/SimpleMemVLA commit 453d6ec293e63318bbc2bce545d0b9256ab1bc51.
See THIRD_PARTY_NOTICES.md in hylo-memory for the upstream MIT attribution.
This is a streaming visual memory, not a semantic MemoryStore implementation.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Mapping

import numpy as np


class SimpleMemVLAHistory:
    """Bounded raw frames, upstream-exact sampling and episode-local feature cache.

    Each observe call is one native camera tick, including demonstration ticks. Sampling
    uses frame indices, as upstream does, rather than irregular wall-clock timestamps.
    """

    def __init__(self, *, image_keys=("front", "wrist"), history_image_keys=("front",),
                 history_video_sec: float = 60.0, history_video_fps: float = 2.0,
                 native_fps: float = 20.0, variable_history: bool = False):
        self.image_keys = tuple(image_keys)
        requested = set(history_image_keys)
        if (not self.image_keys or len(set(self.image_keys)) != len(self.image_keys)
                or not all(isinstance(k, str) and k for k in self.image_keys)):
            raise ValueError("image keys must be nonempty unique camera names")
        if not requested or not requested.issubset(self.image_keys):
            raise ValueError("history cameras must be a nonempty subset of image keys")
        for value in (history_video_sec, history_video_fps, native_fps):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("history duration and frame rates must be finite and positive")
        # Python round, even rounding-up and index clamping intentionally match upstream.
        self.n_frames = max(2, round(history_video_sec * history_video_fps))
        self.n_frames += self.n_frames % 2
        self.stride = max(1, round(native_fps / history_video_fps))
        self.native_fps = float(native_fps)
        self.variable_history = bool(variable_history)
        self.history_image_keys = tuple(k for k in self.image_keys if k in requested)
        self.capacities = {k: (self.n_frames - 1) * self.stride + 1 if k in requested else 1
                           for k in self.image_keys}
        self.buffers = {k: deque(maxlen=capacity) for k, capacity in self.capacities.items()}
        self.frames_seen = dict.fromkeys(self.image_keys, 0)
        self.feature_cache: dict[Any, Any] = {}
        self.episode_id: str | None = None
        self._shapes: dict[str, tuple[int, ...]] = {}

    def reset(self, episode_id: str | None = None) -> None:
        # Keep container identities: inference adapters can hold references to them.
        for key, buffer in self.buffers.items():
            buffer.clear()
            self.frames_seen[key] = 0
        self.feature_cache.clear()
        self._shapes.clear()
        self.episode_id = episode_id

    def observe(self, frames: Mapping[str, np.ndarray]) -> None:
        """Validate the entire tick before appending; own immutable copies of the RGB data."""
        images = {}
        for key in self.image_keys:
            if key not in frames:
                raise ValueError(f"missing camera {key!r}")
            image = np.asarray(frames[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"camera {key!r} must contain uint8 HWC RGB")
            if key in self._shapes and image.shape != self._shapes[key]:
                raise ValueError(f"camera {key!r} changed shape within an episode")
            images[key] = image.copy()
            images[key].flags.writeable = False
        for key, image in images.items():
            self.buffers[key].append(image)
            self.frames_seen[key] += 1
            self._shapes[key] = image.shape

    def positions(self, key: str) -> list[int]:
        """Positions within the retained camera buffer, including upstream padding."""
        current = len(self.buffers[key]) - 1
        if current < 0:
            raise RuntimeError("observe frames before reading memory")
        if key not in self.history_image_keys:
            return [current]
        count = self.n_frames
        if self.variable_history:
            available = current // self.stride + 1
            count = max(2, available + available % 2)
        return [max(0, current - (count - 1 - i) * self.stride) for i in range(count)]

    def frame_ids(self, key: str) -> list[int]:
        start = self.frames_seen[key] - len(self.buffers[key])
        return [start + pos for pos in self.positions(key)]

    def video(self, key: str) -> np.ndarray:
        return np.stack([self.buffers[key][pos] for pos in self.positions(key)])

    def current(self, key: str) -> np.ndarray:
        if not self.buffers[key]:
            raise RuntimeError("observe frames before reading memory")
        return self.buffers[key][-1].copy()

    def patch_keys(self) -> list[tuple[str, int, int]]:
        keys = []
        for camera in self.history_image_keys:
            ids = self.frame_ids(camera)
            keys.extend((camera, ids[i], ids[i + 1]) for i in range(0, len(ids), 2))
        return keys

    def snapshot(self) -> dict:
        return dict(episode_id=self.episode_id, frames_seen=dict(self.frames_seen),
                    retained_frames={k: len(v) for k, v in self.buffers.items()},
                    sampled_frame_ids={k: self.frame_ids(k) if self.buffers[k] else [] for k in self.image_keys},
                    sampled_fps=self.native_fps / self.stride, variable_history=self.variable_history)
