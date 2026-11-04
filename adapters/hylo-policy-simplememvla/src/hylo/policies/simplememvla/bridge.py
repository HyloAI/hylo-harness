"""Bind Hylo video memory to upstream inference without copying the model implementation."""
from functools import lru_cache

from hylo.memory.simplememvla import SimpleMemVLAHistory


def history_from_config(config):
    image_keys = tuple(config.image_keys)
    history_keys = getattr(config, "history_image_keys", None) or image_keys
    native_fps = getattr(config, "native_video_fps", None)
    if native_fps is None:
        native_fps = config.control_frequency_hz
    return SimpleMemVLAHistory(image_keys=image_keys, history_image_keys=history_keys,
                              history_video_sec=config.history_video_sec, history_video_fps=config.history_video_fps,
                              native_fps=native_fps, variable_history=bool(getattr(config, "variable_history", False)))


@lru_cache(maxsize=8)
def memory_policy_type(upstream_type):
    """The inherited tokenizer, preprocessing, feature caching and model remain upstream.

    The frame store, sampling indices, padding and patch identities come from Hylo.
    This also works with upstream BatchedEvalPolicy, so one model can serve many episodes.
    """
    class HyloMemoryPolicy(upstream_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = history_from_config(self.vla.config)
            if len({key.split(".")[-1] for key in self.image_keys}) != len(self.image_keys):
                raise ValueError("SimpleMemVLA camera tags must be unique")
            self._buffers = self.memory.buffers
            self._frames_seen = self.memory.frames_seen
            self._patch_feat_cache = self.memory.feature_cache

        def reset(self):
            super().reset()
            self.memory.reset()
            # Upstream reset replaces its counter dict; reconnect to Hylo's containers.
            self._buffers = self.memory.buffers
            self._frames_seen = self.memory.frames_seen
            self._patch_feat_cache = self.memory.feature_cache

        def observe(self, frames):
            self.memory.observe(frames)

        def _history_positions(self, key):
            return self.memory.positions(key)

        def _history_global_indices(self, key):
            return self.memory.frame_ids(key)

        def _history_video(self, key):
            return self.memory.video(key)

        def _video_patch_keys(self):
            return [(key.split(".")[-1], a, b) for key, a, b in self.memory.patch_keys()]

    return HyloMemoryPolicy
