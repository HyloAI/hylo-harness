from types import SimpleNamespace

import numpy as np

from hylo.policies.simplememvla.bridge import memory_policy_type


class UpstreamStub:
    def __init__(self):
        self.vla = SimpleNamespace(config=SimpleNamespace(image_keys=["images.front", "images.wrist"],
            history_image_keys=["images.front"], history_video_sec=2, history_video_fps=2,
            native_video_fps=4, variable_history=False))
        self.image_keys = self.vla.config.image_keys
    def reset(self):
        for buf in self._buffers.values():
            buf.clear()
        self._frames_seen = {k: 0 for k in self.image_keys}
        self._patch_feat_cache.clear()
        self._prompt_cache = None


def test_upstream_private_views_and_reset_share_hylo_ownership():
    policy = memory_policy_type(UpstreamStub)()
    for i in range(9):
        policy.observe({k: np.full((2, 2, 3), i, dtype=np.uint8) for k in policy.image_keys})
    assert policy._history_global_indices("images.front") == [2, 4, 6, 8]
    assert policy._video_patch_keys() == [("front", 2, 4), ("front", 6, 8)]
    np.testing.assert_array_equal(policy._history_video("images.front")[:, 0, 0, 0], [2, 4, 6, 8])
    policy._patch_feat_cache[("front", 2, 4)] = "embedding"
    assert policy.memory.feature_cache[("front", 2, 4)] == "embedding"
    policy.reset()
    assert policy._frames_seen is policy.memory.frames_seen
    assert policy._buffers is policy.memory.buffers
    assert policy._patch_feat_cache is policy.memory.feature_cache
    assert not policy._patch_feat_cache and policy._prompt_cache is None
