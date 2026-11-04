import numpy as np
import pytest

from hylo.memory.simplememvla import SimpleMemVLAHistory


def frames(value):
    return {k: np.full((2, 2, 3), value % 256, dtype=np.uint8) for k in ("front", "wrist")}


def test_upstream_fixed_window_padding_stride_and_rollover():
    memory = SimpleMemVLAHistory()
    memory.observe(frames(0))
    assert memory.frame_ids("front") == [0] * 120
    assert memory.patch_keys() == [("front", 0, 0)] * 60
    for i in range(1, 1301):
        memory.observe(frames(i))
    assert memory.frame_ids("front") == list(range(110, 1301, 10))
    assert memory.frame_ids("wrist") == [1300]
    assert len(memory.buffers["front"]) == 1191
    assert len(memory.buffers["wrist"]) == 1
    np.testing.assert_array_equal(memory.video("front")[:, 0, 0, 0], np.arange(110, 1301, 10) % 256)


def test_variable_window_even_padding_and_rounding():
    memory = SimpleMemVLAHistory(variable_history=True, history_video_sec=2.5, history_video_fps=2, native_fps=5)
    assert (memory.n_frames, memory.stride) == (6, 2)  # round(2.5) == 2, as upstream.
    memory.observe(frames(0))
    assert memory.frame_ids("front") == [0, 0]
    for i in range(1, 5):
        memory.observe(frames(i))
    assert memory.frame_ids("front") == [0, 0, 2, 4]
    for i in range(5, 15):
        memory.observe(frames(i))
    assert memory.frame_ids("front") == [4, 6, 8, 10, 12, 14]


def test_atomic_validation_copy_ownership_and_episode_reset():
    memory = SimpleMemVLAHistory()
    image = frames(7)
    memory.observe(image)
    image["front"][:] = 99
    assert memory.current("front").max() == 7
    memory.current("front")[:] = 99
    assert memory.current("front").max() == 7
    before = memory.snapshot()
    with pytest.raises(ValueError, match="missing camera"):
        memory.observe({"front": image["front"]})
    with pytest.raises(ValueError, match="changed shape"):
        memory.observe({**image, "wrist": np.zeros((3, 3, 3), dtype=np.uint8)})
    assert memory.snapshot() == before
    memory.feature_cache[("front", 0, 0)] = object()
    buffers, seen, features = memory.buffers, memory.frames_seen, memory.feature_cache
    memory.reset("new-episode")
    assert buffers is memory.buffers and seen is memory.frames_seen and features is memory.feature_cache
    assert not features and not any(seen.values())
    assert memory.snapshot()["episode_id"] == "new-episode"
    with pytest.raises(RuntimeError):
        memory.video("front")


@pytest.mark.parametrize("kwargs", [{"native_fps": 0}, {"history_video_sec": float("nan")},
                                    {"history_image_keys": ["missing"]}, {"image_keys": ["front", "front"]}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        SimpleMemVLAHistory(**kwargs)
