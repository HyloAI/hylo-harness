import numpy as np
import pytest

from hylo.memory.frames import FrameEvidence


def test_causal_visual_retrieval_and_copy_isolation():
    memory = FrameEvidence(demo_budget=3, rollout_budget=3)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(10):
        image[:] = i * 20
        memory.append(i, image, demonstration=i < 5)
    image[:] = 255
    ids, frames = memory.retrieve(before=9)
    assert ids == sorted(ids)
    assert 0 in ids and 4 in ids and 5 in ids and 8 in ids
    assert len(ids) == 6 and 9 not in ids
    assert frames[0].max() == 0
    frames[0][:] = 255
    assert memory.retrieve(before=9)[1][0].max() == 0
    with pytest.raises(ValueError):
        memory.append(9, image, demonstration=False)
    memory.reset()
    assert memory.retrieve(before=100) == ([], [])
