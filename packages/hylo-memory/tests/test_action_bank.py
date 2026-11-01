import numpy as np
import pytest

from hylo.core.experience import ActionMemory, ActionSegment, EpisodeRef
from hylo.memory.action_bank import ActionBank, FrozenMemoryError


def seg(key, task="t/a", step=0, ep="e0", val=0.0):
    return ActionSegment(key=np.asarray(key, float), actions=np.full((5, 7), val), task_id=task, step=step,
                         source=EpisodeRef(run_id="r", episode_id=ep, split="search"))


def test_cosine_query_task_filter_and_fifo():
    bank = ActionBank(capacity=3)
    assert isinstance(bank, ActionMemory)
    bank.add([seg([1, 0, 0], ep="a"), seg([0, 1, 0], ep="b"), seg([1, 1, 0], task="t/b", ep="c")])
    hits = bank.query(np.array([2.0, 0.1, 0]), k=2)
    assert [h[0].source.episode_id for h in hits] == ["a", "c"] and hits[0][1] > 0.99
    assert [h[0].source.episode_id for h in bank.query(np.array([1.0, 1, 0]), task_id="t/a", k=5)] in (["a", "b"], ["b", "a"])
    bank.add([seg([0, 0, 1], ep="d")])                      # evicts the oldest ("a")
    assert len(bank) == 3 and all(h[0].source.episode_id != "a" for h in bank.query(np.array([1.0, 0, 0]), k=3))


def test_save_load_is_frozen_and_digest_checked(tmp_path):
    bank = ActionBank()
    bank.add([seg([1, 0, 0], val=0.5), seg([0, 1, 0], step=3)])
    digest = bank.save(tmp_path / "b")
    again = ActionBank(str(tmp_path / "b"))
    assert again.snapshot() == digest and len(again) == 2 and again.frozen
    np.testing.assert_allclose(again.query(np.array([1.0, 0, 0]), k=1)[0][0].actions, 0.5)
    with pytest.raises(FrozenMemoryError):
        again.add([seg([1, 0, 0])])
    data = (tmp_path / "b" / "bank.json").read_text().replace('"step": 3', '"step": 4')
    (tmp_path / "b" / "bank.json").write_text(data)
    with pytest.raises(ValueError, match="digest"):
        ActionBank(str(tmp_path / "b"))
