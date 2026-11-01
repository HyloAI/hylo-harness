import numpy as np
import pytest

from hylo.learn.critic import auc


def test_auc_ranks_positives_above_negatives():
    assert auc(np.array([0.9, 0.1]), np.array([True, False])) == 1.0
    assert auc(np.array([0.2, 0.8, 0.6]), np.array([True, False, True])) == 0.0


def test_trainer_refuses_gate_splits(monkeypatch):
    from hylo.learn import critic as critic_train

    monkeypatch.setattr("sys.argv", ["x", "--runs", "r", "--out", "o", "--splits", "search", "gate"])
    with pytest.raises(SystemExit):
        critic_train.main()
