from hylo.agents.shadow import ShadowAgent, divergence, shadow_report
from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
import numpy as np
import pytest


def test_divergence_compares_only_the_executed_prefix():
    a = np.zeros((10, 7)); a[:, -1] = 1
    b = np.zeros((10, 7)); b[5:, :3] = 1.0; b[:, -1] = 1
    assert divergence(a, b, 5) == {"steps": 5, "mean_abs": 0.0, "max_abs": 0.0, "gripper_agree": 1.0}
    assert divergence(a, b, 10)["max_abs"] == 1.0


@pytest.mark.parametrize("background", [False, True])
def test_the_shadow_never_acts_but_every_decision_is_compared(tmp_path, background):
    cfg = ExperimentConfig(name="sh", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                           env={"type": "testing.point_reach"},
                           agent={"type": "shadow", "params": {"background": background, "gripper_dim": None,
                                                               "active": {"type": "testing.p_controller", "params": {"kp": 2.0}},
                                                               "shadow": {"type": "testing.p_controller", "params": {"kp": 0.2}}}})
    plain = run_experiment(cfg.with_overrides({"agent": {"type": "testing.p_controller", "params": {"kp": 2.0}}}),
                           run_root=tmp_path, label="plain")
    shadow = run_experiment(cfg, run_root=tmp_path, label="shadow")
    assert shadow.per_key() == plain.per_key()                   # outcomes are the active agent's alone
    rep = shadow_report(shadow.run_dir)
    steps = sum(s["steps"] for s in shadow.log["samples"])
    assert rep["decisions"] == steps and rep["errors"] == 0      # one comparison per policy call (1-step chunks)
    assert rep["mean_abs"] > 0                                    # a weaker gain would have acted differently
