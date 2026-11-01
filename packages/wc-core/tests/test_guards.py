import numpy as np
import pytest

from worldcurrent.core.safety import KeepOutGuard, NormLimitGuard, SafetyContext, SafetyShield, build_guards
from worldcurrent.core.types import Action, BoxSpace, EnvSpec, ObservationSpec


def _ctx(pos=None):
    return SafetyContext(space=BoxSpace(shape=(2,), low=(-1, -1), high=(1, 1)), state={"pos": np.asarray(pos)} if pos is not None else None)


def test_norm_limit_scales_only_the_selected_dims():
    v = NormLimitGuard(max_norm=0.5, dims=[0, 1]).review(np.array([0.6, 0.8, 1.0]), _ctx())
    assert v.modified and np.allclose(v.data, [0.3, 0.4, 1.0])


def test_keep_out_removes_motion_into_the_box_and_allows_motion_away():
    g = KeepOutGuard("pos", low=[0.4, 0.4], high=[0.6, 0.6], action_dims=[0, 1])
    into = g.review(np.array([0.1, 0.0]), _ctx([0.35, 0.5]))
    assert into.modified and into.data[0] == 0.0
    away = g.review(np.array([-0.1, 0.0]), _ctx([0.35, 0.5]))
    assert not away.modified


def test_guard_plugins_come_from_the_safety_config_and_stay_inside_the_mandatory_checks():
    guards = build_guards({"max_delta": 0.5, "guards": [{"type": "norm_limit", "params": {"max_norm": 0.1}}]})
    assert [g.name for g in guards] == ["delta_limit", "norm_limit"]
    spec = EnvSpec(name="e", version="1", observation=ObservationSpec(state={"pos": (2,)}),
                   action=BoxSpace(shape=(2,), low=(-1, -1), high=(1, 1)), control_hz=10)
    shield = SafetyShield(spec, guards)
    assert shield.guard_names == ["finite", "delta_limit", "norm_limit", "bounds"]
    approved, review = shield.review(Action(data=np.array([0.3, 0.4])))
    assert review.modified and np.isclose(np.linalg.norm(approved.data), 0.1)
    with pytest.raises(KeyError, match="no guards plugin"):
        build_guards({"guards": [{"type": "nope"}]})
