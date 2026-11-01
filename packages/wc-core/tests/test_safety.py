import numpy as np
import pytest

from worldcurrent.core.agent import AgentBase, AgentSpec
from worldcurrent.core.errors import SafetyAbort
from worldcurrent.core.kernel import EpisodeRunner
from worldcurrent.core.safety import DeltaLimitGuard, GuardVerdict, SafetyShield
from worldcurrent.core.types import Action, ActionChunk, EnvCapabilities


class Const(AgentBase):
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def spec(self):
        return AgentSpec(name="const", version="0", action_dim=2)

    def act(self, obs):
        return ActionChunk.single(self.value)


def test_non_finite_action_aborts_episode(env, instance):
    rec = EpisodeRunner(env, Const([np.nan, 0.0])).run(instance)
    assert rec.outcome.status == "safety_abort"


def test_out_of_bounds_is_clamped_and_counted(env, instance):
    runner = EpisodeRunner(env, Const([5.0, -5.0]))
    rec = runner.run(instance)
    assert rec.safety_interventions > 0


def test_extra_guard_cannot_loosen_mandatory_bounds(env):
    class Loosen:
        name = "evil"

        def review(self, data, ctx):
            return GuardVerdict(data * 100, True, "scale up")

    shield = SafetyShield(env.spec(), [Loosen()])
    approved, review = shield.review(Action(data=np.array([0.5, 0.5])))
    assert np.all(np.abs(approved.data) <= 1.0) and review.modified


def test_delta_limit_rate_limits(env):
    shield = SafetyShield(env.spec(), [DeltaLimitGuard(0.1)])
    shield.review(Action(data=np.zeros(2)))
    approved, _ = shield.review(Action(data=np.ones(2)))
    np.testing.assert_allclose(approved.data, [0.1, 0.1])


def test_real_robot_aborts_instead_of_silently_modifying(env):
    spec = env.spec()
    spec = spec.model_copy(update={"capabilities": EnvCapabilities(real_robot=True)})
    shield = SafetyShield(spec)
    with pytest.raises(SafetyAbort):
        shield.review(Action(data=np.array([3.0, 0.0])))


def test_small_bound_overshoot_is_a_clip_not_an_intervention(env):
    shield = SafetyShield(env.spec(), clip_tolerance=0.05)
    approved, review = shield.review(Action(data=np.array([1.03, 0.0])))
    assert approved.data[0] == 1.0 and review.clipped and not review.modified and shield.interventions == 0
    _, review = shield.review(Action(data=np.array([1.5, 0.0])))
    assert review.modified and shield.interventions == 1


def test_real_robot_has_no_clip_tolerance(env):
    spec = env.spec().model_copy(update={"capabilities": EnvCapabilities(real_robot=True)})
    with pytest.raises(SafetyAbort):
        SafetyShield(spec, clip_tolerance=0.05).review(Action(data=np.array([1.01, 0.0])))


def test_workspace_guard_blocks_moves_further_outside(env):
    from worldcurrent.core.types import WorkspaceLimit

    spec = env.spec().model_copy(update={"workspace": WorkspaceLimit(state_key="pos", low=(-0.5, -0.5), high=(0.5, 0.5),
                                                                       action_dims=(0, 1))})
    shield = SafetyShield(spec)
    shield.observe({"pos": np.array([0.6, 0.0])})                 # outside on +x
    approved, review = shield.review(Action(data=np.array([0.5, 0.3])))
    assert approved.data[0] == 0.0 and approved.data[1] == 0.3 and review.modified
    approved, review = shield.review(Action(data=np.array([-0.5, 0.3])))    # moving back inside is allowed
    assert approved.data[0] == -0.5 and not review.modified
    assert "workspace" in shield.guard_names
