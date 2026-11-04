from types import SimpleNamespace

import numpy as np
import pytest

from hylo.core.types import Observation, StepResult
from hylo.memory.simplememvla import SimpleMemVLAHistory
from hylo.policies.simplememvla.agent import SimpleMemVLAAgent


class Runtime:
    config = SimpleNamespace(image_keys=["observation.images.front", "observation.images.wrist"],
                             state_dim=8, action_dim=8, action_horizon=30, control_frequency_hz=20)
    def new_session(self):
        memory = SimpleMemVLAHistory(image_keys=self.config.image_keys, history_image_keys=self.config.image_keys[:1])
        return SimpleNamespace(memory=memory, reset=memory.reset, observe=memory.observe)
    def infer_batch(self, sessions, instructions, states, *, seed):
        self.last = (sessions, instructions, states, seed)
        return [(np.zeros((30, 8)), "move the gripper")]


def context(episode="e"):
    return SimpleNamespace(episode_id=episode, seed=7, task=object(),
                           env_spec=SimpleNamespace(control_hz=20, action=SimpleNamespace(shape=(8,),
                               semantics=SimpleNamespace(control_mode="joint_pos"))))


def observation(i=0):
    return Observation(images={k: np.full((2, 2, 3), i, dtype=np.uint8) for k in ("front", "wrist")},
                       state={"joint_pos": np.arange(7), "gripper_pos": np.array([.01, .02]),
                              "oracle": np.array([99])}, instruction="pick the cube", extra={"success": True})


def test_streaming_steps_no_duplicate_act_frame_or_privileged_inputs():
    runtime = Runtime()
    agent = SimpleMemVLAAgent(runtime=runtime)
    agent.reset(context())
    agent.observe_history([observation(1), observation(2)])
    assert agent.act(observation(3)).execution_steps == 16
    agent.act(observation(3))
    assert agent.session.memory.frames_seen["observation.images.front"] == 3
    agent.on_step(StepResult(observation=observation(4), reward=99, terminated=False, truncated=False,
                            success=True, info={"hidden_target": 3}))
    agent.act(observation(4))
    assert agent.session.memory.frames_seen["observation.images.front"] == 4
    assert runtime.last[1] == ["pick the cube"]
    np.testing.assert_allclose(runtime.last[2][0], [0, 1, 2, 3, 4, 5, 6, .01])
    assert agent.session.memory.snapshot()["episode_id"] == "e"
    with pytest.raises(RuntimeError):
        agent.observe_history([observation()])
    old = agent.session
    agent.on_episode_end(SimpleNamespace(success=True))
    assert not old.memory.feature_cache and not any(old.memory.frames_seen.values())
    with pytest.raises(RuntimeError):
        agent.act(observation())


def test_episode_reset_seed_reproducibility_and_interface_check():
    runtime = Runtime()
    agent = SimpleMemVLAAgent(runtime=runtime)
    agent.reset(context())
    first = agent.act(observation()).provenance["seed"]
    agent.reset(context("next"))
    assert agent.act(observation()).provenance["seed"] == first
    assert agent.session.memory.frames_seen["observation.images.front"] == 1
    bad = context()
    bad.env_spec.control_hz = 10
    with pytest.raises(ValueError, match="control rate"):
        agent.reset(bad)
