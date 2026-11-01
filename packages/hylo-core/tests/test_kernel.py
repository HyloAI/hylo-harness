import numpy as np
import pytest

from hylo.agents.baseline import PControllerAgent, PlanarIKAgent
from hylo.eval.scoring import SuccessScorer
from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.errors import IncompatibleComponents
from hylo.core.kernel import EpisodeRunner
from hylo.core.trace import TraceStore
from hylo.core.types import ActionChunk, BoxSpace


def test_box_space_validates_lengths():
    with pytest.raises(ValueError):
        BoxSpace(shape=(2,), low=(0.0,), high=(1.0, 1.0))


def test_action_chunk_requires_time_axis():
    with pytest.raises(ValueError):
        ActionChunk(actions=np.zeros(2))


def test_incompatible_control_mode_is_an_error_not_a_warning(env):
    with pytest.raises(IncompatibleComponents, match="control_mode"):
        EpisodeRunner(env, PlanarIKAgent())


def test_episode_emits_lifecycle_and_is_deterministic(env, instance, tmp_path):
    actions = []
    for i in range(2):
        store = TraceStore(tmp_path / f"run{i}")
        rec = EpisodeRunner(env, PControllerAgent(kp=2.0), trace=store).run(instance, episode_id="ep")
        tr = store.read("ep")
        types = [e["type"] for e in tr.events]
        assert types[0] == "episode.start" and "episode.end" in types
        assert {"observation", "agent.chunk", "safety.review", "action", "step.result"} <= set(types)
        actions.append(np.stack([e["payload"]["data"] for e in tr.of_type("action")]))
        assert rec.outcome.status == "success"
    np.testing.assert_array_equal(actions[0], actions[1])


def test_scorer_reads_trace_only(env, instance, tmp_path):
    store = TraceStore(tmp_path)
    EpisodeRunner(env, PControllerAgent(kp=2.0), trace=store).run(instance, episode_id="ep")
    assert SuccessScorer().score(TraceStore(tmp_path).read("ep"), instance).value == 1.0


def test_chunk_execution_steps_control_replanning(env, instance):
    class Counting(PControllerAgent):
        calls = 0

        def act(self, obs):
            Counting.calls += 1
            return super().act(obs)

    rec = EpisodeRunner(env, Counting(kp=2.0, horizon=8, execution_steps=4)).run(instance)
    assert Counting.calls == -(-rec.outcome.steps // 4)


class Boom(AgentBase):
    def spec(self):
        return AgentSpec(name="boom", version="0")

    def act(self, obs):
        raise RuntimeError("model crashed")


def test_agent_failure_is_error_and_abstains(env, instance, tmp_path):
    store = TraceStore(tmp_path)
    rec = EpisodeRunner(env, Boom(), trace=store).run(instance, episode_id="ep")
    assert rec.outcome.status == "error" and rec.outcome.success is None
    assert SuccessScorer().score(store.read("ep"), instance).value is None


def test_env_failure_is_infra_invalid(env, instance):
    def broken_step(action):
        raise OSError("sim crashed")

    env.step = broken_step
    rec = EpisodeRunner(env, PControllerAgent()).run(instance)
    assert rec.outcome.status == "infra_invalid"


def test_privileged_state_goes_to_trace_not_agent(env, instance, tmp_path):
    seen = []

    class Spy(PControllerAgent):
        def act(self, obs):
            seen.append(set(obs.state))
            return super().act(obs)

    store = TraceStore(tmp_path)
    EpisodeRunner(env, Spy(kp=2.0), trace=store).run(instance, episode_id="ep")
    assert all("distance" not in keys for keys in seen)
    assert store.read("ep").first("privileged.state") is not None
