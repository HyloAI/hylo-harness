import numpy as np

from worldcurrent.agents.steering import SteeredAgent, SteeringConfig, TriggerRule, select_chunk
from worldcurrent.core.agent import AgentSpec, EpisodeContext
from worldcurrent.core.task import TaskInstance, TaskSpec
from worldcurrent.core.types import (ActionSemantics, BoxSpace, EnvSpec, Observation, ObservationSpec, StepResult)

SPEC = EnvSpec(name="t", version="0", observation=ObservationSpec(state={"eef_pos": (3,), "gripper_qpos": (2,)}),
               action=BoxSpace(shape=(7,), low=(-1.0,) * 7, high=(1.0,) * 7), control_hz=20)


class FakeSampler:
    """Deterministic in the seed, like a flow policy given its noise."""

    def __init__(self):
        self.calls = []

    def spec(self):
        return AgentSpec(name="fake", version="0", action_dim=7,
                         action_semantics=ActionSemantics(control_mode="eef_delta_pose"), components={"policy": "fake"})

    def reset(self, ctx):
        pass

    def default_execution_steps(self):
        return 5

    def sample(self, obs, seeds, noise_scale):
        self.calls.append((list(seeds or []), noise_scale))
        return np.stack([np.random.default_rng(s).normal(size=(10, 7)) * noise_scale for s in (seeds or [0])])


def ctx(seed=3):
    events = []
    task = TaskInstance(spec=TaskSpec(id="t/a", version="1", instruction="x"), seed=seed)
    return EpisodeContext(episode_id="e", task=task, env_spec=SPEC, emit=lambda t, p: events.append((t, p)), seed=seed), events


def obs(z=0.0, width=0.08):
    return Observation(state={"eef_pos": np.array([0.0, 0.0, z]), "gripper_qpos": np.array([width / 2, -width / 2])})


def test_seeded_noise_reproduces_and_is_shared_across_agent_versions():
    a, b = SteeredAgent(FakeSampler(), {"n_samples": 4}), SteeredAgent(FakeSampler(), {"n_samples": 4, "selector": "gripper_vote_medoid"})
    for ag in (a, b):
        ag.reset(ctx()[0])
    ca, cb = a.act(obs()), b.act(obs())
    assert a.sampler.calls[0][0] == b.sampler.calls[0][0]        # common random numbers across versions
    a.reset(ctx()[0])
    np.testing.assert_array_equal(a.act(obs()).actions, ca.actions)  # replayable
    assert cb.provenance["n_samples"] == 4 and cb.provenance["disagreement"] > 0


def test_vote_excludes_the_gripper_minority():
    chunks = np.zeros((3, 10, 7))
    chunks[:2, :, -1], chunks[2, :, -1] = 1.0, -1.0
    chunks[2, :, :3] = 0.01                                        # the minority sample would be the medoid otherwise
    idx, _ = select_chunk(chunks, "gripper_vote_medoid", 10)
    assert idx in (0, 1)


def test_rules_fire_on_features_with_cooldown_and_cap():
    rule = TriggerRule(name="stall", feature="still_steps", value=3, action="backoff",
                       params={"steps": 4, "dz": 0.5}, cooldown=2, max_fires=2)
    ag = SteeredAgent(FakeSampler(), SteeringConfig(rules=[rule]))
    c, events = ctx()
    ag.reset(c)
    ag.act(obs())
    fired = []
    for _ in range(20):
        ag.on_step(StepResult(observation=obs(), reward=0, terminated=False, truncated=False))
        chunk = ag.act(obs())
        fired.append(chunk.provenance.get("model") == "recovery.backoff")
    assert sum(fired) == 2                                         # capped by max_fires
    back = next(e for e in events if e[0] == "agent.decision")[1]
    assert back["rule"] == "stall" and back["features"]["still_steps"] >= 3


def test_grasp_attempts_count_closing_events():
    ag = SteeredAgent(FakeSampler(), {})
    ag.reset(ctx()[0])
    for w in (0.08, 0.02, 0.08, 0.02, 0.02, 0.08, 0.01):
        ag.on_step(StepResult(observation=obs(width=w), reward=0, terminated=False, truncated=False))
    assert ag.features()["grasp_attempts"] == 3
