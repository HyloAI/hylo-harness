import numpy as np
import pytest

from hylo.agents.critic import CriticSteeredAgent
from hylo.instruments.critic import ChunkCritic, FeatureConfig, chunk_features, state_features
from hylo.core.agent import AgentSpec, EpisodeContext
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Observation

CFG = FeatureConfig(tasks=("t/a", "t/b"))
STATE = {"eef_pos": np.zeros(3), "eef_axis_angle": np.zeros(3), "gripper_qpos": np.array([0.04, -0.04])}


def _chunk(dz: float, grip: float = -1.0, T: int = 20) -> np.ndarray:
    c = np.zeros((T, 7))
    c[:, 2], c[:, -1] = dz, grip
    return c


def _dataset(n: int, rng):
    """Success iff the chunk moves down (dz < 0): only the action carries the signal."""
    xs, ys = [], []
    for _ in range(n):
        dz = rng.uniform(-1, 1)
        s = state_features(STATE, int(rng.integers(0, 300)), "t/a", None, CFG)
        xs.append(np.concatenate([s, chunk_features(_chunk(dz), 10, CFG.prefix_steps)]))
        ys.append(0.8 if dz < 0 else 0.0)
    return np.stack(xs), np.array(ys)


def test_critic_learns_action_signal_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    x, y = _dataset(800, rng)
    critic = ChunkCritic.fit(x, y, CFG, members=3, epochs=40)
    xv, yv = _dataset(300, rng)
    p, pos = critic.predict(xv)[0], yv > 0
    assert (p[pos][:, None] > p[~pos][None, :]).mean() > 0.95      # AUC
    critic.save(tmp_path / "c")
    again = ChunkCritic.load(tmp_path / "c")
    np.testing.assert_allclose(again.predict(xv)[0], critic.predict(xv)[0])
    assert len(again.meta["weights_sha256_16"]) == 16


class FakeSampler:
    def __init__(self, chunks):
        self.chunks = chunks

    def spec(self):
        return AgentSpec(name="fake", version="0", action_dim=7, components={"policy": "fake@0"})

    def reset(self, ctx):
        pass

    def default_execution_steps(self):
        return 10

    def sample(self, observation, seeds, noise_scale):
        return np.stack(self.chunks)


def test_agent_executes_the_highest_valued_chunk_within_the_gripper_vote(tmp_path):
    rng = np.random.default_rng(1)
    x, y = _dataset(800, rng)
    ChunkCritic.fit(x, y, CFG, members=3, epochs=40).save(tmp_path / "c")
    # three open-gripper chunks (majority) and one closing chunk that moves down the most
    chunks = [_chunk(0.6), _chunk(-0.5), _chunk(0.3), _chunk(-0.9, grip=1.0)]
    agent = CriticSteeredAgent(FakeSampler(chunks), {"n_samples": 4, "selector": "gripper_vote_medoid"},
                               {"path": str(tmp_path / "c"), "beta": 0.0})
    task = TaskInstance(spec=TaskSpec(id="t/a", version="1", instruction="x"), seed=0)
    events = []
    agent.reset(EpisodeContext(episode_id="e", task=task, env_spec=None, emit=lambda t, p: events.append(p), seed=0))
    out = agent.act(Observation(state=STATE))
    assert out.provenance["selected"] == 1 and out.provenance["candidates"] == [0, 1, 2]
    assert "critic(" in agent.spec().components["critic"]
    agent.critic_cfg = agent.critic_cfg.model_copy(update={"within_vote": False})
    prov = agent.act(Observation(state=STATE)).provenance
    assert prov["candidates"] == [0, 1, 2, 3] and prov["selected"] == int(np.argmax(prov["critic_mean"]))
