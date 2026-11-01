import numpy as np

from hylo.agents.noise_steer import NoiseSteerConfig, NoiseSteeredAgent, noise_prior
from hylo.core.agent import AgentSpec, EpisodeContext
from hylo.core.experience import ActionSegment, EpisodeRef, PriorDecision
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Observation

REF = EpisodeRef(run_id="r", episode_id="e", split="search")


def noise_seg(v):
    return ActionSegment(key=np.ones(3), actions=np.full((4, 3), v, float) + np.arange(12).reshape(4, 3) * 0.01,
                         task_id="t/a", step=0, source=REF)


def test_noise_prior_is_a_unit_variance_weighted_mean_and_empty_below_the_floor():
    cfg = NoiseSteerConfig(weight=0.4, temperature=1.0)
    d = noise_prior([(noise_seg(1.0), 0.99), (noise_seg(-1.0), 0.98)], cfg)
    assert d.noise is not None and abs(float(np.asarray(d.noise).std()) - 1.0) < 1e-5 and d.noise_weight == 0.4
    assert d.prior is None and d.t0 == 1.0 and d.info["kept"] == 2
    assert noise_prior([], cfg).noise is None


class Mem:
    def __init__(self, sim):
        self.sim, self.queries = sim, []

    def query(self, key, *, task_id=None, k=10):
        self.queries.append(task_id)
        return [(noise_seg(0.5), self.sim)]

    def snapshot(self):
        return "cafe"

    def __len__(self):
        return 1


class Sampler:
    cfg = type("C", (), {"chunk_size": 4})()

    def __init__(self):
        self.decisions = []

    def spec(self):
        return AgentSpec(name="fake", version="0", action_dim=7, components={"policy": "fake@0"})

    def reset(self, ctx):
        pass

    def default_execution_steps(self):
        return 2

    def sample(self, observation, seeds, noise_scale=1.0, prior_fn=None):
        d = prior_fn(np.ones(3)) if prior_fn else PriorDecision(prior=None)
        self.decisions.append(d)
        return np.zeros((len(seeds or [0]), 4, 7))


def _agent(sim):
    s, events = Sampler(), []
    a = NoiseSteeredAgent(s, {"n_samples": 1, "seed_noise": True}, Mem(sim), {"weight": 0.5, "sim_floor": 0.9})
    a.reset(EpisodeContext(episode_id="e", task=TaskInstance(spec=TaskSpec(id="t/a", version="1", instruction="x"), seed=0),
                           env_spec=None, emit=lambda t, p: events.append((t, p)), seed=0))
    return a, s, events


def test_agent_hands_the_retrieved_noise_to_the_sampler_and_logs_it():
    a, s, events = _agent(0.97)
    chunk = a.act(Observation(state={}))
    assert s.decisions[0].noise is not None and chunk.provenance["noise_weight"] == 0.5
    ev = [p for t, p in events if t == "memory.retrieval"][0]
    assert ev["kind"] == "noise" and ev["used_prior"]
    assert "noise_bank(cafe" in a.spec().components["noise_prior"]
    far, s2, _ = _agent(0.5)                                    # below the floor: the unchanged policy
    assert far.act(Observation(state={})).provenance["noise_weight"] == 0.0 and s2.decisions[0].noise is None
