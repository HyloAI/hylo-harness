import numpy as np

from hylo.agents.memory_steer import (MemorySteerConfig, MemorySteeredAgent, elite_prior,
                                              verified_decision_points)
from hylo.core.agent import AgentSpec, EpisodeContext
from hylo.core.experience import ActionSegment, EpisodeRef, PriorDecision
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Observation

REF = EpisodeRef(run_id="r", episode_id="e", split="search")


def seg(dx, grip=-1.0, T=20):
    a = np.zeros((T, 7))
    a[:, 0], a[:, -1] = dx, grip
    return ActionSegment(key=np.ones(3), actions=a, task_id="t/a", step=0, source=REF)


def test_elite_prior_filters_outliers_votes_gripper_and_scores_confidence():
    cfg = MemorySteerConfig(sim_floor=0.9, temperature=0.05)
    cands = [(seg(0.2), 0.99), (seg(0.21), 0.985), (seg(0.19), 0.98), (seg(-0.9), 0.97)]
    d = elite_prior(cands, cfg, horizon=20, compare=10)
    assert d.info["kept"] == 3                                   # the -0.9 outlier is dropped by DTW
    assert abs(d.prior[:, 0].mean() - 0.2) < 0.02
    # (a gripper flip counts as inconsistency too: DTW runs over all action dims, gripper included)
    voted = elite_prior([(seg(0.2), 0.99), (seg(0.2, grip=1.0), 0.95)], cfg, 20, 10)
    assert np.all(voted.prior[:, -1] == -1.0)                    # the more similar candidate's gripper wins
    tight = elite_prior(cands[:2], cfg, 20, 10)
    loose = elite_prior([(seg(0.2), 0.92), (seg(0.6), 0.91)], cfg, 20, 10)
    assert tight.t0 < loose.t0 <= cfg.t0_max                     # confident retrieval -> stronger guidance
    assert elite_prior([], cfg, 20, 10).prior is None


class FakeMemory:
    def __init__(self, segs):
        self.segs, self.queries = segs, []

    def query(self, key, *, task_id=None, k=10):
        self.queries.append(task_id)
        return [(s, 0.99) for s in self.segs][:k]

    def snapshot(self):
        return "deadbeef"

    def __len__(self):
        return len(self.segs)


class FakePriorSampler:
    cfg = type("C", (), {"chunk_size": 20})()

    def __init__(self):
        self.decisions = []

    def spec(self):
        return AgentSpec(name="fake", version="0", action_dim=7, components={"policy": "fake@0"})

    def reset(self, ctx):
        pass

    def default_execution_steps(self):
        return 10

    def sample(self, observation, seeds, noise_scale=1.0, prior_fn=None):
        d = prior_fn(np.ones(3)) if prior_fn else PriorDecision(prior=None)
        self.decisions.append(d)
        base = np.zeros((len(seeds or [0]), 20, 7))
        if d.prior is not None:                                  # stand-in for flow sampling from x_t0
            base += (1 - d.t0) * d.prior[None]
        return base


def test_agent_queries_same_task_memory_and_records_the_decision():
    mem, sampler, events = FakeMemory([seg(0.3), seg(0.31), seg(0.29)]), FakePriorSampler(), []
    agent = MemorySteeredAgent(sampler, {"n_samples": 1}, mem)
    task = TaskInstance(spec=TaskSpec(id="t/a", version="1", instruction="x"), seed=0)
    agent.reset(EpisodeContext(episode_id="e", task=task, env_spec=None, emit=lambda t, p: events.append((t, p)), seed=0))
    chunk = agent.act(Observation(state={}))
    assert mem.queries == ["t/a"] and sampler.decisions[0].prior is not None
    assert chunk.actions[:, 0].mean() > 0 and 0 < chunk.provenance["t0"] < 1
    ev = [p for t, p in events if t == "memory.retrieval"][0]
    assert ev["used_prior"] and ev["kept"] == 3
    assert "bank(deadbeef,n=3" in agent.spec().components["action_memory"]


class FakeTrace:
    episode_id = "ep1"

    def __init__(self, status, split="search"):
        img = {"agentview": np.zeros((4, 4, 3))}
        self.events = [{"type": "episode.start", "payload": {"split": split, "task_id": "t/a"}},
                       {"type": "observation", "payload": {"state": {}, "images": img}},
                       {"type": "agent.chunk", "payload": {"actions": np.zeros((5, 7)), "provenance": {"model": "p"}}},
                       {"type": "step.result", "payload": {}}, {"type": "step.result", "payload": {}},
                       {"type": "observation", "payload": {"state": {}, "images": {}}},
                       {"type": "agent.chunk", "payload": {"actions": np.ones((5, 7)), "provenance": {"model": "p"}}},
                       {"type": "observation", "payload": {"state": {}, "images": img}},
                       {"type": "agent.chunk", "payload": {"actions": np.ones((5, 7)), "provenance": {"model": "recovery.backoff"}}},
                       {"type": "episode.end", "payload": {"status": status, "steps": 4}}]

    def first(self, t):
        return next(e for e in self.events if e["type"] == t)

    def last(self, t):
        return [e for e in self.events if e["type"] == t][-1]


def test_only_imaged_policy_chunks_of_verified_dev_successes():
    pts = list(verified_decision_points(FakeTrace("success")))
    assert len(pts) == 1 and pts[0]["step"] == 0 and pts[0]["episode_id"] == "ep1"
    assert list(verified_decision_points(FakeTrace("failure"))) == []
    assert list(verified_decision_points(FakeTrace("success", split="gate"))) == []
