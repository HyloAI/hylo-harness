from types import SimpleNamespace

import numpy as np
import pytest

from hylo.agents.event_memory import EventMemoryAgent
from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.episode_memory import RecallQuery
from hylo.core.types import ActionChunk, Observation, StepResult
from hylo.memory.episode import EventMemory
from hylo.memory.signals import Signal, SignalExtractor


class RecordingAgent(AgentBase):
    def spec(self):
        return AgentSpec(name="unchanged-policy", version="1", required_state=())

    def act(self, observation):
        self.observation = observation
        return ActionChunk(actions=np.zeros((8, 2)), execution_steps=8)


def context(episode="one"):
    events = []
    return SimpleNamespace(episode_id=episode, task=SimpleNamespace(spec=SimpleNamespace(id="test")),
                           emit=lambda name, value: events.append((name, value))), events


def obs(step, pulse, cue=3, visible=True):
    return Observation(state={"pulse": np.array([pulse]), "cue": np.array([cue]),
                              "visible": np.array([visible])}, time=step, instruction="Follow the cue")


def result(observation):
    return StepResult(observation=observation, reward=999, success=True, terminated=False, truncated=False,
                      info={"oracle_target": "never passed to the extractor"})


def make_agent(**kw):
    memory = EventMemory()
    extractor = SignalExtractor([Signal(kind="press", subject="button", value_key="pulse", mode="rising"),
                                 Signal(kind="target", subject="object", value_key="cue", visible_key="visible")])
    queries = [RecallQuery(name="presses", op="count", kind="press", complete=True),
               RecallQuery(name="target", op="latest", kind="target")]
    return EventMemoryAgent(RecordingAgent(), memory, extractor, queries, **kw)


def test_transient_inside_chunk_is_observed_once_and_context_is_external():
    agent = make_agent(max_execution_steps=4)
    ctx, trace = context()
    agent.reset(ctx)
    original = obs(0, 0)
    chunk = agent.act(original)
    assert chunk.execution_steps == 4
    for i, pulse in enumerate([1, 1, 0, 1], start=1):
        agent.on_step(result(obs(i, pulse, cue=999, visible=False)))
    agent.act(obs(4, 1, cue=999, visible=False))
    counts = agent.memory.recall(agent.queries[0])
    assert counts.status == "known" and counts.value == 2
    assert agent.memory.recall(agent.queries[1]).value == 3
    assert original.instruction == "Follow the cue"
    assert '"presses"' in agent.inner.observation.instruction
    assert "oracle_target" not in agent.inner.observation.instruction
    assert agent.spec().required_state == ("pulse", "cue", "visible")
    assert any(name == "memory.events.retrieval" for name, _ in trace)


def test_reset_clears_extractor_edges_and_episode_recall():
    agent = make_agent()
    ctx, _ = context()
    agent.reset(ctx)
    agent.act(obs(0, 0))
    agent.on_step(result(obs(1, 1)))
    ctx2, _ = context("two")
    agent.reset(ctx2)
    agent.act(obs(0, 0, cue=8))
    assert agent.memory.recall(agent.queries[0]).value == 0
    assert agent.memory.recall(agent.queries[1]).value == 8


def test_context_budget_does_not_truncate_json_or_mutate_task():
    agent = make_agent(max_context_chars=2)
    ctx, trace = context()
    agent.reset(ctx)
    agent.act(obs(0, 0))
    assert agent.inner.observation.instruction == "Follow the cue"
    _, event = trace[-1]
    assert event["results"] == [] and event["omitted"] == ["presses", "target"]


def test_hidden_edge_stream_is_unknown_not_a_fabricated_count():
    extractor = SignalExtractor([Signal(kind="press", subject="b", value_key="pulse", visible_key="visible", mode="rising")])
    memory = EventMemory()
    from hylo.core.episode_memory import EpisodeScope
    scope = EpisodeScope(task_id="test", episode_id="e")
    extractor.reset(scope)
    memory.reset(scope)
    for step, observation in enumerate([obs(0, 0), obs(1, 1, visible=False), obs(2, 0)]):
        memory.append(extractor.observe(observation, step))
    assert memory.recall(RecallQuery(name="count", op="count", kind="press", complete=True)).status == "unknown"


def test_bad_budgets_and_duplicate_query_names_rejected():
    with pytest.raises(ValueError):
        make_agent(max_execution_steps=0)
