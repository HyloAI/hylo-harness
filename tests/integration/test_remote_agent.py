"""AgentServer / RemoteAgent: an agent in another process behaves like a local one, and its trace
events (decisions, memory access) still reach the parent's trace."""

from worldcurrent.agents.baseline import PControllerAgent
from worldcurrent.core.kernel import EpisodeRunner
from worldcurrent.core.remote.agent_client import RemoteAgent
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.trace import TraceStore


def test_remote_agent_matches_local_agent(env, tasks):
    inst = TaskInstance(spec=tasks["reach_push"], seed=4)
    local = EpisodeRunner(env, PControllerAgent(kp=2.0)).run(inst).outcome
    remote = RemoteAgent("worldcurrent.agents.baseline:PControllerAgent", {"kp": 2.0})
    try:
        out = EpisodeRunner(env, remote).run(inst).outcome
    finally:
        remote.close()
    assert (out.status, out.steps) == (local.status, local.steps)


def test_remote_agent_events_reach_the_parent_trace(env, tasks, tmp_path):
    remote = RemoteAgent("worldcurrent.agents.composable:make_landmark_agent", {})
    store = TraceStore(tmp_path)
    try:
        EpisodeRunner(env, remote, trace=store).run(TaskInstance(spec=tasks["reach_remembered"], seed=1), episode_id="ep")
    finally:
        remote.close()
    types = {e["type"] for e in store.read("ep").events}
    assert {"agent.decision", "memory.retrieval"} <= types
