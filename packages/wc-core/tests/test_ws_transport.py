import urllib.request

import numpy as np
import pytest

pytest.importorskip("websockets")

from worldcurrent.core.errors import EnvironmentFault  # noqa: E402
from worldcurrent.core.kernel import EpisodeRunner  # noqa: E402
from worldcurrent.core.task import TaskInstance  # noqa: E402
from worldcurrent.core.remote.ws import WSAgent, WSEnvironment, make_server  # noqa: E402
from worldcurrent.testing.conformance import check_environment  # noqa: E402
from worldcurrent.testing.point_reach import PointReachEnv, reach_tasks  # noqa: E402

ENV = "worldcurrent.testing.point_reach:PointReachEnv"
AGENT = "worldcurrent.agents.baseline:PControllerAgent"


@pytest.fixture
def env_server():
    srv = make_server("env", ENV, {}, port=0).start()
    yield srv
    srv.shutdown()


def test_episode_over_websocket_matches_local(env_server):
    task = TaskInstance(spec=reach_tasks()[0], seed=3)
    agent_srv = make_server("agent", AGENT, {"kp": 2.0}, port=0).start()
    try:
        env, agent = WSEnvironment(env_server.url), WSAgent(agent_srv.url)
        remote = EpisodeRunner(env, agent).run(task).outcome
        from worldcurrent.agents.baseline import PControllerAgent

        local = EpisodeRunner(PointReachEnv(), PControllerAgent(kp=2.0)).run(task).outcome
        assert (remote.status, remote.steps) == (local.status, local.steps)
        np.testing.assert_array_equal(env.reset(task).state["pos"], PointReachEnv().reset(task).state["pos"])
        env.close()
        agent.close()
    finally:
        agent_srv.shutdown()


def test_remote_env_passes_conformance_and_connections_are_isolated(env_server):
    rep = check_environment(lambda: WSEnvironment(env_server.url), reach_tasks()[1])
    assert rep.ok, rep.summary()
    a, b = WSEnvironment(env_server.url), WSEnvironment(env_server.url)
    a.reset(TaskInstance(spec=reach_tasks()[0], seed=0))
    with pytest.raises(EnvironmentFault):                 # b has its own, never-reset env instance
        b.step(__import__("worldcurrent.core.types", fromlist=["Action"]).Action(data=np.zeros(2)))
    a.close()
    b.close()


def test_token_and_healthz():
    srv = make_server("env", ENV, {}, port=0, token="s3cret").start()
    try:
        assert urllib.request.urlopen(srv.url.replace("ws://", "http://") + "/healthz").read() == b"OK\n"
        with pytest.raises(EnvironmentFault):
            WSEnvironment(srv.url, connect_timeout_s=0.5)
        env = WSEnvironment(srv.url, token="s3cret")
        assert env.spec().name
        env.close()
    finally:
        srv.shutdown()
