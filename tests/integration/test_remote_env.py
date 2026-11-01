import numpy as np
import pytest

from worldcurrent.agents.baseline import PControllerAgent
from worldcurrent.core import codec
from worldcurrent.core.kernel import EpisodeRunner
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.types import Action
from worldcurrent.core.remote.client import RemoteEnvironment


def test_codec_rejects_object_arrays():
    with pytest.raises(TypeError):
        codec.encode(np.array([object()]))


def test_remote_env_matches_in_process(env, tasks):
    remote = RemoteEnvironment("worldcurrent.testing.point_reach:PointReachEnv")
    try:
        assert remote.spec() == env.spec()
        for name in ("reach_random", "reach_push"):
            inst = TaskInstance(spec=tasks[name], seed=7)
            local = EpisodeRunner(env, PControllerAgent(kp=2.0)).run(inst).outcome
            rem = EpisodeRunner(remote, PControllerAgent(kp=2.0)).run(inst).outcome
            assert (local.status, local.steps) == (rem.status, rem.steps)
    finally:
        remote.close()
