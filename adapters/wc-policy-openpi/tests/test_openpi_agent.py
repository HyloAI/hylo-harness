import threading

import msgpack
import numpy as np
import pytest
import websockets.sync.server as wss

from worldcurrent.core.agent import EpisodeContext
from worldcurrent.core.errors import AgentError
from worldcurrent.core.kernel import EpisodeRunner
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.remote import wire
from worldcurrent.policies.openpi.agent import OpenPIAgent, resize_with_pad


def test_wire_format_is_openpis():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    # the map layout openpi_client.msgpack_numpy produces, spelled out by hand
    expected = msgpack.packb({"a": {b"__ndarray__": True, b"data": arr.tobytes(), b"dtype": "<f4", b"shape": (2, 3)},
                              "s": {b"__npgeneric__": True, b"data": 2.5, b"dtype": "<f4"}})
    # np.float64 subclasses float, so msgpack packs it natively (in openpi too); float32 takes the generic path
    assert wire.packb({"a": arr, "s": np.float32(2.5)}) == expected
    back = wire.unpackb(expected)
    assert back["a"].dtype == np.float32 and np.array_equal(back["a"], arr) and back["s"].dtype == np.float32
    with pytest.raises(TypeError):
        wire.packb({"x": np.array([object()])})


def test_resize_with_pad_keeps_aspect():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = resize_with_pad(img, 224, 224)
    assert out.shape == (224, 224, 3) and out[0, 0].sum() == 0 and out[112, 112].sum() == 765


class FakeOpenPIServer:
    """Behaves like openpi's WebsocketPolicyServer: metadata first, then one action dict per observation."""

    def __init__(self, horizon=10, fail=False):
        self.seen, self.horizon, self.fail = [], horizon, fail
        self.srv = wss.serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None)
        self.url = f"ws://127.0.0.1:{self.srv.socket.getsockname()[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def _handle(self, conn):
        conn.send(wire.packb({"policy": "fake-pi05", "action_horizon": self.horizon}))
        for frame in conn:
            obs = wire.unpackb(frame)
            self.seen.append(obs)
            if self.fail:
                conn.send("Traceback: boom")
                return
            target = obs["observation/state"][:2]
            act = np.zeros((self.horizon, 7), np.float32)
            act[:, :2] = target
            conn.send(wire.packb({"actions": act, "server_timing": {"infer_ms": 3.0}}))


def _double_obs_env():
    from worldcurrent.testing.point_reach import PointReachEnv, reach_tasks
    return PointReachEnv(), TaskInstance(spec=reach_tasks()[0], seed=1)


def test_agent_maps_libero_inputs_and_executes_chunks():
    from worldcurrent.core.types import Observation

    srv = FakeOpenPIServer()
    try:
        agent = OpenPIAgent(srv.url)
        obs = Observation(state={"eef_pos": np.array([0.1, 0.2, 0.3]), "eef_axis_angle": np.zeros(3),
                                 "gripper_qpos": np.array([0.04, -0.04])},
                          images={"agentview": np.zeros((256, 256, 3), np.uint8), "wrist": np.zeros((256, 256, 3), np.uint8)},
                          instruction="put the bowl on the plate")
        chunk = agent.act(obs)
        req = srv.seen[-1]
        assert set(req) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
        assert req["observation/image"].shape == (224, 224, 3) and req["observation/state"].dtype == np.float32
        assert req["observation/state"].shape == (8,) and req["prompt"] == "put the bowl on the plate"
        assert chunk.actions.shape == (10, 7) and chunk.execution_steps == 5
        assert agent.spec().components["server.policy"] == "fake-pi05"
        agent.close()
    finally:
        srv.srv.shutdown()


def test_server_error_frame_becomes_agent_error():
    srv = FakeOpenPIServer(fail=True)
    try:
        agent = OpenPIAgent(srv.url, inputs={"observation/state": "state:pos", "prompt": "instruction"}, preset=None)
        env, task = _double_obs_env()
        with pytest.raises(AgentError, match="boom"):
            agent.act(env.reset(task))
    finally:
        srv.srv.shutdown()


def test_runs_in_the_kernel_with_custom_mapping():
    srv = FakeOpenPIServer(horizon=4)
    try:
        agent = OpenPIAgent(srv.url, preset=None, inputs={"observation/state": "state:pos,vel", "prompt": "instruction"},
                            action_dim=2, execution_steps=2)
        env, task = _double_obs_env()
        res = EpisodeRunner(env, agent).run(task)
        assert res.outcome.steps > 0 and len(srv.seen) >= res.outcome.steps // 2
        agent.close()
    finally:
        srv.srv.shutdown()
