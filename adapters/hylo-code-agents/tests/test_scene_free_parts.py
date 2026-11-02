"""Free-mode library modules on their own, with no model and no rendering: the recorded-episode helpers
(scene_episode), the checker report as feedback (scene_free_prompt) and the privileged free toolkit
(scene_free_tools) over a fixed privileged state."""

import numpy as np
import pytest

from hylo.codeagents.scene_check import CheckReport, problem
from hylo.codeagents.scene_episode import FrameRecorder, write_mp4
from hylo.codeagents.scene_free_prompt import EXAMPLE_1, EXAMPLE_2, SYSTEM, feedback_text
from hylo.codeagents.scene_free_tools import FREE_REFERENCE, PrivilegedFreeTools
from hylo.core.robot import ArmSpec, HandSpec, RobotSpec
from hylo.core.types import Observation


class _Obs:
    def __init__(self, img):
        self.images = {"front": img}


class _Env:
    def __init__(self):
        self.closed = False
        self.extra_attr = 7

    def spec(self):
        return "spec"

    def reset(self, task):
        return _Obs(np.zeros((4, 4, 3), np.uint8))

    def step(self, action):
        class R:
            observation = _Obs(np.full((4, 4, 3), 9, np.uint8))
        return R()

    def close(self):
        self.closed = True


def test_the_frame_recorder_keeps_one_camera_and_passes_everything_else():
    env = FrameRecorder(_Env(), "front")
    env.reset(None)
    env.step(None)
    assert len(env.frames) == 2 and env.frames[1][0, 0, 0] == 9 and env.spec() == "spec" and env.extra_attr == 7
    env.close()
    assert env._env.closed


def test_write_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = [np.full((32, 48, 3), i * 10, np.uint8) for i in range(10)]
    res = write_mp4(frames, tmp_path / "a.mp4", fps=10)
    assert res["written"] and res["frames"] == 10 and (tmp_path / "a.mp4").stat().st_size == res["bytes"]
    assert write_mp4([], tmp_path / "b.mp4") == {"written": False, "reason": "no frames"}


def test_feedback_lists_problems_warnings_and_the_settled_scene():
    bad = CheckReport(ok=False, problems=(problem("unstable", "cube moved 3 cm", "cube"),), stages=("program", "load",
                                                                                                  "physics"),
                      settled={"cube": {"pos": [0, 0, 0.02]}})
    text = feedback_text(bad)
    assert "rejected the program (stages run: program, load, physics)" in text and "1. [unstable] cube moved" in text
    assert '- cube: {"pos": [0, 0, 0.02]}' in text and text.endswith("complete corrected program.")
    good = CheckReport(ok=True, warnings=(problem("unverified", "no demo", severity="warning"),),
                       stages=("program", "load", "physics", "reach", "satisfiable"))
    text = feedback_text(good)
    assert text.startswith("The checker passed every stage") and "- warning [unverified] no demo" in text
    assert "{robots}" in SYSTEM and "def demo(sim)" in EXAMPLE_1 and "composite" in EXAMPLE_2


def _robot(arms):
    hand = HandSpec(kind="parallel", tcp_in_ee=np.eye(4).tolist(), open=(1.0,), closed=(0.0,), max_width=0.08)
    return RobotSpec(arms=tuple(ArmSpec(name=a, model="none.xml", joints=("j",), ee_frame="ee", hand=hand, lower=(-1.0,),
                                        upper=(1.0,), home=(0.0,), action_slice=(2 * i, 2 * i + 1),
                                        hand_slice=(2 * i + 1, 2 * i + 2)) for i, a in enumerate(arms)),
                     table_height=0.0)


STATE = {"bodies": {"box": {"pos": [0, 0, 0.02], "quat": [1, 0, 0, 0], "bounds": [[-0.02, -0.02, 0], [0.02, 0.02, 0.04]],
                            "top": 0.04, "free": True, "mocap": False, "parent": None, "label": "box"}},
         "cables": {"rope": {"nodes": [[0, 0, 0.01], [0.03, 0, 0.01], [0.06, 0, 0.01]], "bodies": ["a", "b", "c"],
                             "radius": 0.01, "anchored": False, "label": "rope"}},
         "flexes": {"cloth": {"vertices": [[0, 0, 0], [0.1, 0, 0]], "bodies": None, "radius": 0.004, "dim": 2,
                              "label": "cloth"}},
         "joints": {"hinge": {"type": "hinge", "value": 0.1, "range": [0, 1], "units": "rad", "body": "lid"}},
         "sites": {"knob": [0, 0, 0.1]}, "table": {"height": 0.0, "x": [-0.5, 0.5], "y": [-0.4, 0.4]}}


class _Exec:
    def observation(self):
        return Observation(state={}, extra={"privileged": STATE})


def test_the_privileged_free_toolkit_reads_bodies_cables_cloth_and_joints():
    tools = PrivilegedFreeTools(_robot(["arm"]), _Exec())
    s = tools.scene()
    assert s["cables"]["rope"]["nodes"].shape == (3, 3) and s["flexes"]["cloth"]["vertices"].shape == (2, 3)
    assert isinstance(s["bodies"]["box"]["bounds"][0], np.ndarray) and s["joints"]["hinge"]["value"] == 0.1
    box = tools.points("box", spacing=0.01)
    assert np.allclose(box.min(0), [-0.02, -0.02, 0]) and np.allclose(box.max(0), [0.02, 0.02, 0.04])
    rope = tools.points("rope", spacing=0.01)
    assert rope[:, 0].max() == pytest.approx(0.07) and rope[:, 2].max() == pytest.approx(0.02)
    assert tools.points("cloth").shape == (2, 3)
    with pytest.raises(KeyError, match="no body, cable or flex 'ghost'"):
        tools.points("ghost")
    T = tools.top_down([0.1, 0.2, 0.03], yaw=np.pi / 2)
    assert np.allclose(T[:3, 0], [0, 0, -1]) and np.allclose(T[:3, 1], [0, 1, 0], atol=1e-9)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.03]) and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-9
    table = tools.scene_table()
    assert table.height == 0.0 and np.allclose(table.extent, [1.0, 0.8]) and tools.scene_table(0.2).height == 0.2
    ref = tools.extra_reference()
    assert ref.startswith(FREE_REFERENCE.splitlines()[0]) and "one arm" in ref
    assert "two arms" in PrivilegedFreeTools(_robot(["left", "right"]), _Exec()).extra_reference()
