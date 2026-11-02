"""Dexterous hands: finger commands in the kinematic model, top-down multi-finger grasp goals from presets, the hand's
surface at a preset, `holding` from fingers that stop short, and compliant fingers on the final approach."""

import numpy as np
import pytest

from hylo.core.robot import ArmSpec, HandSpec, RobotSpec
from hylo.core.types import Observation
from hylo.manip import Goal, Tools
from hylo.manip.datatypes import Table

import manip_testkit as tk

# the test arm with a three-finger hand instead of the parallel gripper: a thumb (one joint) and two fingers (two
# joints each) that curl towards the TCP about the hand's y axis
HAND = """
          <body name="hand" pos="0 0 0.08">
            <joint name="j7" axis="0 0 1" range="-2.9 2.9"/>
            <geom type="box" size="0.03 0.04 0.012" pos="0 0 0.012" rgba="0.3 0.3 0.3 1"/>
            <body name="thumb" pos="-0.03 0 0.024">
              <joint name="t0" axis="0 1 0" range="-0.5 1.2"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.008"/>
            </body>
            <body name="f1a" pos="0.03 0.02 0.024">
              <joint name="f1a" axis="0 1 0" range="-1.2 0.5"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.008"/>
              <body name="f1b" pos="0 0 0.04">
                <joint name="f1b" axis="0 1 0" range="-1.4 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
              </body>
            </body>
            <body name="f2a" pos="0.03 -0.02 0.024">
              <joint name="f2a" axis="0 1 0" range="-1.2 0.5"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.008"/>
              <body name="f2b" pos="0 0 0.04">
                <joint name="f2b" axis="0 1 0" range="-1.4 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
              </body>
            </body>
          </body>"""
# TCP 6 cm in front of the palm; +x approach along the hand's z, +y closing from the thumb (-x) to the fingers (+x)
TCP = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.06], [0, 0, 0, 1]]
OPEN = (-0.3, 0.3, 0.0, 0.3, 0.0)
POWER = (0.9, -0.9, -1.0, -0.9, -1.0)
JOINTS = ("t0", "f1a", "f1b", "f2a", "f2b")


@pytest.fixture(scope="module")
def hand_spec(tmp_path_factory):
    d = tmp_path_factory.mktemp("dex")
    body = tk.ARM_BODY
    start = body.index('<body name="hand"')
    end = body.index("</body>\n        </body>\n      </body>")
    body = body[:start] + HAND.strip() + "\n" + body[end + len("</body>\n"):]
    path = d / "dexarm.xml"
    path.write_text(f'<mujoco model="dex">{tk.HEADER}<worldbody>{body}</worldbody></mujoco>')
    hand = HandSpec(kind="dexterous", tcp_in_ee=TCP, open=OPEN, closed=POWER, max_width=0.08, joints=JOINTS,
                    presets={"power": POWER, "pinch": (0.7, -0.6, -0.6, -0.6, -0.6)})
    arm = ArmSpec(name="arm", model=str(path), joints=tuple(f"j{i}" for i in range(1, 8)), ee_frame="hand",
                  lower=(-2.9, -2.0, -2.9, -2.6, -2.9, -2.2, -2.9), upper=(2.9, 2.0, 2.9, 2.6, 2.9, 2.2, 2.9),
                  home=tk.ARM_HOME, hand=hand, action_slice=(0, 7), hand_slice=(7, 12))
    return RobotSpec(arms=(arm,), table_height=0.0)


class StuckFingers(tk.KinematicExecutor):
    """Fingers follow the command except that they stop where an object would be (at `stop`)."""

    def __init__(self, spec, stop=None):
        super().__init__(spec)
        self.stop = stop

    def run(self, actions):
        out = super().run(actions)
        if self.stop is not None:
            cmd = self.hand["arm"]
            self.hand["arm"] = np.where(np.abs(cmd) > np.abs(self.stop), self.stop, cmd)
        return out


def test_finger_commands_move_the_model_fingers(hand_spec):
    robot = Tools(hand_spec, tk.KinematicExecutor(hand_spec))
    kin = robot._models()[0]["arm"]
    kin.set_hand(POWER)
    np.testing.assert_allclose(kin.d.qpos[kin.finger_adr], POWER)
    kin.set_hand((5.0, 5.0, 5.0, 5.0, 5.0))                               # clipped to the joint ranges
    assert np.all(kin.d.qpos[kin.finger_adr] <= kin.finger_range[:, 1] + 1e-9)
    open_pts, power_pts = robot.hand_points(arm="arm"), robot.hand_points("power")
    assert open_pts.shape[1] == 3 and len(open_pts) > 50
    # closing curls the fingertips towards the TCP's closing axis (they end nearer the thumb along -y)
    assert power_pts[:, 1].max() < open_pts[:, 1].max() - 0.01


def test_hand_grasps_from_presets(hand_spec):
    robot = Tools(hand_spec, tk.KinematicExecutor(hand_spec))
    table = Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]))
    pts = tk.box_points((0.45, 0.1, 0.025), (0.02, 0.025, 0.025), yaw=0.3)
    goals = robot.hand_grasps(pts, table, kind="pinch")
    assert len(goals) == 4 and all(isinstance(g, Goal) for g in goals)
    for g in goals:
        T = np.asarray(g.tcp)
        assert np.allclose(T[:3, 0], [0, 0, -1], atol=1e-6)                # approach straight down
        assert abs(T[2, 3] - 0.025) < 0.004 and np.allclose(T[:2, 3], [0.45, 0.1], atol=0.004)
        assert np.allclose(g.pre[:3, 3], T[:3, 3] + [0, 0, 0.10])
        assert tuple(g.hand_before) == OPEN and tuple(g.hand) == hand_spec.arms[0].hand.presets["pinch"]
    # first: the closing axis nearest the base frame's +x (thumb on the near side); the box faces are at yaw 0.3
    y = np.asarray(goals[0].tcp)[:3, 1]
    assert y @ np.array([1.0, 0.0, 0.0]) > 0.9
    wide = tk.box_points((0.45, 0.1, 0.025), (0.06, 0.02, 0.025))           # 12 cm along one axis: two goals left
    assert len(robot.hand_grasps(wide, table)) == 2
    with pytest.raises(KeyError):
        robot.hand_grasps(pts, table, kind="hook")


def test_parallel_only_and_dexterous_only_verbs(hand_spec, arm_spec):
    table = Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]))
    pts = tk.box_points((0.45, 0.1, 0.025), (0.02, 0.02, 0.025))
    with pytest.raises(ValueError):
        Tools(hand_spec, tk.KinematicExecutor(hand_spec)).grasps(pts, table)
    with pytest.raises(ValueError):
        Tools(arm_spec, tk.KinematicExecutor(arm_spec)).hand_grasps(pts, table)


def test_holding_from_fingers_that_stop_short(hand_spec):
    ex = StuckFingers(hand_spec)
    robot = Tools(hand_spec, ex)
    assert robot.holding() is False
    tcp = robot.status()["tcp"]["arm"]
    robot.run(robot.plan(Goal("reach", tcp=tcp, hand=POWER)))
    assert robot.holding() is False                                         # closed on nothing: reached the command
    ex.stop = np.array([0.4, -0.4, -0.3, -0.4, -0.3])                       # an object in the way
    robot.run(robot.plan(Goal("reach", tcp=tcp, hand=POWER)))
    assert robot.holding() is True
    robot.run(robot.plan(Goal("reach", tcp=tcp, hand=OPEN)))
    assert robot.holding() is False                                         # an open command never holds


def test_fingers_may_touch_the_table_on_the_final_approach(hand_spec):
    robot = Tools(hand_spec, tk.KinematicExecutor(hand_spec))
    table = Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]))
    lowest = float(robot.hand_points(arm="arm")[:, 0].max())                # open fingers reach past the TCP
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    T[:3, 3] = [0.45, 0.1, lowest - 0.01]                                   # fingertips 1 cm into the table
    goal = robot.grasp_at(T)
    assert robot.plan(goal, table) is not None, robot.why()                 # compliant fingers: allowed
    T2 = T.copy()
    T2[2, 3] = -0.08                                                        # the palm itself in the table: refused
    assert robot.plan(robot.grasp_at(T2), table) is None
