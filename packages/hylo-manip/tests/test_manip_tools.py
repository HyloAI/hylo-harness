import numpy as np

from hylo.manip import Executor, Goal, Tools, View, api_reference
from hylo.manip.geometry import apply, inv
from hylo.manip.tools import DEXTEROUS_ONLY, VERBS

import manip_testkit as tk

RED, BLUE = (0.9, 0.1, 0.1, 1), (0.1, 0.2, 0.9, 1)


def make(tmp_path, arm_spec, **kw):
    objects = {"red_block": ((0.45, 0.08, 0.025), (0.025, 0.02, 0.025), RED, 0.3),
               "blue_block": ((0.42, -0.12, 0.04), (0.03, 0.03, 0.04), BLUE, 0.0)}
    ex = tk.StubExecutor(tmp_path, objects, arm_spec, hw=(120, 160), noise=0.0005)
    shown = []
    return ex, Tools(arm_spec, ex, show=lambda img, note: shown.append((img.shape, note)), **kw), shown


def test_pick_and_place_end_to_end(tmp_path, arm_spec):
    ex, robot, _ = make(tmp_path, arm_spec)
    assert isinstance(ex, Executor)
    view = robot.look()
    table = robot.table(view)
    det = robot.objects(view, robot.find(view, "red block"), table)[0]
    box = robot.box(det.points, table)
    plan = robot.plan(robot.grasps("arm", det.points, table, scene=view.points), table, view.points)
    assert plan is not None
    res = robot.run(plan)
    assert res["steps"] == len(plan.actions) and res["tcp_error"]["pos"] < 2e-3
    assert robot.holding("arm") is True and ex.attached is not None

    tcp_grasp = res["tcp"]["arm"]
    up = tcp_grasp.copy()
    up[2, 3] += 0.12
    res = robot.run(robot.plan(Goal("lift", "arm", tcp=up), table, view.points))
    check = robot.carried(robot.look(), det.points, tcp_grasp, res["tcp"]["arm"], table)
    assert check["carried"] and abs(check["lifted"] - 0.12) < 2e-3

    obj_in_tcp = inv(plan.goal.tcp) @ box.pose
    obj_points = apply(inv(box.pose), det.points)
    view2 = robot.look()
    place = robot.place_at("arm", obj_in_tcp, obj_points, (0.55, 0.2), yaw=0.0, table=table, hover=0.005)
    plan2 = robot.plan(place, table, view2.points)
    assert plan2 is not None, robot.why()
    robot.run(plan2)
    assert robot.holding("arm") is False and ex.attached is None
    T = ex.pose("red_block")
    assert np.allclose(T[:2, 3], [0.55, 0.2], atol=0.01)
    yaw = np.arctan2(T[1, 0], T[0, 0])
    assert abs((yaw + np.pi / 2) % np.pi - np.pi / 2) < np.radians(5)

    retreat = robot.status()["tcp"]["arm"].copy()
    retreat[2, 3] += 0.1
    assert robot.plan(Goal("retreat", "arm", tcp=retreat), table, robot.look().points) is not None
    home = robot.plan(Goal("home", "arm", joints=np.array(arm_spec.arms[0].home)), table)
    assert robot.run(home)["steps"] > 0
    assert np.allclose(robot.status()["q"]["arm"], arm_spec.arms[0].home)


def test_holding_false_when_closed_on_nothing(tmp_path, arm_spec):
    ex, robot, _ = make(tmp_path, arm_spec)
    tcp = robot.status()["tcp"]["arm"]
    robot.run(robot.plan(Goal("reach", "arm", tcp=tcp, hand=(0.0,))))
    assert robot.status()["hand"]["arm"][0] < 0.01 and robot.holding("arm") is False


def test_status_look_show_and_stale_plans(tmp_path, arm_spec):
    ex, robot, shown = make(tmp_path, arm_spec)
    st = robot.status()
    assert {"step", "max_steps", "instruction", "ended", "q", "hand", "tcp"} <= set(st)
    assert st["tcp"]["arm"].shape == (4, 4)
    view = robot.look()
    assert isinstance(view, View) and view.cams["front"].depth.shape == (120, 160) and len(view.points) > 1000
    robot.show(view, "all cameras")
    robot.show(view.cams["front"].rgb)
    assert shown == [((120, 480, 3), "all cameras"), ((120, 160, 3), "")]
    tcp = st["tcp"]["arm"].copy()
    tcp[1, 3] += 0.1
    p1 = robot.plan(Goal("reach", "arm", tcp=tcp))
    tcp[1, 3] -= 0.2
    robot.run(robot.plan(Goal("reach", "arm", tcp=tcp)))
    stale = robot.run(p1)
    assert stale["steps"] == 0 and "plan again" in stale["error"]


def test_api_reference_lists_every_verb(arm_spec):
    robot = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    ref = api_reference(robot)
    for verb in VERBS:
        assert verb == "Goal" or getattr(Tools, verb).__doc__
        assert (f"robot.{verb}(" in ref) == (verb not in DEXTEROUS_ONLY)     # multi-finger verbs: dexterous hands
    assert "arm 'arm': 7 joints, parallel hand" in ref and "max width 0.080 m" in ref
    assert "self" not in ref.split("## Types")[0].replace("itself", "")
    assert len(ref) < 12000


def test_reference_example_runs(tmp_path, arm_spec):
    from hylo.manip.tools import EXAMPLE

    ex, robot, _ = make(tmp_path, arm_spec)
    exec(EXAMPLE, {"robot": robot, "np": np, "print": lambda *a: None})
    assert np.allclose(ex.pose("red_block")[:2, 3], [0.5, 0.2], atol=0.01) and ex.attached is None


def test_printing_toolkit_values_does_not_dump_arrays():
    from hylo.manip.datatypes import Box, CameraView, Detection, Mask, View

    m = Mask(label="red block", camera="front", mask=np.ones((240, 320), bool), score=1.0, instance=4)
    v = View(cams={"front": CameraView(rgb=np.zeros((240, 320, 3), np.uint8), depth=None, K=None,
                                       T_world_cam=np.eye(4))}, points=np.zeros((5000, 3)), segments={4: "red block"})
    d = Detection(label="red block", score=1.0, points=np.ones((900, 3)))
    b = Box(center=np.array([0.1234567, 0.2, 0.3]), size=np.array([0.045, 0.045, 0.045]), yaw=0.123456, bottom=0.0)
    texts = [repr(m), repr(v), repr(d), repr(b), repr([m, m])]
    assert all(len(t) < 300 for t in texts)
    assert "pixels=76800" in texts[0] and "(5000, 3)" in texts[1] and "900 points" in texts[2] and "0.123" in texts[3]
