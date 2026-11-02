import numpy as np

from hylo.manip import Table, Tools
from hylo.manip.geometry import apply

import manip_testkit as tk

TABLE = Table(0.0, np.r_[0, 0, 1.0])


def tools_for(spec):
    return Tools(spec, tk.KinematicExecutor(spec))


def test_top_down_grasp_across_the_narrow_side(arm_spec):
    tools = tools_for(arm_spec)
    yaw = 0.5
    pts = tk.box_points((0.45, 0.05, 0.025), (0.03, 0.02, 0.025), yaw=yaw)
    goals = tools.grasps("arm", pts, TABLE)
    assert goals and [g.rank for g in goals] == sorted(g.rank for g in goals)
    g = goals[0]
    assert np.allclose(g.tcp[:3, 0], [0, 0, -1], atol=1e-6)                      # approach straight down
    assert abs(abs(g.tcp[:3, 1] @ [-np.sin(yaw), np.cos(yaw), 0]) - 1) < 1e-6    # closing across the 4 cm side
    assert np.allclose(g.tcp[:2, 3], [0.45, 0.05], atol=0.003)
    assert 0.01 < g.tcp[2, 3] < 0.05
    assert np.allclose(g.pre[:3, 3], g.tcp[:3, 3] + [0, 0, 0.10])
    assert tuple(g.hand_before) == (1.0,) and tuple(g.hand) == (0.0,)
    assert tools.plan(goals, TABLE) is not None


def test_width_check_and_side_grasps(arm_spec):
    tools = tools_for(arm_spec)
    wide = tk.box_points((0.45, 0.0, 0.03), (0.06, 0.05, 0.03))           # 12 x 10 cm: wider than the hand
    assert tools.grasps("arm", wide, TABLE) == []
    tall = tk.box_points((0.45, 0.0, 0.08), (0.02, 0.02, 0.08))           # 16 cm tall post
    kinds = {("top" if abs(g.tcp[2, 0]) > 0.9 else "side") for g in tools.grasps("arm", tall, TABLE)}
    assert kinds == {"top", "side"}
    side = [g for g in tools.grasps("arm", tall, TABLE) if abs(g.tcp[2, 0]) < 0.1]
    assert all(g.tcp[:3, 0] @ np.r_[0.45, 0, 0] / 0.45 > 0.3 for g in side)  # approached from the robot's side


def test_grasps_avoid_scene_points(arm_spec):
    tools = tools_for(arm_spec)
    block = tk.box_points((0.45, 0.0, 0.025), (0.03, 0.02, 0.025))
    goals = tools.grasps("arm", block, TABLE)
    assert any(abs(g.tcp[1, 1]) > 0.99 for g in goals)                    # closing along y is possible
    neighbour = tk.box_points((0.45, 0.055, 0.03), (0.04, 0.01, 0.03))    # a wall 2.5 cm beside the block in +y
    blocked = tools.grasps("arm", block, TABLE, scene=neighbour)
    assert all(abs(g.tcp[1, 1]) < 0.99 for g in blocked)                  # the pads would hit it


def test_grasp_at_builds_pre_pose(arm_spec):
    tools = tools_for(arm_spec)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = [0, 0, -1], [1, 0, 0], [0, -1, 0]
    T[:3, 3] = [0.4, 0.1, 0.05]
    g = tools.grasp_at("arm", T, hand=(0.3,))
    assert np.allclose(g.pre[:3, 3], [0.4, 0.1, 0.15]) and tuple(g.hand) == (0.3,) and g.verb == "grasp"


def test_place_at_geometry(arm_spec):
    ex = tk.KinematicExecutor(arm_spec)
    tools = Tools(arm_spec, ex)
    T_tcp_now = tools.status()["tcp"]["arm"]
    # object frame 2 cm below the TCP, rotated 0.3 rad about the TCP's x (approach) axis
    c, s = np.cos(0.3), np.sin(0.3)
    oit = np.eye(4)
    oit[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    oit[:3, 3] = [0.02, 0, 0]
    T_obj_now = T_tcp_now @ oit
    obj_pts = tk.box_points((0, 0, 0), (0.03, 0.02, 0.025))
    table = Table(0.05, np.r_[0, 0, 1.0])
    g = tools.place_at("arm", oit, obj_pts, (0.5, -0.1), yaw=1.0, table=table, hover=0.01)
    T_obj = g.tcp @ oit
    world = apply(T_obj, obj_pts)
    assert abs(world[:, 2].min() - 0.06) < 1e-9                              # lowest point at table + hover
    assert np.allclose(T_obj[:2, 3], [0.5, -0.1])
    assert abs(np.arctan2(T_obj[1, 0], T_obj[0, 0]) - 1.0) < 1e-6                # heading of its x axis
    tilt_now = np.arccos(np.clip(T_obj_now[2, 2], -1, 1))
    assert abs(np.arccos(np.clip(T_obj[2, 2], -1, 1)) - tilt_now) < 1e-9
    assert np.allclose(g.pre[:3, 3], g.tcp[:3, 3] + [0, 0, 0.10]) and tuple(g.hand) == (1.0,) and g.hand_before is None
    keep = tools.place_at("arm", oit, obj_pts, (0.5, -0.1), table=table)    # yaw None: heading unchanged
    assert np.allclose((keep.tcp @ oit)[:3, :3], T_obj_now[:3, :3])


def test_panda_grasps_on_a_block(panda_spec):
    tools = tools_for(panda_spec)
    pts = tk.box_points((0.5, -0.1, 0.025), (0.03, 0.02, 0.025), yaw=-0.7)
    goals = tools.grasps("arm", pts, TABLE)
    assert goals and abs(goals[0].tcp[2, 0] + 1) < 1e-6
    p = tools.plan(goals, TABLE)
    assert p is not None and p.info["method"].endswith("line")
    r = tools.run(p)
    assert r["tcp_error"]["pos"] < 1e-3 and r["tcp_error"]["rot"] < 0.01
