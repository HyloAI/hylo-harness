"""Fixes found by running programs on mujoco/manip-v0: what the robot itself looks like in observed points, a held
object carried as attached geometry, placing on a surface other than the table, straight lines near joint limits
and flat markings on the table; a faster voxel filter (perception time per look)."""

import numpy as np

from hylo.manip import Box, CameraView, Goal, Mask, Table, Tools, View
from hylo.manip.geometry import apply, inv
from hylo.manip.perception import detect_objects

import manip_testkit as tk

TABLE = Table(0.0, np.r_[0, 0, 1.0])


def down(x, y, z, yaw=0.0):
    T = np.eye(4)
    a, c = np.array([0, 0, -1.0]), np.array([np.cos(yaw), np.sin(yaw), 0.0])
    T[:3, :3] = np.stack([a, c, np.cross(a, c)], axis=1)
    T[:3, 3] = [x, y, z]
    return T


def test_robot_filter_covers_the_fixed_base_and_visual_geoms(tmp_path):
    # the camera sees visual geoms; a visual-only shell 2 cm outside link2's collision capsule and the fixed base
    # (no joint, so no moving collision geom) must both be removed from observed points
    body = tk.ARM_BODY.replace('<geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.045"/>',
                               '<geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.045"/>'
                               '<geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.065" contype="0" '
                               'conaffinity="0" group="2"/>')
    path = tmp_path / "shell.xml"
    path.write_text(f'<mujoco model="shell">{tk.HEADER}<worldbody>{body}</worldbody></mujoco>')
    spec = tk.arm_spec(path)
    tools = Tools(spec, tk.KinematicExecutor(spec))
    kin, cols, planner = tools._models()
    q, hand = {"arm": np.array(spec.arms[0].home)}, {"arm": np.array([1.0])}
    seen = planner.robot_points(q, hand)["arm"]
    base = np.c_[0.07 * np.cos(np.linspace(0, 6, 40)), 0.07 * np.sin(np.linspace(0, 6, 40)), np.full(40, 0.05)]
    arm = kin["arm"]
    arm.set_q(q["arm"])
    arm.forward()
    link2 = arm.m.body("link2").id
    local = np.c_[np.full(30, 0.065), np.zeros(30), np.linspace(0.05, 0.3, 30)]
    shell = apply(arm.base, local @ arm.d.xmat[link2].reshape(3, 3).T + arm.d.xpos[link2])
    obs = planner.obstacles("arm", q, hand, 0.0, np.vstack([base, shell]))
    assert obs.points is None or len(obs.points) == 0
    assert np.min(np.linalg.norm(seen[:, None] - base[None], axis=2), axis=0).max() < 0.02


def held_box_tools(arm_spec):
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    tools.run(tools.plan(Goal("reach", "arm", tcp=down(0.42, 0.15, 0.2))))
    held = tk.box_points((0.03, 0.0, 0.0), (0.03, 0.02, 0.02), spacing=0.008)     # 3 cm along +x: below the TCP
    return tools, held


def test_held_object_is_attached_geometry(arm_spec):
    tools, held = held_box_tools(arm_spec)
    goal_tcp = down(0.42, -0.15, 0.2)
    shelf = np.c_[np.random.default_rng(0).uniform([0.35, -0.22], [0.49, -0.08], (600, 2)), np.full(600, 0.15)]
    T_now = tools.status()["tcp"]["arm"]
    seen_held = apply(T_now, held)                       # the held object as observed where it is now
    scene = np.vstack([shelf, seen_held])
    free = tools.plan(Goal("reach", "arm", tcp=goal_tcp), TABLE, scene)
    assert free is not None                              # the hand alone clears the shelf 5 cm below the TCP
    assert tools.plan(Goal("reach", "arm", tcp=goal_tcp, attached=held), TABLE, scene) is None
    assert "goal in collision (points)" in tools.why()   # the held object would go through it
    higher = tools.plan(Goal("reach", "arm", tcp=down(0.42, -0.15, 0.26), attached=held), TABLE, scene)
    assert higher is not None                            # and its own observed points are not obstacles


def test_place_at_rests_on_the_surface_below(arm_spec):
    tools, _ = held_box_tools(arm_spec)
    T_now = tools.status()["tcp"]["arm"]
    box = Box(center=T_now[:3, 3] + [0, 0, -0.03], size=np.array([0.04, 0.04, 0.04]), yaw=0.3,
              bottom=T_now[2, 3] - 0.05)
    oit = inv(T_now) @ box.pose
    block = tk.box_points((0.5, -0.1, 0.0225), (0.0225, 0.0225, 0.0225), yaw=0.2)
    plane = np.c_[np.random.default_rng(1).uniform([0.3, -0.3], [0.7, 0.1], (3000, 2)), np.zeros(3000)]
    scene = np.vstack([block, plane])
    g = tools.place_at("arm", oit, box, (0.5, -0.1), table=TABLE, hover=0.005, scene=scene)
    bottom = (g.tcp @ oit)[2, 3] - 0.02                  # a Box counts its hidden bottom
    assert abs(bottom - 0.05) < 0.003 and g.attached is not None and len(g.attached) > 50
    beside = tools.place_at("arm", oit, box, (0.62, -0.1), table=TABLE, hover=0.005, scene=scene)
    assert abs((beside.tcp @ oit)[2, 3] - 0.02 - 0.005) < 0.003    # nothing under it: the table
    given = tools.place_at("arm", oit, box, (0.5, -0.1), table=TABLE, hover=0.005, support=0.1)
    assert abs((given.tcp @ oit)[2, 3] - 0.02 - 0.105) < 1e-9


def test_straight_line_subdivides_fast_joint_steps(arm_spec):
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    tools.run(tools.plan(Goal("reach", "arm", tcp=down(0.45, 0.0, 0.15))))
    tools.config.line_jump = 0.03                         # each 0.05 rad step of the turn moves joint 7 by 0.05
    turn = down(0.45, 0.0, 0.15, yaw=1.2)
    p = tools.plan(Goal("lift", "arm", tcp=turn), TABLE)
    assert p is not None and p.info["method"] == "line"
    q = p.actions[:, :7]
    assert np.max(np.abs(np.diff(q[:-20], axis=0))) < 0.03 + 1e-9 or p.info["waypoints"] > 25


def test_flat_markings_are_detected():
    H, W, f = 80, 80, 80.0
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])               # looking straight down from 1 m
    T[:3, 3] = [0.0, 0.0, 1.0]
    depth = np.ones((H, W), np.float32)
    mask = np.zeros((H, W), bool)
    mask[20:50, 30:60] = True                            # a square zone drawn on the table
    cam = CameraView(rgb=np.zeros((H, W, 3), np.uint8), depth=depth, K=K, T_world_cam=T)
    view = View(cams={"top": cam}, points=np.zeros((0, 3)))
    dets = detect_objects(view, [Mask(label="green zone", camera="top", mask=mask)], TABLE)
    assert len(dets) == 1
    pts = dets[0].points
    assert np.all(np.abs(pts[:, 2]) < 0.01) and np.ptp(pts[:, 0]) > 0.3 and np.ptp(pts[:, 1]) > 0.3


def test_voxel_down_is_the_per_voxel_mean():
    from hylo.manip.geometry import voxel_down

    pts = np.random.default_rng(2).uniform([-0.6, -0.6, -0.1], [0.6, 0.6, 0.8], (20000, 3))
    out = voxel_down(pts, 0.05)
    keys = np.floor(pts / 0.05).astype(int)
    uniq, inv_idx = np.unique(keys, axis=0, return_inverse=True)
    ref = np.array([pts[inv_idx.ravel() == i].mean(0) for i in range(len(uniq))])
    assert out.shape == ref.shape and np.allclose(out, ref)
