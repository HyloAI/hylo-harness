"""Dexterous grasping and retargeting on a synthetic three-finger hand (MJCF written here; no downloads, no
rendering): grasp-quality math on analytic cases, object geometry, the hand model, grasp planning, trajectory
bending onto a grasp, retargeting with known answers, and the toolkit verbs."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from hylo.core.robot import HandSpec
from hylo.manip.dexgrasp import DexGraspPlanner, ObjectGeom, PlannerConfig, mesh_surface, to_goals
from hylo.manip.geometry import frame, pose, rotz
from hylo.manip.grasp_quality import (cone_edges, contact_wrenches, epsilon, force_closure, grasp_epsilon,
                                      holding_effort)
from hylo.manip.handmodel import HandModel
from hylo.manip.handpath import bend_onto_grasp, blend_pose
from hylo.manip.retarget import Retargeter, human_frame, keypoints_from_robot

# a palm with a thumb (2 joints) and two fingers (2 joints each) curling about the palm's y axis towards each other
HAND_XML = """<mujoco model="hand3"><compiler angle="radian" autolimits="true"/>
<default><joint damping="0.05" armature="0.001"/><geom rgba="0.6 0.6 0.6 1"/></default>
<worldbody>
  <body name="palm">
    <geom type="box" size="0.035 0.03 0.01" pos="0 0 0.01"/>
    <site name="wrist" pos="0 0 -0.02"/>
    <body name="thumb_a" pos="-0.03 0 0.02">
      <joint name="th0" axis="0 1 0" range="-0.6 1.3"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.008"/>
      <body name="thumb_b" pos="0 0 0.04">
        <joint name="th1" axis="0 1 0" range="0 1.4"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
      </body>
    </body>
    <body name="index_a" pos="0.03 0.018 0.02">
      <joint name="ix0" axis="0 1 0" range="-1.3 0.6"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.008"/>
      <body name="index_b" pos="0 0 0.045">
        <joint name="ix1" axis="0 1 0" range="-1.4 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
      </body>
    </body>
    <body name="middle_a" pos="0.03 -0.018 0.02">
      <joint name="md0" axis="0 1 0" range="-1.3 0.6"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.045" size="0.008"/>
      <body name="middle_b" pos="0 0 0.045">
        <joint name="md1" axis="0 1 0" range="-1.4 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
      </body>
    </body>
  </body>
</worldbody>
<actuator>
  <position joint="th0" kp="3"/><position joint="th1" kp="3"/><position joint="ix0" kp="3"/>
  <position joint="ix1" kp="3"/><position joint="md0" kp="3"/><position joint="md1" kp="3"/>
</actuator></mujoco>"""
JOINTS = ("th0", "th1", "ix0", "ix1", "md0", "md1")
OPEN = (-0.45, 0.1, 0.45, -0.1, 0.45, -0.1)
POWER = (1.1, 1.0, -1.1, -1.0, -1.1, -1.0)
PINCH = (0.8, 0.5, -0.8, -0.5, -0.8, -0.5)
# TCP 6 cm out of the palm (+z of the palm), +x approach along the palm's z, +y closing from the thumb to the fingers
TCP = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.06], [0, 0, 0, 1]]
SPEC = HandSpec(kind="dexterous", tcp_in_ee=TCP, open=OPEN, closed=POWER, max_width=0.1, joints=JOINTS,
                presets={"power": POWER, "pinch": PINCH, "flat": (0.0,) * 6})


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    path = tmp_path_factory.mktemp("hand3") / "hand3.xml"
    path.write_text(HAND_XML)
    return HandModel(str(path), "palm", SPEC, name="hand3_right")


def box_parts(half, centre=(0.0, 0.0, 0.0), yaw=0.0):
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * half
    return (c @ rotz(yaw).T) + centre


# ---------------------------------------------------------------------------------------------------- quality
def test_epsilon_of_the_cross_polytope_is_analytic():
    W = np.vstack([np.eye(6), -np.eye(6)])
    assert epsilon(W) == pytest.approx(1 / np.sqrt(6), rel=1e-9)
    shifted = W + [0.6, 0, 0, 0, 0, 0]                    # nearest facets x1 + ... = 1 now 0.4 / sqrt(6) away
    assert epsilon(shifted) == pytest.approx(0.4 / np.sqrt(6), rel=1e-9)
    assert epsilon(W + [1.5, 0, 0, 0, 0, 0]) < 0          # origin outside: negative
    assert epsilon(W[:, :5].repeat(1, axis=0) @ np.eye(5, 6)) == 0.0   # flat set (rank 5): no closure


def test_friction_cone_edges():
    E = cone_edges([0.0, 0.0, 1.0], 0.5, 8)
    assert E.shape == (8, 3) and np.allclose(E[:, 2], 1.0)
    assert np.allclose(np.linalg.norm(E[:, :2], axis=1), 0.5)
    assert cone_edges([0, 0, 1.0], 0.0).shape == (1, 3)


def test_force_closure_on_analytic_contact_sets():
    c = np.zeros(3)
    # two antipodal frictional contacts cannot resist torque about the line joining them
    P2, N2 = np.array([[0.03, 0, 0], [-0.03, 0, 0]]), np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    assert not force_closure(contact_wrenches(P2, N2, c, 0.5, 8, 0.03))
    # three contacts 120 degrees apart on a sphere's equator, with friction: force closure
    t = np.radians([0, 120, 240])
    P3 = 0.03 * np.stack([np.cos(t), np.sin(t), np.zeros(3)], 1)
    N3 = -P3 / 0.03
    e3 = grasp_epsilon(P3, N3, c, 0.5, 8, 0.03)
    assert e3 > 0
    assert grasp_epsilon(P3, N3, c, 0.0, 8, 0.03) <= 0     # frictionless: no closure
    assert grasp_epsilon(P3, N3, c, 0.9, 8, 0.03) > e3     # more friction, larger margin
    R = frame([0.3, 0.2, 0.9], [1.0, 0.0, 0.0])            # the measure does not depend on the frame
    assert grasp_epsilon(P3 @ R.T, N3 @ R.T, c, 0.5, 8, 0.03) == pytest.approx(e3, rel=0.05)


def test_holding_effort():
    t = np.radians([0, 120, 240])
    P = 0.03 * np.stack([np.cos(t), np.sin(t), np.zeros(3)], 1)
    g = np.array([0, 0, -1.0, 0, 0, 0])
    eff = holding_effort(P, -P / 0.03, np.zeros(3), g, 0.5, 8, 0.03)
    # three pinching contacts hold weight w with friction: total normal force >= w / mu
    assert 1 / 0.5 - 1e-6 <= eff < 3.0
    assert holding_effort(P[:1], -P[:1] / 0.03, np.zeros(3), g, 0.5, 8, 0.03) == float("inf")


# ---------------------------------------------------------------------------------------------------- object
def test_object_geometry_sdf_and_ray_entry():
    obj = ObjectGeom.from_parts([box_parts([0.02, 0.02, 0.02], [0, 0, 0.02])])
    d, n = obj.sdf(np.array([[0, 0, 0.02], [0, 0, 0.05], [0.03, 0, 0.02]]))
    assert d[0] == pytest.approx(-0.02, abs=1e-6) and d[1] == pytest.approx(0.01, abs=1e-6)
    assert np.allclose(n[1], [0, 0, 1]) and np.allclose(n[2], [1, 0, 0])
    s = obj.entry(np.array([[0, 0, 0.10], [0.05, 0, 0.10], [0.01, 0.01, 0.10]]), np.array([0, 0, -1.0]))
    assert s[0] == pytest.approx(0.06) and np.isinf(s[1]) and s[2] == pytest.approx(0.06)
    # a concave L of two parts: the ray enters the part it meets first
    L = ObjectGeom.from_parts([box_parts([0.03, 0.01, 0.01], [0, 0, 0.01]), box_parts([0.01, 0.01, 0.03], [0.02, 0, 0.03])])
    s = L.entry(np.array([[-0.02, 0, 0.1], [0.02, 0, 0.1]]), np.array([0, 0, -1.0]))
    assert s[0] == pytest.approx(0.08) and s[1] == pytest.approx(0.04)
    assert ObjectGeom.from_points(np.random.default_rng(0).uniform(-0.02, 0.02, (300, 3)) + [0, 0, 0.05],
                                  table_z=0.0).sdf(np.array([[0, 0, 0.005]]))[0][0] < 0   # extruded to the table


def test_mesh_surface_samples_lie_on_the_mesh():
    P = box_parts([0.02, 0.03, 0.01])
    h = ConvexHull(P)
    pts = mesh_surface(P, h.simplices, 0.004)
    assert len(pts) > 100
    assert np.abs(np.abs(pts) - [0.02, 0.03, 0.01]).min(1).max() < 1e-3     # within a millimetre (voxel means)


# ---------------------------------------------------------------------------------------------------- hand
def test_hand_model_structure(hand):
    assert [f.name for f in hand.fingers] == ["thumb", "index", "middle"]
    assert hand.fingers[hand.thumb].name == "thumb"
    assert np.allclose(hand.wrist_local, [0, 0, -0.02])     # the model's wrist site
    hand.set(np.eye(4), OPEN)
    tips = hand.tips()
    assert tips[0][1] < 0 < tips[1][1]                       # thumb on -y of the TCP, fingers on +y
    J = hand.tip_jacobians()
    assert J.shape == (3, 3, 6) and np.allclose(J[0][:, 2:], 0)    # the thumb tip does not move with the fingers
    pts, fid = hand.points_in_tcp(POWER)
    assert len(pts) == len(fid) and set(np.unique(fid)) == {-1, 0, 1, 2}


# ---------------------------------------------------------------------------------------------------- planning
@pytest.fixture(scope="module")
def box_grasps(hand):
    obj = ObjectGeom.from_parts([box_parts([0.02, 0.02, 0.025], [0.0, 0.0, 0.025], yaw=0.3)], mass=0.1)
    planner = DexGraspPlanner(hand, PlannerConfig(n_dirs=16, n_rolls=4, refine_rounds=1, refine_samples=6))
    return obj, planner, planner.plan(obj, table_z=0.0)


def test_planner_finds_force_closure_grasps_on_a_box(box_grasps):
    obj, planner, gs = box_grasps
    assert gs and gs[0].force_closure and gs[0].epsilon > 0.05
    assert [g.score for g in gs] == sorted(g.score for g in gs)
    g = gs[0]
    assert g.penetration <= planner.cfg.max_penetration
    assert "thumb" in g.contact_fingers and len(set(g.contact_fingers) - {"palm"}) >= 2
    assert g.approach[2] < 0.3                              # never from below the table
    # the squeeze command lies past the contact angles towards the closing target
    target = planner.closing_target(g.style)
    assert np.all(np.abs(target - g.q_squeeze) <= np.abs(target - g.q_contact) + 1e-9)
    planner.hand.set(g.tcp, g.q_contact)
    P, _ = planner.hand.surface()
    assert P[:, 2].min() > -1e-3                            # nothing under the table


def test_planner_prefers_a_given_pose_and_makes_goals(box_grasps, hand):
    obj, planner, gs = box_grasps
    tilt = np.array([0.7, 0.0, -0.7]) / np.sqrt(0.98)
    want = pose(frame(tilt, [0.0, 1.0, 0.0]), obj.centre - 0.05 * tilt)
    pref = planner.plan(obj, table_z=0.0, prefer={"tcp": want, "w_rot": 3.0})
    assert pref and pref[0].approach @ tilt > max(0.8, gs[0].approach @ tilt)
    goals = to_goals(gs[:3], "arm", pre_dist=0.1)
    assert all(g.verb == "grasp" and g.arm == "arm" for g in goals)
    assert np.allclose(goals[0].pre[:3, 3], gs[0].tcp[:3, 3] - 0.1 * gs[0].approach)
    assert tuple(goals[0].hand) == tuple(np.round(gs[0].q_squeeze, 4))


def test_bend_onto_grasp_phases(box_grasps):
    _, _, gs = box_grasps
    g = gs[0]
    T = 40
    tcp = np.stack([pose(None, [0.0, 0.0, 0.3 - 0.005 * t]) for t in range(T)])
    q = np.tile(np.asarray(OPEN), (T, 1))
    a, b = 20, 30
    out_tcp, out_q, info = bend_onto_grasp(tcp, q, g, a, b, fps=10.0, ramp_s=0.6, approach_s=0.3)
    assert np.allclose(out_tcp[a], g.tcp)                   # on the grasp at the grasp moment
    assert np.allclose(out_tcp[:a - 6], tcp[:a - 6]) and np.allclose(out_q[:a - 6], q[:a - 6])
    assert np.allclose(out_q[b - 1], g.q_squeeze)           # holding at the end of the interval
    rel0 = np.linalg.inv(tcp[a]) @ out_tcp[a]
    assert np.allclose(np.linalg.inv(tcp[b - 1]) @ out_tcp[b - 1], rel0)   # the offset rides on the motion
    assert info["first_frame"] == a - 6
    mid = blend_pose(tcp[0], g.tcp, 0.5)
    assert np.allclose(mid[:3, 3], (tcp[0][:3, 3] + g.tcp[:3, 3]) / 2)


# ---------------------------------------------------------------------------------------------------- retarget
def test_retargeting_recovers_known_answers(hand):
    rng = np.random.default_rng(1)
    qs = [hand.open + w * (hand.closed - hand.open) for w in np.linspace(0, 1, 8)]
    Te = pose(frame([0.2, 0.1, 1.0], [1.0, 0.0, 0.0]), [0.1, -0.2, 0.4])
    K = np.stack([keypoints_from_robot(hand, q, Te) for q in qs])
    rt = Retargeter(hand, scale=1.0, w_smooth=1e-4, w_rest=1e-5)
    out = rt.run(K)
    assert out.tip_error_mm.max() < 0.5 and out.pair_error_mm.max() < 0.5
    assert np.abs(out.q - np.asarray(qs)).max() < 0.02     # this hand's joints are identifiable from its tips
    assert np.allclose(out.ee[:, :3, 3], Te[:3, 3], atol=1e-6) and np.allclose(out.ee[0, :3, :3], Te[:3, :3], atol=1e-6)
    assert np.allclose(out.tcp, out.ee @ hand.tcp_in_ee)
    s = out.summary()
    assert s["fingers"] == {"thumb": "thumb", "index": "index", "middle": "middle"} and s["ms_per_frame"] > 0
    # missing frames are filled in, the auto scale is near 1 for a hand the robot's size, joint limits hold
    valid = np.ones(len(K), bool)
    valid[3] = False
    noisy = K + rng.normal(0, 0.002, K.shape)
    out2 = Retargeter(hand).run(noisy, valid=valid)
    assert 0.8 < out2.scale < 1.25 and np.all(np.isfinite(out2.q))
    assert np.all(out2.q >= hand.lower - 1e-9) and np.all(out2.q <= hand.upper + 1e-9)
    assert np.allclose(out2.q[3], (out2.q[2] + out2.q[4]) / 2, atol=1e-6)


def test_retarget_palm_frame_and_mirroring(hand):
    k = keypoints_from_robot(hand, hand.open)
    F = human_frame(k)
    assert np.allclose(F[:3, :3].T @ F[:3, :3], np.eye(3), atol=1e-9) and np.linalg.det(F[:3, :3]) > 0
    rt = Retargeter(hand, scale=1.0)
    same = rt.run(k)
    mir = rt.run(k, side="left")                # a left human hand on this right hand is mirrored
    assert not same.mirrored and mir.mirrored
    assert np.isclose(np.linalg.det(mir.ee[0, :3, :3]), 1.0)


# ---------------------------------------------------------------------------------------------------- toolkit
def test_toolkit_verbs_on_a_dexterous_arm(tmp_path):
    import manip_testkit as tk
    from hylo.core.robot import ArmSpec, RobotSpec
    from hylo.manip import Tools, api_reference
    from hylo.manip.datatypes import Table

    hand_body = HAND_XML.split("<worldbody>")[1].split("</worldbody>")[0].replace('<body name="palm">',
                                                                                 '<body name="palm" pos="0 0 0.08">')
    body = tk.ARM_BODY
    start = body.index('<body name="hand"')
    end = body.index("</body>\n        </body>\n      </body>")
    body = body[:start] + hand_body.strip() + "\n" + body[end + len("</body>\n"):]
    path = tmp_path / "dexarm.xml"
    path.write_text(f'<mujoco model="dexarm">{tk.HEADER}<worldbody>{body}</worldbody></mujoco>')
    arm = ArmSpec(name="arm", model=str(path), joints=tuple(f"j{i}" for i in range(1, 7)), ee_frame="palm",
                  lower=(-2.9, -2.0, -2.9, -2.6, -2.9, -2.2), upper=(2.9, 2.0, 2.9, 2.6, 2.9, 2.2),
                  home=tk.ARM_HOME[:6], hand=SPEC, action_slice=(0, 6), hand_slice=(6, 12))
    spec = RobotSpec(arms=(arm,), table_height=0.0)
    robot = Tools(spec, tk.KinematicExecutor(spec))
    table = Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]))
    corners = box_parts([0.02, 0.02, 0.025], [0.45, 0.1, 0.025])
    pts = mesh_surface(corners, ConvexHull(corners).simplices)
    robot._dex("arm")[1].cfg.n_dirs, robot._dex("arm")[1].cfg.n_rolls = 16, 4
    goals = robot.dex_grasps(pts, table, top=4)
    assert 0 < len(goals) <= 4 and all(g.verb == "grasp" for g in goals)
    assert [g.rank for g in goals] == sorted(g.rank for g in goals)
    rh = keypoints_from_robot(robot._dex("arm")[0], OPEN)
    out = robot.retarget(np.stack([rh, rh]))
    assert out["hand"].shape == (2, 6) and out["tcp"].shape == (2, 4, 4) and out["tip_error_mm"].shape == (2,)
    ref = api_reference(robot)
    assert "robot.dex_grasps(" in ref and "robot.retarget(" in ref and "robot.hand_grasps(" in ref
    parallel = tk.arm_spec(tk.write_arm(tmp_path))
    with pytest.raises(ValueError):
        Tools(parallel, tk.KinematicExecutor(parallel)).dex_grasps(pts, table)
