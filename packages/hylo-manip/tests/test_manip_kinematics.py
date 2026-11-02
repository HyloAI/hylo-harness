import time

import mujoco
import numpy as np

from hylo.manip.geometry import angle
from hylo.manip.kinematics import Kinematics

import manip_testkit as tk


def ik_success(spec, n=150, seed=1):
    arm = Kinematics(spec)["arm"]
    rng = np.random.default_rng(seed)
    ok, worst, times = 0, 0.0, []
    for _ in range(n):
        T = arm.fk(rng.uniform(arm.lower, arm.upper))
        t0 = time.perf_counter()
        r = arm.ik(T, arm.home, rng=rng)
        times.append(time.perf_counter() - t0)
        if r.ok:
            ok += 1
            back = arm.fk(r.q)
            worst = max(worst, np.linalg.norm(back[:3, 3] - T[:3, 3]))
            assert np.all(r.q >= arm.lower - 1e-9) and np.all(r.q <= arm.upper + 1e-9)
    return ok / n, worst, float(np.mean(times))


def test_fk_matches_mujoco_and_base_offset(arm_model, arm_spec):
    arm = Kinematics(arm_spec)["arm"]
    m = mujoco.MjModel.from_xml_path(str(arm_model))
    d = mujoco.MjData(m)
    q = np.array([0.3, -0.4, 0.5, 1.2, -0.3, 0.8, 0.2])
    d.qpos[:7] = q
    mujoco.mj_kinematics(m, d)
    hand = np.eye(4)
    hand[:3, :3] = d.xmat[m.body("hand").id].reshape(3, 3)
    hand[:3, 3] = d.xpos[m.body("hand").id]
    assert np.allclose(arm.fk(q), hand @ np.array(tk.TCP_IN_HAND, float), atol=1e-9)

    base = [[0, -1, 0, 0.2], [1, 0, 0, -0.1], [0, 0, 1, 0.05], [0, 0, 0, 1]]
    moved = arm_spec.model_copy(update={"arms": (arm_spec.arms[0].model_copy(update={"base_in_world": base}),)})
    arm2 = Kinematics(moved)["arm"]
    assert np.allclose(arm2.fk(q), np.array(base) @ arm.fk(q), atol=1e-9)


def test_jacobian_matches_finite_differences(arm_spec):
    arm = Kinematics(arm_spec)["arm"]
    q = np.array([0.1, 0.5, -0.2, 1.0, 0.4, 0.7, -0.3])
    T = arm.fk(q)
    J = arm.jacobian(T)
    eps = 1e-6
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        T2 = arm.fk(dq)
        assert np.allclose((T2[:3, 3] - T[:3, 3]) / eps, J[:3, i], atol=1e-4)
        w = mujoco_rotvec(T2[:3, :3] @ T[:3, :3].T) / eps
        assert np.allclose(w, J[3:, i], atol=1e-4)


def mujoco_rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def test_ik_round_trip_test_arm(arm_spec):
    rate, worst, _ = ik_success(arm_spec)
    assert rate >= 0.95 and worst <= 1e-3


def test_ik_free_axis(arm_spec):
    arm = Kinematics(arm_spec)["arm"]
    T = arm.fk(np.array([0.2, 0.6, 0.0, 1.4, 0.0, 1.0, 0.0]))
    target = T.copy()
    target[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])   # rotated 90 deg about TCP x
    target[:3, 3] += [0.02, 0.03, -0.02]
    r = arm.ik(target, arm.home, free="x")
    assert r.ok
    got = arm.fk(r.q)
    assert np.linalg.norm(got[:3, 3] - target[:3, 3]) < 1e-3
    assert np.dot(got[:3, 0], target[:3, 0]) > np.cos(0.01)


def test_ik_unreachable_reports_residual(arm_spec):
    arm = Kinematics(arm_spec)["arm"]
    T = np.eye(4)
    T[:3, 3] = [2.0, 0, 0.5]
    r = arm.ik(T, arm.home, n_random=2)
    assert not r.ok and r.pos_err > 0.5


def test_panda_fk_and_ik(panda_spec):
    arm = Kinematics(panda_spec)["arm"]
    T = arm.fk(arm.home)
    assert np.allclose(T[:3, 0], [0, 0, -1], atol=1e-3)          # home: approach straight down
    assert 0.4 < T[0, 3] < 0.7 and 0.3 < T[2, 3] < 0.7
    rate, worst, mean_s = ik_success(panda_spec, n=150)
    print(f"panda IK success {rate:.3f}, worst position error {worst * 1e3:.2f} mm, mean {mean_s * 1e3:.1f} ms")
    assert rate >= 0.95 and worst <= 1e-3


def test_panda_ik_tabletop_poses(panda_spec):
    """Top-down poses over the usual workspace: every one reachable."""
    arm = Kinematics(panda_spec)["arm"]
    rng = np.random.default_rng(3)
    ok = 0
    for _ in range(60):
        T = np.eye(4)
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        T[:3, :3] = np.stack([[0, 0, -1], [np.cos(yaw), np.sin(yaw), 0],
                              np.cross([0, 0, -1], [np.cos(yaw), np.sin(yaw), 0])], axis=1)
        T[:3, 3] = [rng.uniform(0.3, 0.65), rng.uniform(-0.3, 0.3), rng.uniform(0.02, 0.3)]
        r = arm.ik(T, arm.home, rng=rng)
        ok += r.ok and angle(arm.fk(r.q)[:3, :3].T @ T[:3, :3]) < 0.01
    assert ok >= 59
