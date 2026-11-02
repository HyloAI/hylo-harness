import time

import numpy as np

from hylo.manip import Goal, Table, Tools
from hylo.manip.collision import Obstacles
from hylo.manip.planning import check_limits, time_parameterize

import manip_testkit as tk


def down(x, y, z, yaw=0.0):
    """TCP pose approaching straight down, closing axis at `yaw`."""
    T = np.eye(4)
    a, c = np.array([0, 0, -1.0]), np.array([np.cos(yaw), np.sin(yaw), 0.0])
    T[:3, :3] = np.stack([a, c, np.cross(a, c)], axis=1)
    T[:3, 3] = [x, y, z]
    return T


def rows_clear(tools, plan, points, table, margin=0.005):
    """Every planned row (joints and hand) is free of the points, the table and self-collision."""
    _, cols, _ = tools._models()
    c = cols["arm"]
    obs = Obstacles(table=table, clearance=0.0, points=points, margin=margin)
    spec = tools.spec.arms[0]
    for row in plan.actions:
        c.arm.set_hand(row[spec.hand_slice[0]:spec.hand_slice[1]])
        hit = c.check(row[spec.action_slice[0]:spec.action_slice[1]], obs)
        if hit:
            return hit
    return None


def test_time_parameterize_respects_limits():
    path = [np.zeros(3), np.array([1.0, 0.5, -0.2]), np.array([1.2, 1.5, 0.3]), np.array([1.25, 1.55, 0.31])]
    vmax, amax, dt = np.array([1.0, 0.5, 2.0]), np.array([2.0, 1.0, 3.0]), 0.05
    traj = time_parameterize(path, vmax, amax, dt)
    assert np.allclose(traj[-1], path[-1])
    rv, ra = check_limits(traj, path[0], vmax, amax, dt)
    assert rv <= 1.0 + 1e-6 and ra <= 1.05
    # a dense, nearly straight path does not stop at every sample
    line = [np.array([s, 2 * s]) for s in np.linspace(0, 1, 101)]
    t_line = time_parameterize(line, np.ones(2), np.full(2, 4.0), dt)
    t_direct = time_parameterize([line[0], line[-1]], np.ones(2), np.full(2, 4.0), dt)
    assert len(t_line) <= len(t_direct) + 2


def test_plan_timing_and_hand_rows(arm_spec):
    ex = tk.KinematicExecutor(arm_spec)
    tools = Tools(arm_spec, ex, max_vel=0.8, max_acc=1.5)
    p = tools.plan(Goal("reach", "arm", tcp=down(0.45, 0.15, 0.15), hand_before=(1.0,), hand=(0.0,)), Table(0.0, np.r_[0, 0, 1.0]))
    assert p is not None
    q0 = np.array(arm_spec.arms[0].home)
    rv, ra = check_limits(p.actions[:, :7], q0, np.full(7, 0.8), np.full(7, 1.5), 1 / 20)
    assert rv <= 1.0 + 1e-6 and ra <= 1.05
    hold = p.info["hand_steps"]
    assert hold == 20 and np.all(p.actions[-hold:, 7] == 0.0) and np.all(p.actions[:-hold, 7] == 1.0)
    assert np.allclose(p.actions[-hold:, :7], p.actions[-1, :7])
    r = tools.run(p)
    assert r["steps"] == len(p.actions) and r["tcp_error"]["pos"] < 1e-3 and r["tcp_error"]["rot"] < 0.01


def test_plan_is_deterministic_and_reports_unreachable(arm_spec):
    goal = Goal("reach", "arm", tcp=down(0.5, -0.2, 0.1))
    a = Tools(arm_spec, tk.KinematicExecutor(arm_spec), seed=4).plan(goal)
    b = Tools(arm_spec, tk.KinematicExecutor(arm_spec), seed=4).plan(goal)
    assert np.array_equal(a.actions, b.actions)
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    assert tools.plan([Goal("reach", "arm", tcp=down(1.5, 0, 0.2)), Goal("reach", "arm", tcp=down(0, 0, -0.5))]) is None
    assert "goal 0" in tools.why() and "goal 1" in tools.why() and "unreachable" in tools.why()


def test_plan_takes_goals_by_rank(arm_spec):
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    far = Goal("reach", "arm", tcp=down(1.5, 0, 0.2), rank=0)
    ok1 = Goal("reach", "arm", tcp=down(0.4, 0.1, 0.2), rank=2)
    ok2 = Goal("reach", "arm", tcp=down(0.4, -0.1, 0.2), rank=1)
    p = tools.plan([far, ok1, ok2])
    assert p.goal is ok2


def wall(center, half, spacing=0.01):
    return tk.box_points(center, half, spacing=spacing)


def test_planner_avoids_box_and_table(arm_spec):
    ex = tk.KinematicExecutor(arm_spec)
    tools = Tools(arm_spec, ex)
    table = Table(0.0, np.r_[0, 0, 1.0])
    obstacle = wall((0.42, -0.12, 0.17), (0.12, 0.02, 0.17))
    goal = Goal("reach", "arm", tcp=down(0.42, -0.32, 0.08))
    p0 = tools.plan(goal, table)                       # without the obstacle: straight joint motion
    assert p0.info["method"] == "direct" and rows_clear(tools, p0, obstacle, 0.0) == "points"
    p = tools.plan(goal, table, scene=obstacle)
    assert p is not None and p.info["method"] == "rrt"
    assert rows_clear(tools, p, obstacle, 0.0) is None
    tools.run(p)
    assert np.linalg.norm(tools.status()["tcp"]["arm"][:3, 3] - goal.tcp[:3, 3]) < 1e-3


def test_pre_pose_then_straight_line(arm_spec):
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    tcp = down(0.5, 0.0, 0.03)
    pre = down(0.5, 0.0, 0.13)
    p = tools.plan(Goal("grasp", "arm", tcp=tcp, pre=pre, hand_before=(1.0,), hand=(0.0,)), Table(0.0, np.r_[0, 0, 1.0]))
    assert p is not None and p.info["method"].endswith("+line")
    arm = tools._arm("arm")
    xy = np.array([arm.fk(r[:7])[:2, 3] for r in p.actions])
    zs = np.array([arm.fk(r[:7])[2, 3] for r in p.actions])
    below = zs < 0.125                                  # the final approach: straight down over the target
    assert below.any() and np.allclose(xy[below], [0.5, 0.0], atol=2e-3)


def test_lift_moves_in_a_straight_line(arm_spec):
    ex = tk.KinematicExecutor(arm_spec)
    tools = Tools(arm_spec, ex)
    tools.run(tools.plan(Goal("reach", "arm", tcp=down(0.45, 0.1, 0.05))))
    start = tools.status()["tcp"]["arm"]
    up = start.copy()
    up[2, 3] += 0.15
    p = tools.plan(Goal("lift", "arm", tcp=up), Table(0.0, np.r_[0, 0, 1.0]))
    assert p.info["method"] == "line"
    arm = tools._arm("arm")
    assert all(np.linalg.norm(arm.fk(r[:7])[:2, 3] - start[:2, 3]) < 2e-3 for r in p.actions)


def test_panda_plans_around_obstacle(panda_spec):
    ex = tk.KinematicExecutor(panda_spec)
    tools = Tools(panda_spec, ex)
    table = Table(0.0, np.r_[0, 0, 1.0])
    obstacle = wall((0.5, -0.17, 0.3), (0.3, 0.02, 0.3))     # 60 cm high, 60 cm long
    goal = Goal("grasp", "arm", tcp=down(0.5, -0.35, 0.03), pre=down(0.5, -0.35, 0.13), hand_before=(1.0,),
                hand=(0.0,))
    p0 = tools.plan(goal, table)
    assert p0.info["method"] == "direct+line" and rows_clear(tools, p0, obstacle, 0.0) == "points"
    t0 = time.perf_counter()
    p = tools.plan(goal, table, scene=obstacle)
    dt = time.perf_counter() - t0
    print(f"panda plan around obstacle: {dt:.2f} s, method {p.info['method']}, {p.info['duration_s']:.1f} s motion")
    assert p is not None and p.info["method"] == "rrt+line" and rows_clear(tools, p, obstacle, 0.0) is None
    rv, ra = check_limits(p.actions[:, :7], np.array(panda_spec.arms[0].home), np.ones(7), np.full(7, 2.0), 0.05)
    assert rv <= 1.0 + 1e-6 and ra <= 1.05
    r = tools.run(p)
    assert r["tcp_error"]["pos"] < 1e-3


def test_panda_planning_time_over_random_tabletop_goals(panda_spec):
    tools = Tools(panda_spec, tk.KinematicExecutor(panda_spec))
    table = Table(0.0, np.r_[0, 0, 1.0])
    rng = np.random.default_rng(0)
    times, found = [], 0
    for _ in range(12):
        x, y, yaw = rng.uniform(0.35, 0.65), rng.uniform(-0.3, 0.3), rng.uniform(-1.5, 1.5)
        goal = Goal("grasp", "arm", tcp=down(x, y, 0.03, yaw), pre=down(x, y, 0.13, yaw), hand_before=(1.0,),
                    hand=(0.0,))
        t0 = time.perf_counter()
        p = tools.plan(goal, table)
        times.append(time.perf_counter() - t0)
        found += p is not None
    print(f"panda tabletop grasp plans: {found}/12 found, mean {np.mean(times):.3f} s, max {np.max(times):.3f} s")
    assert found >= 11


def test_joint_goals_in_collision_are_refused(arm_spec):
    tools = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    table = Table(0.0, np.r_[0, 0, 1.0])
    folded = np.array([0, 0, 0, 2.6, 0, 2.2, 0.0])                       # hand folded back onto the upper arm
    assert tools.plan(Goal("home", "arm", joints=folded), table) is None and "(self)" in tools.why()
    into_table = np.array([0, 1.9, 0, 0.3, 0, 0, 0.0])
    assert tools.plan(Goal("home", "arm", joints=into_table), table) is None and "(table)" in tools.why()
