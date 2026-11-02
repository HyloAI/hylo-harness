"""Several arms: the arm= defaults, plans for one arm with the other as an obstacle, running arms together (merged,
checked step by step for arms touching), re-timing arms to move in step, and the single-arm API unchanged."""

from pathlib import Path

import numpy as np
import pytest

from hylo.core.robot import ArmSpec, HandSpec, RobotSpec
from hylo.manip import Goal, Tools, api_reference

import manip_testkit as tk

BASES = {"left": (0.0, 0.35, 0.0), "right": (0.0, -0.35, 0.0)}


def _prefixed(prefix: str, pos) -> str:
    body = tk.ARM_BODY
    for name in ("base", "link1", "link2", "link3", "link4", "hand", "lf", "rf"):
        body = body.replace(f'name="{name}"', f'name="{prefix}{name}"')
    for j in ("j1", "j2", "j3", "j4", "j5", "j6", "j7", "f1", "f2"):
        body = body.replace(f'name="{j}"', f'name="{prefix}{j}"')
    return body.replace(f'<body name="{prefix}base">', f'<body name="{prefix}base" pos="{pos[0]} {pos[1]} {pos[2]}">', 1)


def _arm(name: str, model: Path, prefix: str, slices, base=None) -> ArmSpec:
    hand = HandSpec(kind="parallel", tcp_in_ee=tk.TCP_IN_HAND, open=(1.0,), closed=(0.0,), max_width=0.08,
                    joints=(f"{prefix}f1", f"{prefix}f2"))
    T = np.eye(4)
    if base is not None:
        T[:3, 3] = base
    return ArmSpec(name=name, model=str(model), joints=tuple(f"{prefix}j{i}" for i in range(1, 8)),
                   ee_frame=f"{prefix}hand", base_in_world=T.tolist(), lower=(-2.9, -2.0, -2.9, -2.6, -2.9, -2.2, -2.9),
                   upper=(2.9, 2.0, 2.9, 2.6, 2.9, 2.2, 2.9), home=tk.ARM_HOME, hand=hand,
                   action_slice=slices[0], hand_slice=slices[1])


@pytest.fixture(scope="module")
def shared_spec(tmp_path_factory):
    """Both arms in one model (as in the ALOHA and G1 models)."""
    d = tmp_path_factory.mktemp("two")
    path = d / "two.xml"
    path.write_text(f'<mujoco model="two">{tk.HEADER}<worldbody>{_prefixed("l_", BASES["left"])}'
                    f'{_prefixed("r_", BASES["right"])}</worldbody></mujoco>')
    return RobotSpec(arms=(_arm("left", path, "l_", ((0, 7), (7, 8))), _arm("right", path, "r_", ((8, 15), (15, 16)))),
                     table_height=0.0)


@pytest.fixture(scope="module")
def separate_spec(tmp_path_factory):
    """Each arm its own model file, placed by base_in_world."""
    d = tmp_path_factory.mktemp("sep")
    paths = {}
    for side in ("left", "right"):
        paths[side] = d / f"{side}.xml"
        paths[side].write_text(f'<mujoco model="{side}">{tk.HEADER}<worldbody>{tk.ARM_BODY}</worldbody></mujoco>')
    return RobotSpec(arms=(_arm("left", paths["left"], "", ((0, 7), (7, 8)), BASES["left"]),
                           _arm("right", paths["right"], "", ((8, 15), (15, 16)), BASES["right"])), table_height=0.0)


def reach(robot, arm, xyz):
    """A top-down TCP goal at xyz for one arm."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])     # +x (approach) down, +y closing along y
    T[:3, 3] = xyz
    return Goal("reach", arm, tcp=T)


def test_arm_defaults_and_errors(shared_spec, arm_spec):
    single = Tools(arm_spec, tk.KinematicExecutor(arm_spec))
    assert single.holding() is False                                     # the only arm
    t = single.status()["tcp"]["arm"].copy()
    t[2, 3] += 0.05
    p = single.plan(Goal("lift", tcp=t))                                 # Goal.arm None: the only arm
    assert p is not None and p.goal.arm == "arm"
    two = Tools(shared_spec, tk.KinematicExecutor(shared_spec))
    with pytest.raises(ValueError, match="pass arm="):
        two.holding()
    with pytest.raises(KeyError):
        two.holding(arm="middle")
    assert two.holding(arm="left") is False


def test_old_and_new_calling_forms_agree(tmp_path, arm_spec):
    objects = {"red_block": ((0.45, 0.08, 0.025), (0.025, 0.02, 0.025), (0.9, 0.1, 0.1, 1), 0.3)}
    ex = tk.StubExecutor(tmp_path, objects, arm_spec, hw=(96, 128))
    robot = Tools(arm_spec, ex)
    view = robot.look()
    table = robot.table(view)
    det = robot.objects(view, robot.find(view, "red block"), table)[0]
    old = robot.grasps("arm", det.points, table, scene=view.points)
    new = robot.grasps(det.points, table, scene=view.points)
    named = robot.grasps(det.points, table, view.points, arm="arm")
    assert old and len(old) == len(new) == len(named)
    assert all(np.allclose(a.tcp, b.tcp) and np.allclose(a.tcp, c.tcp) for a, b, c in zip(old, new, named))
    g_old, g_new = robot.grasp_at("arm", old[0].tcp), robot.grasp_at(old[0].tcp)
    assert np.allclose(g_old.pre, g_new.pre) and g_new.arm == "arm"
    with pytest.raises(TypeError):
        robot.grasp_at("arm", old[0].tcp, arm="arm")


@pytest.mark.parametrize("which", ["shared", "separate"])
def test_together_runs_both_arms_at_once(which, shared_spec, separate_spec):
    spec = shared_spec if which == "shared" else separate_spec
    ex = tk.KinematicExecutor(spec)
    robot = Tools(spec, ex)
    pl = robot.plan(reach(robot, "left", (0.45, 0.30, 0.30)))
    pr = robot.plan(reach(robot, "right", (0.40, -0.25, 0.20)))
    assert pl is not None and pr is not None
    merged = robot.together({"left": pl, "right": pr})
    assert merged is not None, robot.why()
    assert len(merged.actions) == max(len(pl.actions), len(pr.actions))
    left, right = spec.arm("left"), spec.arm("right")
    n = min(len(pl.actions), len(pr.actions))
    np.testing.assert_allclose(merged.actions[:n, slice(*left.action_slice)], pl.actions[:n, slice(*left.action_slice)])
    np.testing.assert_allclose(merged.actions[:n, slice(*right.action_slice)], pr.actions[:n, slice(*right.action_slice)])
    out = robot.run({"left": pl, "right": pr})
    assert out["steps"] == len(merged.actions) and set(out["tcp_error"]) == {"left", "right"}
    assert out["tcp_error"]["left"]["pos"] < 2e-3 and out["tcp_error"]["right"]["pos"] < 2e-3


@pytest.mark.parametrize("which", ["shared", "separate"])
def test_arms_that_would_touch_are_refused(which, shared_spec, separate_spec):
    spec = shared_spec if which == "shared" else separate_spec
    ex = tk.KinematicExecutor(spec)
    robot = Tools(spec, ex)
    pl = robot.plan(reach(robot, "left", (0.45, -0.05, 0.25)))           # each alone is fine; crossing at once
    pr = robot.plan(reach(robot, "right", (0.45, 0.05, 0.25)))           # they meet in the middle
    assert pl is not None and pr is not None
    assert robot.together([pl, pr]) is None and "touches" in robot.why()
    out = robot.run({"left": pl, "right": pr})
    assert out["steps"] == 0 and "touches" in out["error"] and ex.step == 0
    assert robot.together([pl, pl]) is None and "one plan per arm" in robot.why()


def test_one_arm_plans_around_the_other(shared_spec):
    ex = tk.KinematicExecutor(shared_spec)
    robot = Tools(shared_spec, ex)
    robot.run(robot.plan(reach(robot, "right", (0.45, 0.0, 0.25))))       # right arm parked in the middle
    assert robot.plan(reach(robot, "left", (0.45, 0.0, 0.25))) is None  # where the right hand is
    assert "collision (self)" in robot.why()
    p = robot.plan(reach(robot, "left", (0.45, 0.10, 0.35)))            # just above it
    assert p is not None, robot.why()
    _, planner = robot._models()[0], robot._models()[2]
    left, right = shared_spec.arm("left"), shared_spec.arm("right")
    for row in p.actions:
        q = {"left": row[slice(*left.action_slice)], "right": row[slice(*right.action_slice)]}
        assert planner.arms_clear(q, {"left": [1.0], "right": [1.0]}) is None


def test_sync_moves_tcps_in_step(shared_spec):
    ex = tk.KinematicExecutor(shared_spec)
    robot = Tools(shared_spec, ex)
    robot.run({"left": robot.plan(reach(robot, "left", (0.45, 0.30, 0.20))),
               "right": robot.plan(reach(robot, "right", (0.45, -0.30, 0.20)))})
    st = robot.status()
    ups = {}
    for arm, dz in (("left", 0.15), ("right", 0.15)):
        t = st["tcp"][arm].copy()
        t[2, 3] += dz
        ups[arm] = robot.plan(Goal("lift", arm, tcp=t))
    plain = robot.together(ups)
    synced = robot.together(ups, sync=True)
    assert plain is not None and synced is not None
    kin = robot._models()[0]
    z0 = {a: st["tcp"][a][2, 3] for a in ups}
    frac = {a: [] for a in ups}
    for row in synced.actions:
        for a in ups:
            s = shared_spec.arm(a)
            frac[a].append((kin[a].fk(row[slice(*s.action_slice)])[2, 3] - z0[a]) / 0.15)
    gap = np.max(np.abs(np.array(frac["left"]) - np.array(frac["right"])))
    assert gap < 0.03 and abs(frac["left"][-1] - 1.0) < 0.01             # in step, and both arrive
    res = robot.run(ups, sync=True)
    assert res["tcp_error"]["left"]["pos"] < 3e-3 and res["tcp_error"]["right"]["pos"] < 3e-3


def test_api_reference_for_two_arms(shared_spec):
    ref = api_reference(Tools(shared_spec, tk.KinematicExecutor(shared_spec)))
    assert "arm 'left'" in ref and "arm 'right'" in ref and "Several arms" in ref
    assert "robot.together(" in ref and 'robot.run({"left": pl, "right": pr})' in ref
    assert "robot.grasps(points" in ref and "arm: str | None = None" in ref
    assert len(ref) < 14000
