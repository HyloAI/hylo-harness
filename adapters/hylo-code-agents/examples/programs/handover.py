"""handover: pass the named bar from the arm that reaches it to the arm that reaches the named zone, which puts it
there (perception only: cameras and segmentation).

Each arm reaches only about 10 cm past the middle of the table. The arm on the bar's side grasps it near one end and
holds it up in the middle of the table, turned along y; the other arm grasps the free end from above; the first lets
go and leaves; the second puts the bar in the zone. Both grippers close along x at the handover, each with its TCP's
+z side (where the wrist camera mount sticks out) facing away from the other gripper."""
import math

import numpy as np

HAND_AT = 0.06            # each gripper this far from the bar's centre at the handover
HANDOVER_Z = 0.16         # bar centre height at the handover


def mentioned(text, labels):
    text = " " + text.lower().replace(",", " ").replace(".", " ").replace(";", " ") + " "
    return [lab for lab in sorted(labels, key=len, reverse=True) if " " + lab + " " in text]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def topdown(psi, pos):
    y = np.array([math.cos(psi), math.sin(psi), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def closing_yaw(T):
    return math.atan2(T[1, 1], T[0, 1])


def detect(robot, view, table, label, min_points=15):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= min_points]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def run_goal(robot, goal, table, scene=None):
    p = robot.plan(goal, table, scene)
    if p is None:
        print(goal.arm, goal.verb, "no plan:", robot.why())
        return None
    return robot.run(p)


def move_tcp(robot, arm, verb, dz=0.0, hand=None, table=None):
    t = robot.status()["tcp"][arm].copy()
    t[2, 3] += dz
    return run_goal(robot, robot.Goal(verb, arm, tcp=t, hand=hand), table)


def first_grasp(robot, arm, det, box, table, view):
    """Goals for grasping the bar near one end, one per closing direction, each with the end that leaves the free
    end on the far side of the handover (ranked by how far the wrist turns on the way). Returns [(goal, psi_a)]."""
    u = np.array([math.cos(box.yaw), math.sin(box.yaw), 0.0])          # the bar's long axis
    top = box.bottom + box.size[2]
    out = []
    for psi in (box.yaw + math.pi / 2, box.yaw - math.pi / 2):
        psi_a = 0.0 if abs(wrap(psi)) < math.pi / 2 else math.pi       # closing yaw at the handover
        dpsi = wrap(psi_a - psi)
        a_side = -math.cos(psi_a)                                      # side (y) of the centre this gripper holds
        sgn = 1.0 if (rotz(dpsi) @ u)[1] * a_side > 0 else -1.0
        pos = box.center + sgn * HAND_AT * u
        pos[2] = top - 0.012
        g = robot.grasp_at(topdown(psi, pos), arm=arm)
        g.rank = abs(dpsi)
        out.append((g, psi_a))
    return out


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    names = mentioned(instruction, labels)
    bar = next((n for n in names if n.endswith("bar")), None)
    zone = next((n for n in names if n.endswith("zone")), None)
    det = detect(robot, view, table, bar)
    zdet = detect(robot, view, table, zone)
    if det is None or zdet is None:
        print("cannot see", bar if det is None else zone)
        return
    box = robot.box(det.points, table)
    target = robot.box(zdet.points).center[:2]
    src = "left" if box.center[0] < 0 else "right"
    dst = "right" if src == "left" else "left"
    print(bar, "->", zone, "| from", src, "to", dst)

    # 1. the arm on the bar's side grasps it near one end and lifts it
    held = False
    for attempt in range(2):
        cands = first_grasp(robot, src, det, box, table, view)
        plan = robot.plan([c[0] for c in cands], table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return
        psi_a = next(p for g, p in cands if g is plan.goal)
        out = robot.run(plan)
        if out.get("ended"):
            return
        held = robot.holding(arm=src)
        move_tcp(robot, src, "lift", 0.10, table=table)
        if held and robot.holding(arm=src):
            break
        move_tcp(robot, src, "retreat", 0.0, hand=(1.0,), table=table)
        view = robot.look()
        det = detect(robot, view, table, bar)
        if det is None:
            return
        box = robot.box(det.points, table)
    else:
        print("could not pick up the bar")
        return

    # 2. carry it to the middle of the table, turned along y (closing yaw psi_a)
    T = robot.status()["tcp"][src]
    c_now = (T @ np.linalg.inv(plan.goal.tcp) @ box.pose)[:3, 3]
    turn = rotz(wrap(psi_a - closing_yaw(T)))
    hy = float(np.clip(box.center[1], -0.04, 0.14))
    goal_tcp = np.eye(4)
    goal_tcp[:3, :3] = turn @ T[:3, :3]
    goal_tcp[:3, 3] = np.array([0.0, hy, HANDOVER_Z]) + turn @ (T[:3, 3] - c_now)
    out = run_goal(robot, robot.Goal("lift", src, tcp=goal_tcp), table)
    if out is None or out.get("ended"):
        return

    # 3. the other arm grasps the free end from above
    for attempt in range(2):
        view = robot.look()
        masks = robot.find(view, bar)
        dets = robot.objects(view, masks, table)
        dets = [d for d in dets if len(d.points) >= 10]
        if not dets:
            print("cannot see the bar in the air")
            return
        pts = max(dets, key=lambda d: len(d.points)).points
        abox = robot.box(pts)                                         # in the air: no table under it
        u = np.array([math.cos(abox.yaw), math.sin(abox.yaw), 0.0])
        tsrc = robot.status()["tcp"][src][:3, 3]
        a_dir = 1.0 if (tsrc - abox.center) @ u > 0 else -1.0
        pb = abox.center - a_dir * HAND_AT * u
        pb[2] = abox.bottom + abox.size[2] - 0.012
        g = robot.grasp_at(topdown(psi_a + math.pi, pb), arm=dst)
        plan_b = robot.plan(g, table, view.points)
        if plan_b is None:
            print("no grasp plan for the free end:", robot.why())
            return
        out = robot.run(plan_b)
        if out.get("ended"):
            return
        if robot.holding(arm=dst):
            break
        print("missed the free end; retrying")
        move_tcp(robot, dst, "retreat", 0.06, hand=(1.0,), table=table)
    else:
        return

    # 4. the first arm lets go and leaves; the second puts the bar in the zone
    move_tcp(robot, src, "retreat", 0.0, hand=(1.0,), table=table)
    move_tcp(robot, src, "retreat", 0.08, table=table)
    view = robot.look()
    obj_in_tcp = np.linalg.inv(plan_b.goal.tcp) @ abox.pose
    place = robot.place_at(obj_in_tcp, abox, target, table=table, hover=0.008, support=table.height, arm=dst)
    place.free = "x"
    home = robot.Goal("home", src, joints=np.array(robot.spec.arm(src).home))
    plans = {}
    for goal in (place, home):
        p = robot.plan(goal, table, view.points if goal is place else None)
        if p is None:
            print(goal.arm, goal.verb, "no plan:", robot.why())
        else:
            plans[goal.arm] = p
    if dst not in plans:
        return
    out = robot.run(plans)
    if out.get("error"):
        print("together refused:", out["error"])
        out = robot.run(plans[dst])
    if out.get("ended"):
        return
    move_tcp(robot, dst, "retreat", 0.08, table=table)
