"""g1_pick_place: the humanoid puts the named block on the named plate with one three-finger hand (perception only:
cameras and segmentation). The hand on the block's side (y > 0: the left) grasps it like a claw from above."""
import math

import numpy as np


def mentioned(text, labels):
    """Segment labels the text names, in the order they appear (longest label wins where labels overlap)."""
    text = " " + text.lower().replace(",", " ").replace(".", " ").replace(":", " ") + " "
    found, taken = [], []
    for lab in sorted(labels, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(" " + lab + " ", start)
            if i < 0:
                break
            span = (i, i + len(lab) + 1)
            if not any(a < span[1] and span[0] < b for a, b in taken):
                taken.append(span)
                found.append((i, lab))
            start = i + 1
    return [pair[1] for pair in sorted(found)]


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 15]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def move(robot, arm, dz, table, hand=None, verb="lift"):
    t = robot.status()["tcp"][arm].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal(verb, arm, tcp=t, hand=hand), table)
    return robot.run(p) if p is not None else None


def pick(robot, arm, label, table, tries=2):
    """Claw grasp from above and lift; returns (grasp tcp, box) or None."""
    for attempt in range(tries):
        view = robot.look()
        det = detect(robot, view, table, label)
        if det is None:
            print("cannot see", label)
            return None
        box = robot.box(det.points, table)
        plan = robot.plan(robot.hand_grasps(det.points, table, kind="power", arm=arm), table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        move(robot, arm, 0.10, table)
        if robot.holding(arm=arm):
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed")
        move(robot, arm, 0.0, table, hand=robot.spec.arm(arm).hand.open, verb="retreat")
    return None


def place(robot, arm, grasp, box, xy, table):
    view = robot.look()
    goal = robot.place_at(np.linalg.inv(grasp) @ box.pose, box, xy, table=table, hover=0.008, scene=view.points, arm=arm)
    goal.free = "x"                                  # any yaw on the plate: easier for the arm to reach
    plan = robot.plan(goal, table, view.points)
    if plan is None:
        print("no place plan:", robot.why())
        return None
    out = robot.run(plan)
    if not out.get("ended"):
        move(robot, arm, 0.10, table)
    return out


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    names = mentioned(instruction, labels)
    block = next((n for n in names if n.endswith("block")), None)
    plate = next((n for n in names if n.endswith("plate")), None)
    print("block:", block, "| plate:", plate)
    b, p = detect(robot, view, table, block), detect(robot, view, table, plate)
    if b is None or p is None:
        print("cannot see", block if b is None else plate)
        return
    arm = "left" if float(np.mean(b.points[:, 1])) > 0 else "right"
    target = robot.box(p.points, table).center[:2]
    got = pick(robot, arm, block, table)
    if got is not None:
        place(robot, arm, got[0], got[1], target, table)
