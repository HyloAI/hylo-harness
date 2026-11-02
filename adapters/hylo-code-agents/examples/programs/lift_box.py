"""lift_box: lift the named box with both arms by its two end handles and hold it level (perception only).

The box is wider than a gripper opens; a thin handle stands up at each end. The program finds the handles in the
box's points (the part standing above the rest), grasps one with each arm from above at the same time, and lifts
both TCPs along straight lines re-timed together, so the box stays level."""
import math

import numpy as np

ARMS = ("left", "right")


def mentioned(text, labels):
    text = " " + text.lower().replace(",", " ").replace(".", " ") + " "
    return [lab for lab in sorted(labels, key=len, reverse=True) if " " + lab + " " in text]


def height_cm(instruction):
    words = instruction.lower().replace(",", " ").split()
    for i, w in enumerate(words[:-1]):
        if words[i + 1] == "cm":
            try:
                return float(w)
            except ValueError:
                pass
    return 10.0


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 30]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def handles(points, box):
    """The two handles: the points within 3 cm of the top, split by side along the box's long axis. Returns
    [(centre xy, top z)] sorted by x (the left arm's first)."""
    u = np.array([math.cos(box.yaw), math.sin(box.yaw)])
    top = float(np.percentile(points[:, 2], 99))
    high = points[points[:, 2] > top - 0.03]
    along = (high[:, :2] - box.center[:2]) @ u
    out = []
    for sel in (along < 0, along > 0):
        pts = high[sel]
        if len(pts) < 5:
            return None
        out.append((pts[:, :2].mean(axis=0), float(np.percentile(pts[:, 2], 98))))
    return sorted(out, key=lambda h: h[0][0])


def topdown(psi, pos):
    """TCP pose pointing down (+x down) with the closing axis (+y) at yaw psi."""
    y = np.array([math.cos(psi), math.sin(psi), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    names = mentioned(instruction, labels) or [lab for lab in labels if lab.endswith("box")]
    dz = height_cm(instruction) / 100.0
    det = detect(robot, view, table, names[0])
    if det is None:
        print("cannot see", names[0])
        return
    box = robot.box(det.points, table)
    found = handles(det.points, box)
    if found is None:
        print("no handles found")
        return
    plans = {}
    for arm, (xy, top) in zip(ARMS, found):
        pos = np.array([xy[0], xy[1], top - 0.016])
        goals = [robot.grasp_at(topdown(box.yaw + k * math.pi, pos), arm=arm) for k in (0, 1)]
        for k, g in enumerate(goals):
            g.rank = k
        p = robot.plan(goals, table, view.points)
        if p is None:
            print(arm, "no grasp plan:", robot.why())
            return
        plans[arm] = p
    out = robot.run(plans)
    print("grasped", out.get("error"), {a: robot.holding(arm=a) for a in ARMS}, "step", robot.status()["step"])
    if out.get("ended") or not all(robot.holding(arm=a) for a in ARMS):
        return
    for rise in (dz + 0.03, 0.005):
        st = robot.status()
        ups = {}
        for arm in ARMS:
            t = st["tcp"][arm].copy()
            t[2, 3] += rise
            ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t, hand=(0.0,)), table)
            if ups[arm] is None:
                print(arm, "no lift plan:", robot.why())
                return
        out = robot.run(ups, sync=True)
        print("lifted", rise, out.get("error"), "step", robot.status()["step"])
        if out.get("ended"):
            return
