"""g1_press_button: the humanoid presses the button of the named colour (perception only: cameras and segmentation).

The hand on the button's side closes (the power preset) and pushes the cap down with the underside of its curled
fingers, coming straight down from above; how far the closed hand reaches past the TCP comes from robot.hand_points."""
import math

import numpy as np

TRAVEL = 0.008            # the cap's travel; a press registers at 5 mm


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


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    names = [n for n in mentioned(instruction, labels) if n.endswith("button")]
    if not names:
        print("no button named in", instruction)
        return
    det = detect(robot, view, table, names[0])
    if det is None:
        print("cannot see", names[0])
        return
    box = robot.box(det.points, table)
    top = box.bottom + box.size[2]
    arm = "left" if box.center[1] > 0 else "right"
    fist = robot.spec.arm(arm).hand.presets["power"]
    lead = float(robot.hand_points("power", arm=arm)[:, 0].max())     # the closed hand's lowest point under the TCP
    print(names[0], "top", round(top, 3), "arm", arm, "hand reaches", round(lead, 3), "past the TCP")
    goals = []
    for k, psi in enumerate((0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)):
        T = np.eye(4)
        y = np.array([math.cos(psi), math.sin(psi), 0.0])
        T[:3, 0], T[:3, 1] = [0.0, 0.0, -1.0], y
        T[:3, 2] = np.cross(T[:3, 0], y)
        T[:3, 3] = [box.center[0], box.center[1], top + lead - TRAVEL + 0.001]
        pre = T.copy()
        pre[2, 3] += 0.06
        goals.append(robot.Goal("press", arm, tcp=T, pre=pre, hand_before=fist, rank=k))
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no press plan:", robot.why())
        return
    out = robot.run(plan)
    if out.get("ended"):
        return
    t = robot.status()["tcp"][arm].copy()
    t[2, 3] += 0.06
    p = robot.plan(robot.Goal("retreat", arm, tcp=t), table)
    if p is not None:
        robot.run(p)
