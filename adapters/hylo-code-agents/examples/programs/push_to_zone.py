"""push_to_zone: push the large box (too wide to grasp) into the named zone (perception only)."""
import math

import numpy as np

ARM = "arm"


def mentioned(text, labels):
    """Segment labels the text names, in the order they appear (longest label wins where labels overlap)."""
    text = " " + text.lower().replace(",", " ").replace(".", " ") + " "
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
    dets = [d for d in dets if len(d.points) >= 20]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def topdown(at, closing):
    """TCP pose pointing straight down at `at` with the closing axis along `closing` (xy)."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([closing[0], closing[1], 0.0]) / np.linalg.norm(closing)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, np.cross(x, y)
    T[:3, 3] = at
    return T


def move(robot, verb, T, table, scene=None, **kw):
    plan = robot.plan(robot.Goal(verb, ARM, tcp=T, **kw), table, scene)
    if plan is None:
        print(verb, "failed:", robot.why())
        return None
    return robot.run(plan)


def inside(box, zbox, margin):
    c, s = math.cos(zbox.yaw), math.sin(zbox.yaw)
    d = box.center[:2] - zbox.center[:2]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return bool(np.all(np.abs(local) < zbox.size[0] / 2 - margin))


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    zones = [lab for lab in labels if "zone" in lab]
    named = [lab for lab in mentioned(instruction, labels) if lab in zones]
    zone_label = named[0] if named else (zones[0] if zones else "zone")
    box_label = next((lab for lab in labels if "box" in lab), "large box")
    print("push", box_label, "into", zone_label)
    zone = detect(robot, view, table, zone_label)
    if zone is None:
        print("cannot see", zone_label)
        return
    zbox = robot.box(zone.points, table)
    h = table.height
    for attempt in range(4):
        view = robot.look()
        det = detect(robot, view, table, box_label)
        if det is None:
            print("cannot see", box_label)
            return
        box = robot.box(det.points, table)
        if inside(box, zbox, 0.035):
            print("inside after", attempt, "pushes")
            break
        d = zbox.center[:2] - box.center[:2]
        dist = float(np.linalg.norm(d))
        u = d / dist
        rel = math.atan2(u[1], u[0]) - box.yaw
        ext = box.size[0] / 2 * abs(math.cos(rel)) + box.size[1] / 2 * abs(math.sin(rel))
        side = np.array([-u[1], u[0]])
        start = box.center[:2] - u * (ext + 0.045)
        end = box.center[:2] - u * (ext + 0.012) + u * dist
        zc = h + 0.022
        if move(robot, "reach", topdown([start[0], start[1], h + 0.12], side), table, view.points,
                hand_before=(1.0,)) is None:
            return
        move(robot, "push", topdown([start[0], start[1], zc], side), table)
        out = move(robot, "push", topdown([end[0], end[1], zc], side), table)
        back = end - u * 0.04
        move(robot, "retreat", topdown([back[0], back[1], zc], side), table)
        move(robot, "lift", topdown([back[0], back[1], h + 0.12], side), table)
        if out is not None and out.get("ended"):
            return
