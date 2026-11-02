"""put_in_bowl: put the named object in the named bowl (perception only: cameras and segmentation)."""
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


def parse(instruction, labels):
    """(object, bowl) from "put the X in the W", "drop the X into the W", "place the X inside the W"."""
    text = instruction.lower().strip().rstrip(".")
    cut = max(text.rfind(" in "), text.rfind(" into "), text.rfind(" inside "))
    head, tail = (text[:cut], text[cut:]) if cut >= 0 else (text, "")
    obj, tgt = mentioned(head, labels), mentioned(tail, labels)
    if obj and tgt:
        return obj[0], tgt[0]
    for p in ("put the ", "place the ", "drop the "):                  # no segment labels: use the words
        if head.startswith(p):
            head = head[len(p):]
    for p in (" inside the ", " into the ", " in the "):
        tail = tail.replace(p, "")
    return head.strip(), tail.strip()


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 20]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def up(robot, dz, table, scene=None):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t), table, scene)
    return robot.run(p) if p is not None else None


def pick(robot, label, table, tries=3):
    """Grasp and lift the object; returns (grasp tcp, box) or None."""
    for attempt in range(tries):
        view = robot.look()
        det = detect(robot, view, table, label)
        if det is None:
            print("cannot see", label)
            return None
        box = robot.box(det.points, table)
        plan = robot.plan(robot.grasps(ARM, det.points, table, scene=view.points), table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding(ARM)
        up(robot, 0.12, table)
        if held and robot.holding(ARM):
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed; retrying")
        opened = robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,))
        p = robot.plan(opened, table)
        if p is not None:
            robot.run(p)
    return None


def place(robot, grasp_tcp, box, xy, table, yaw=None, hover=0.012):
    view = robot.look()
    goal = robot.place_at(ARM, np.linalg.inv(grasp_tcp) @ box.pose, box, xy, yaw=yaw, table=table, hover=hover,
                          scene=view.points)
    plan = robot.plan(goal, table, view.points)
    if plan is None:
        print("no place plan:", robot.why())
        return False
    out = robot.run(plan)
    up(robot, 0.10, table)
    return not out.get("ended")


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    item, target = parse(instruction, labels)
    print("object:", item, "| bowl:", target)
    bowl = detect(robot, view, table, target)
    if bowl is None:
        print("cannot see", target)
        return
    bbox = robot.box(bowl.points, table)
    got = pick(robot, item, table)
    if got is None:
        return
    # the bowl's centre: the middle of its rim (the box of its points); drop from just above the bottom
    place(robot, got[0], got[1], bbox.center[:2], table, hover=0.015)
