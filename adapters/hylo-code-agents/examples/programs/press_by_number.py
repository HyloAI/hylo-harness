"""press_by_number: press the numbered buttons in the order the instruction gives (perception only)."""
import numpy as np

ARM = "arm"


def digits_in(text):
    return [int(ch) for ch in text if ch.isdigit()]


def press_order(instruction, available):
    """The digits to press, in order: listed explicitly, or all of them ascending / descending."""
    text = instruction.lower()
    listed = digits_in(text)
    if listed:
        return listed
    ascending = "ascending" in text or text.find("lowest") < text.find("highest")
    return sorted(available, reverse=not ascending)


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 20]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def topdown(at, yaw=0.0):
    """TCP pose pointing straight down at `at`, closing axis at `yaw`."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
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


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    buttons = {}
    for label in view.segments.values():
        if label.startswith("button") and digits_in(label):
            buttons[digits_in(label)[0]] = label
    order = press_order(instruction, sorted(buttons))
    print("buttons", sorted(buttons), "press", order)
    for k in order:
        if k not in buttons:
            print("no button", k)
            return
        view = robot.look()
        det = detect(robot, view, table, buttons[k])
        if det is None:
            print("cannot see", buttons[k])
            return
        box = robot.box(det.points, table)
        top = box.bottom + box.size[2]
        x, y = box.center[0], box.center[1]
        yaw = np.arctan2(y, x + 0.5) + np.pi / 2           # closing axis across the reach direction
        if move(robot, "reach", topdown([x, y, top + 0.10], yaw), table, view.points, hand_before=(0.0,)) is None:
            return
        move(robot, "press", topdown([x, y, top + 0.02], yaw), table)
        # the cap travels 8 mm against a spring and registers at 5: aim past the end of its travel, the joint
        # servos stop on it
        out = move(robot, "press", topdown([x, y, top - 0.010], yaw), table)
        move(robot, "retreat", topdown([x, y, top + 0.10], yaw), table)
        if out is not None and out.get("ended"):
            return
