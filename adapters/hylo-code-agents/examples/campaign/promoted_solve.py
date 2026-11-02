import math
import numpy as np

ARM = "arm"


def mentioned(text, labels):
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


def stack_order(instruction, labels):
    text = instruction.lower().strip().rstrip(".")
    names = mentioned(text, labels)
    if "bottom" in text:
        order = []
        for n in names:
            if n not in order:
                order.append(n)
        return order
    on = {}
    for clause in text.replace(", then", ";").replace(" then ", ";").split(";"):
        cut = clause.rfind(" on ")
        if cut < 0:
            continue
        a, b = mentioned(clause[:cut], labels), mentioned(clause[cut:], labels)
        if not a and b and names:
            a = [n for n in mentioned(clause, labels) if n not in b][:1]
        if a and b:
            on[a[0]] = b[0]
    lowers = set(on.values())
    bottom = [b for b in lowers if b not in on]
    if not bottom:
        return names
    order = [bottom[0]]
    above = {v: k for k, v in on.items()}
    while order[-1] in above and above[order[-1]] not in order:
        order.append(above[order[-1]])
    return order


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
            continue
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding(ARM)
        up(robot, 0.12, table)
        if held and robot.holding(ARM):
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed; retrying")
        p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
        if p is not None:
            robot.run(p)
    return None


def held_yaw(robot, grasp_tcp, box):
    T = robot.status()["tcp"][ARM] @ np.linalg.inv(grasp_tcp) @ box.pose
    return math.atan2(T[1, 0], T[0, 0])


def aligned(yaw, ref):
    k = round((yaw - ref) / (math.pi / 2))
    return ref + k * math.pi / 2


def stacked(lower_box, upper_box):
    return (np.linalg.norm(upper_box.center[:2] - lower_box.center[:2]) < 0.015
            and abs((upper_box.bottom + upper_box.size[2]) - (lower_box.bottom + lower_box.size[2]) - 0.045) < 0.01)


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    order = stack_order(instruction, labels)
    print("bottom to top:", order)
    prev_box = None
    for idx, label in enumerate(order):
        print("handling", label)
        # pick the block
        res = pick(robot, label, table)
        if res is None:
            print("failed to pick", label)
            return
        grasp_tcp, box = res
        # if not the first block, place on previous
        if prev_box is not None:
            yaw = aligned(held_yaw(robot, grasp_tcp, box), prev_box.yaw)
            view = robot.look()
            goal = robot.place_at(ARM, np.linalg.inv(grasp_tcp) @ box.pose, box,
                                  xy=prev_box.center[:2], yaw=yaw,
                                  table=table, hover=0.004, scene=view.points)
            plan = robot.plan(goal, table, view.points)
            if plan is None:
                print("no place plan:", robot.why())
                return
            out = robot.run(plan)
            up(robot, 0.08, table)
            if out.get("ended"):
                return
            # verify placement
            view = robot.look()
            top = detect(robot, view, table, label)
            if top is None:
                print("could not verify placement of", label)
                return
            tbox = robot.box(top.points, table)
            print("level", idx, "offset", np.round(tbox.center[:2] - prev_box.center[:2], 3),
                  "top height", round(tbox.bottom + tbox.size[2], 3))
            if not stacked(prev_box, tbox):
                print("placement not stacked correctly; aborting")
                return
            prev_box = tbox
        else:
            # first block: just lift and keep
            prev_box = box
    print("stacking complete")
