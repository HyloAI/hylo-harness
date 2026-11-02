"""g1_two_hands: the humanoid puts each named block on its named plate, both hands at once (perception only).

Each block and its plate lie on one hand's side, and the step limit is shorter than two pick-and-place motions one
after the other: each hand's motion is planned separately (the other hand holds still) and the two run together."""
import numpy as np

ARMS = ("left", "right")


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


def pairs(instruction, labels):
    names = mentioned(instruction, labels)
    return list(zip([n for n in names if n.endswith("block")], [n for n in names if n.endswith("plate")]))


def together(robot, goals, table, scene=None):
    """Plan each arm's goal, run them at once (one after the other if the arms would touch)."""
    plans = {}
    for arm, goal in goals.items():
        p = robot.plan(goal, table, scene)
        if p is None:
            print(arm, "no plan:", robot.why())
        else:
            plans[arm] = p
    if not plans:
        return None
    out = robot.run(plans)
    if out.get("error"):
        print("together refused:", out["error"])
        for arm in plans:
            p = robot.plan(goals[arm], table, scene)
            if p is not None:
                out = robot.run(p)
    return out


def lifts(robot, arms, dz):
    st = robot.status()
    out = {}
    for arm in arms:
        t = st["tcp"][arm].copy()
        t[2, 3] += dz
        out[arm] = robot.Goal("lift", arm, tcp=t)
    return out


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    jobs = {}
    for block, plate in pairs(instruction, labels):
        b, p = detect(robot, view, table, block), detect(robot, view, table, plate)
        if b is None or p is None:
            print("cannot see", block if b is None else plate)
            continue
        arm = "left" if float(np.mean(b.points[:, 1])) > 0 else "right"
        jobs[arm] = {"det": b, "box": robot.box(b.points, table), "target": robot.box(p.points, table).center[:2],
                     "label": block}
    print({a: j["label"] for a, j in jobs.items()})
    # each hand steps through grasp -> lift -> place -> retreat on its own; every round both hands' next motions
    # are planned (the other hand still) and run together, so a hand that missed retries while the other places
    state = {arm: "grasp" for arm in jobs}
    tries = {arm: 0 for arm in jobs}
    for round_ in range(10):
        goals = {}
        for arm, st in state.items():
            j = jobs[arm]
            t = robot.status()["tcp"][arm].copy()
            if st == "grasp":
                cands = robot.hand_grasps(j["det"].points, table, kind="power", arm=arm)
                plan = robot.plan(cands, table, view.points)
                if plan is None:
                    print(arm, "no grasp plan:", robot.why())
                    state[arm] = "done"
                    continue
                j["grasp"] = plan.goal.tcp
                goals[arm] = plan.goal
            elif st in ("lift", "retreat"):
                t[2, 3] += 0.10 if st == "lift" else 0.08
                goals[arm] = robot.Goal("lift", arm, tcp=t)
            elif st == "reopen":
                goals[arm] = robot.Goal("retreat", arm, tcp=t, hand=robot.spec.arm(arm).hand.open)
            elif st == "place":
                g = robot.place_at(np.linalg.inv(j["grasp"]) @ j["box"].pose, j["box"], j["target"], table=table,
                                   hover=0.008, scene=view.points, arm=arm)
                g.free = "x"
                goals[arm] = g
        if not goals:
            break
        out = together(robot, goals, table, view.points if any(state[a] in ("grasp", "place") for a in goals) else None)
        if out is None or out.get("ended"):
            return
        view = robot.look()
        for arm in goals:
            st = state[arm]
            if st == "grasp":
                state[arm] = "lift"
            elif st == "lift":
                state[arm] = "place" if robot.holding(arm=arm) else "reopen"
                print(arm, "holding" if state[arm] == "place" else "missed", "at step", robot.status()["step"])
            elif st == "reopen":
                tries[arm] += 1
                det = detect(robot, view, table, jobs[arm]["label"])
                if det is None or tries[arm] > 2:
                    state[arm] = "done"
                else:
                    jobs[arm]["det"], jobs[arm]["box"] = det, robot.box(det.points, table)
                    state[arm] = "grasp"
            elif st == "place":
                state[arm] = "retreat"
            elif st == "retreat":
                state[arm] = "done"
        state = {a: s for a, s in state.items() if s != "done"}
