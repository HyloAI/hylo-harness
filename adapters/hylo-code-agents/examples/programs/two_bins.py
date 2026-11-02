"""two_bins: put each block in the named bin, both arms at once (perception only: cameras and segmentation).

Each block and its bin sit on one arm's side of the table, and the step limit is shorter than two pick-and-place
motions one after the other: the program plans each arm's motion and runs them together."""
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


def pairs(instruction, labels):
    """[(block label, bin label)] from "... the X block in the Y bin and the Z block in the W bin ..."."""
    names = mentioned(instruction, labels)
    blocks = [n for n in names if n.endswith("block")]
    bins = [n for n in names if n.endswith("bin")]
    return list(zip(blocks, bins))


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 15]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def side(points):
    return "left" if float(np.mean(points[:, 0])) < 0 else "right"


def lift_goals(robot, arms, dz, hand=None):
    st = robot.status()
    goals = {}
    for arm in arms:
        t = st["tcp"][arm].copy()
        t[2, 3] += dz
        goals[arm] = robot.Goal("lift", arm, tcp=t, hand=hand)
    return goals


def run_together(robot, goals, table, scene=None):
    """Plan each arm's goal (the others hold still), then run all of them at once."""
    plans = {}
    for arm, goal in goals.items():
        p = robot.plan(goal, table, scene)
        if p is None:
            print(arm, "no plan:", robot.why())
            continue
        plans[arm] = p
    if not plans:
        return None
    out = robot.run(plans)
    if out.get("error"):
        print("together refused:", out["error"], "- running one after the other")
        for arm, goal in goals.items():
            p = robot.plan(goal, table, scene)
            if p is not None:
                out = robot.run(p)
    return out


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    jobs = {}
    for block, bin_ in pairs(instruction, labels):
        b = detect(robot, view, table, block)
        t = detect(robot, view, table, bin_)
        if b is None or t is None:
            print("cannot see", block if b is None else bin_)
            continue
        jobs[side(b.points)] = {"block": block, "bin": bin_, "det": b, "box": robot.box(b.points, table),
                                "target": robot.box(t.points, table).center[:2]}
    print({arm: (j["block"], j["bin"]) for arm, j in jobs.items()})
    for attempt in range(2):
        todo = [arm for arm in ARMS if arm in jobs and not jobs[arm].get("held")]
        if not todo:
            break
        grasps = {}
        for arm in todo:
            j = jobs[arm]
            cands = robot.grasps(j["det"].points, table, scene=view.points, arm=arm)
            if cands:
                grasps[arm] = cands
        plans = {}
        for arm, cands in grasps.items():
            p = robot.plan(cands, table, view.points)
            if p is None:
                print(arm, "no grasp plan:", robot.why())
                continue
            plans[arm] = p
        if not plans:
            return
        out = robot.run(plans)
        if out.get("error"):
            print("together refused:", out["error"])
            for arm, p in plans.items():
                robot.run(robot.plan(p.goal, table, view.points))
        if out.get("ended"):
            return
        for arm, p in plans.items():
            jobs[arm]["grasp"] = p.goal.tcp
            jobs[arm]["held"] = robot.holding(arm=arm)
        print("holding", {arm: jobs[arm].get("held") for arm in plans}, "step", robot.status()["step"])
        run_together(robot, lift_goals(robot, list(plans), 0.10), table)
        missed = [arm for arm in plans if not jobs[arm]["held"]]
        if missed:
            run_together(robot, lift_goals(robot, missed, 0.0, hand=(1.0,)), table)
            view = robot.look()
            for arm in missed:
                det = detect(robot, view, table, jobs[arm]["block"])
                if det is not None:
                    jobs[arm]["det"], jobs[arm]["box"] = det, robot.box(det.points, table)
    view = robot.look()
    goals = {}
    for arm, j in jobs.items():
        if not j.get("held"):
            continue
        obj_in_tcp = np.linalg.inv(j["grasp"]) @ j["box"].pose
        goal = robot.place_at(obj_in_tcp, j["box"], j["target"], table=table, hover=0.015,
                              support=table.height + 0.006, arm=arm)
        goal.free = "x"                      # any yaw in the bin: a six-joint arm cannot keep every orientation
        goals[arm] = goal
    print("placing at step", robot.status()["step"])
    out = run_together(robot, goals, table, view.points)
    print("placed at step", robot.status()["step"])
    if out is None or out.get("ended"):
        return
    run_together(robot, lift_goals(robot, list(goals), 0.08), table)
