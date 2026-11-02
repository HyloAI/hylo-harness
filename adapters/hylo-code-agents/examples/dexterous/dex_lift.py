"""dex_lift: a floating multi-finger hand lifts the named object (perception only: cameras and segmentation).
Multi-finger grasps from the object's points (robot.dex_grasps), the first reachable one planned and run, then a
straight lift; a second try from a fresh look when the first one misses."""
import numpy as np


def detect(robot, view, table, label):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= 15]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def solve(robot):
    instruction = robot.status()["instruction"] or ""
    label = instruction.split("lift the", 1)[-1].strip() or instruction
    arm = robot.spec.arms[0].name
    for attempt in range(2):
        view = robot.look()
        table = robot.table(view)
        det = detect(robot, view, table, label)
        if det is None:
            print("cannot see", label)
            return
        goals = robot.dex_grasps(det.points, table, scene=view.points, arm=arm)
        print(f"attempt {attempt}: {len(goals)} grasps")
        plan = robot.plan(goals, table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return
        out = robot.run(plan)
        if out.get("ended"):
            return
        up = robot.status()["tcp"][arm].copy()
        up[2, 3] += 0.14
        lift = robot.plan(robot.Goal("lift", arm, tcp=up), table)
        if lift is None:
            print("no lift plan:", robot.why())
            return
        out = robot.run(lift)
        if out.get("ended"):
            return
        print("missed; opening and trying again")
        robot.run(robot.plan(robot.Goal("retreat", arm, tcp=up, hand=robot.spec.arms[0].hand.open), table))
