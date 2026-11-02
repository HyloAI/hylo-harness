"""sort_by_colour: put every block in the tray of its colour (perception only: cameras and segmentation)."""
import math

import numpy as np

ARM = "arm"


def detect(robot, view, table, label):
    dets = robot.objects(view, robot.find(view, label), table)
    dets = [d for d in dets if len(d.points) >= 20]
    return max(dets, key=lambda d: (d.score, len(d.points))) if dets else None


def up(robot, dz, table, scene=None):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t), table, scene)
    return robot.run(p) if p is not None else None


def pick_at(robot, label, xy, table, tries=3):
    """Grasp and lift the instance of `label` nearest `xy`; returns (grasp tcp, box) or None."""
    for attempt in range(tries):
        view = robot.look()
        dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= 20]
        det = min(dets, key=lambda d: np.linalg.norm(d.points.mean(0)[:2] - xy)) if dets else None
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
        p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
        if p is not None:
            robot.run(p)
    return None


def place_yaw(robot, grasp_tcp, box, ref, across=None):
    """Object yaw for placing: lined up with `ref` (a cube turns by quarter turns), the fingers' closing axis
    across `across` (a world xy direction) when given, else the turn nearest the object's yaw now."""
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    closing = np.linalg.inv(obj_in_tcp)[:3, 1]                  # the closing axis in the object frame
    T = robot.status()["tcp"][ARM] @ obj_in_tcp
    now = math.atan2(T[1, 0], T[0, 0])
    best = None
    for k in range(4):
        yaw = ref + k * math.pi / 2
        yaw = now + math.remainder(yaw - now, 2 * math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        axis = np.array([c * closing[0] - s * closing[1], s * closing[0] + c * closing[1]])
        cost = abs(yaw - now)
        if across is not None:
            cost += 10 * abs(float(axis @ across)) / (np.linalg.norm(axis) + 1e-9)
        if best is None or cost < best[0]:
            best = (cost, yaw)
    return best[1]


def colour_of(label):
    return label.split()[0]


def in_tray(xy, tray, margin=0.04):
    ax = np.array([math.cos(tray.yaw), math.sin(tray.yaw)])
    local = (xy - tray.center[:2]) @ np.array([ax, [-ax[1], ax[0]]]).T
    return bool(np.all(np.abs(local) < margin))


def blocks_seen(robot, view, table, labels):
    """(label, xy) of every block instance in view (two blocks of one colour share a label)."""
    out = []
    for lab in sorted(set(labels)):
        for det in robot.objects(view, robot.find(view, lab), table):
            if len(det.points) >= 20:
                out.append((lab, robot.box(det.points, table).center[:2]))
    return out


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    labels = sorted(v for v in view.segments.values() if v not in ("table", "robot"))
    trays = {colour_of(lab): lab for lab in labels if "tray" in lab}
    blocks = [lab for lab in labels if "tray" not in lab and colour_of(lab) in trays]
    print("trays", sorted(trays.values()), "blocks", blocks)
    boxes = {}
    for colour, lab in trays.items():
        det = detect(robot, view, table, lab)
        if det is not None:
            boxes[colour] = robot.box(det.points, table)
    for sweep in range(3):                       # place every block, then look again and fix what is off
        todo = blocks_seen(robot, view, table, blocks)
        count = {}
        for lab, xy in todo:
            count[colour_of(lab)] = count.get(colour_of(lab), 0) + 1
        moved = 0
        for lab, xy0 in todo:
            colour = colour_of(lab)
            if colour not in boxes:
                print("cannot see the", colour, "tray")
                continue
            tray = boxes[colour]
            if in_tray(xy0, tray):
                continue                                              # already in its tray
            ax = np.array([math.cos(tray.yaw), math.sin(tray.yaw)])   # two blocks: side by side along x
            if count[colour] == 1:
                slot, across = tray.center[:2], None
            else:                                                     # the free side of the tray
                others = [xy for lb, xy in blocks_seen(robot, robot.look(), table, [lab])
                          if in_tray(xy, tray) and colour_of(lb) == colour]
                side = -1.0 if others and float((others[0] - tray.center[:2]) @ ax) > 0 else 1.0
                slot, across = tray.center[:2] + side * 0.024 * ax, ax
            got = pick_at(robot, lab, xy0, table)
            if got is None:
                continue
            yaw = place_yaw(robot, got[0], got[1], tray.yaw, across)
            view = robot.look()
            goal = robot.place_at(ARM, np.linalg.inv(got[0]) @ got[1].pose, got[1], slot, yaw=yaw, table=table,
                                  hover=0.008, scene=view.points)
            plan = robot.plan(goal, table, view.points)
            if plan is None:
                print("no place plan:", robot.why())
                continue
            out = robot.run(plan)
            up(robot, 0.10, table)
            moved += 1
            if out.get("ended"):
                return
        if moved == 0:
            break
        view = robot.look()
        print("sweep", sweep, "moved", moved)
