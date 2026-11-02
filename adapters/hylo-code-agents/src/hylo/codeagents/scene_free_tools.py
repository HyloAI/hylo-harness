"""The privileged toolkit for free scenes: hylo-manip's Tools plus the simulator's free-form state.

    tools: {type: hylo.codeagents.scene_free_tools:PrivilegedFreeTools}     (env mujoco.free_scene, privileged_obs)

robot.scene() returns rigid bodies, cable nodes, flex vertices, joint values and sites; robot.points(name) turns a
body, cable or flex into points for the grasping verbs; robot.top_down(xyz, yaw) builds a top-down TCP pose. The
reference text (with a one-arm or a two-arm example, by the robot) is prepended to the toolkit's own.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hylo.manip.tools import Table, Tools

PRIVILEGED_FREE_TOOLS = "hylo.codeagents.scene_free_tools:PrivilegedFreeTools"


class PrivilegedFreeTools(Tools):
    """The hylo-manip toolkit plus PRIVILEGED free-scene state (the env must run with privileged_obs: true)."""

    privileged = True

    def _priv(self) -> dict[str, Any]:
        priv = (self.executor.observation().extra or {}).get("privileged")
        if not priv:
            raise RuntimeError("no privileged state in the observation: run the env with privileged_obs: true")
        return priv

    def scene(self) -> dict[str, Any]:
        """PRIVILEGED: the scene now, from the simulator: {"bodies": {name: {"pos", "quat", "bounds": (min, max),
        "top", "free", "mocap", "parent", "label"}}, "cables": {prefix: {"nodes" (N, 3) in chain order, "bodies",
        "radius", "label"}}, "flexes": {name: {"vertices" (N, 3), "bodies", "radius", "label"}}, "joints": {name:
        {"type", "value", "range", "units", "body"}}, "sites": {name: (3,)}, "table": {"height", "x", "y"}}."""
        p = self._priv()
        arr = np.asarray
        out = {"bodies": {}, "cables": {}, "flexes": {}, "joints": dict(p["joints"]),
               "sites": {k: arr(v) for k, v in p["sites"].items()}, "table": dict(p["table"])}
        for n, b in p["bodies"].items():
            out["bodies"][n] = {**b, "pos": arr(b["pos"]), "quat": arr(b["quat"]),
                                "bounds": (arr(b["bounds"][0]), arr(b["bounds"][1]))}
        for n, c in p["cables"].items():
            out["cables"][n] = {**c, "nodes": arr(c["nodes"])}
        for n, f in p["flexes"].items():
            out["flexes"][n] = {**f, "vertices": arr(f["vertices"])}
        return out

    def points(self, name: str, spacing: float = 0.006) -> np.ndarray:
        """PRIVILEGED: (N, 3) world points of a thing for the toolkit's grasping verbs: a body's bounding-box
        surface, a cable's nodes filled in along the chain (and around its radius), a flex's vertices."""
        p = self._priv()
        if name in p["cables"]:
            c = p["cables"][name]
            n = np.asarray(c["nodes"], float)
            r = float(c["radius"]) or 0.005
            seg = [n[i] + (n[i + 1] - n[i]) * t for i in range(len(n) - 1)
                   for t in np.linspace(0, 1, max(2, int(np.linalg.norm(n[i + 1] - n[i]) / spacing) + 1))]
            pts = np.asarray(seg)
            ring = [np.array([dx, dy, dz]) * r for dx, dy, dz in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0),
                                                                    (0, 1, 0), (0, -1, 0))]
            return np.concatenate([pts + o for o in ring])
        if name in p["flexes"]:
            return np.asarray(p["flexes"][name]["vertices"], float)
        if name not in p["bodies"]:
            raise KeyError(f"no body, cable or flex {name!r}; bodies {sorted(p['bodies'])}, cables "
                           f"{sorted(p['cables'])}, flexes {sorted(p['flexes'])}")
        lo, hi = np.asarray(p["bodies"][name]["bounds"][0]), np.asarray(p["bodies"][name]["bounds"][1])
        out = []
        for ax in range(3):
            u, v = [k for k in range(3) if k != ax]
            gu = np.linspace(lo[u], hi[u], max(2, int(np.ceil((hi[u] - lo[u]) / spacing)) + 1))
            gv = np.linspace(lo[v], hi[v], max(2, int(np.ceil((hi[v] - lo[v]) / spacing)) + 1))
            uu, vv = np.meshgrid(gu, gv)
            for side in (lo[ax], hi[ax]):
                q = np.zeros((uu.size, 3))
                q[:, u], q[:, v], q[:, ax] = uu.ravel(), vv.ravel(), side
                out.append(q)
        return np.concatenate(out)

    def top_down(self, xyz: Sequence[float], yaw: float = 0.0) -> np.ndarray:
        """A top-down TCP pose (4x4) at `xyz`: approach (+x) straight down, closing axis (+y) at angle `yaw` about
        world z (0: fingers close along world y). Use with robot.grasp_at / robot.Goal."""
        a = np.array([0.0, 0.0, -1.0])
        y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        T = np.eye(4)
        T[:3, :3] = np.stack([a, y, np.cross(a, y)], axis=1)
        T[:3, 3] = np.asarray(xyz, float)
        return T

    def scene_table(self, height: float | None = None) -> Table:
        """PRIVILEGED: the table as a Table (height from the simulator), or a support plane at `height`."""
        t = self._priv()["table"]
        x, y = t["x"], t["y"]
        return Table(height=float(t["height"] if height is None else height), normal=np.array([0.0, 0.0, 1.0]),
                     extent=np.array([x[1] - x[0], y[1] - y[0]], float))

    def extra_reference(self) -> str:
        two = len(self.spec.arms) > 1
        return FREE_REFERENCE + (FREE_TWO_ARM_EXAMPLE if two else FREE_ONE_ARM_EXAMPLE)


FREE_REFERENCE = """## Privileged simulator state (this run is flagged privileged)
This task was built as a free-form scene; robot.scene() gives the simulator's ground truth. Prefer it to perception.
robot.scene() -> dict
    {"bodies": {name: {"pos", "quat", "bounds": (min, max), "top", "free", "mocap", "parent", "label"}},
     "cables": {prefix: {"nodes": (N, 3) in chain order, "bodies", "radius", "label"}},
     "flexes": {name: {"vertices": (N, 3), "bodies", "radius", "label"}},
     "joints": {name: {"type": "hinge" | "slide", "value", "range", "units", "body"}},
     "sites": {name: (3,)}, "table": {"height", "x", "y"}}; fresh at every call.
robot.points(name, spacing=0.006) -> (N, 3) world points of a body, cable or flex (for robot.grasps / robot.box).
robot.top_down(xyz, yaw=0.0) -> 4x4 top-down TCP pose at xyz (closing axis at angle yaw about z): for grasp_at.
robot.scene_table(height=None) -> Table: the table, or a support plane at `height`.
Thin things (a cable end, a cloth edge): grasp with robot.grasp_at(robot.top_down(p, yaw), arm=...) where p is the
node / vertex position but at least 1.7 cm above the surface under it (the fingertips reach about 1.5 cm below the
TCP, and plan() refuses poses whose fingers would hit the table); yaw across the cable (perpendicular to the line
between neighbouring nodes). Move a held thing with robot.Goal("move", arm,
tcp=T) through robot.plan(goal, table) (a joint-space path) and check robot.scene() after every motion.
"""

FREE_ONE_ARM_EXAMPLE = """
Example (one arm: lift the end of a cable and lay it at a point):
    s = robot.scene(); table = robot.scene_table()
    n = s["cables"]["rope"]["nodes"]; end, nxt = n[-1], n[-2]
    d = end - nxt; yaw = math.atan2(d[1], d[0]) + math.pi / 2
    p = end.copy(); p[2] = max(p[2], table.height + 0.017)
    out = robot.run(robot.plan(robot.grasp_at(robot.top_down(p, yaw)), table)); print(out["tcp_error"])
    up = out["tcp"]["arm"].copy(); up[2, 3] += 0.10
    robot.run(robot.plan(robot.Goal("move", tcp=up), table))
    to = robot.top_down([0.1, 0.2, table.height + 0.05], yaw)
    robot.run(robot.plan(robot.Goal("move", tcp=to, hand=robot.spec.arms[0].hand.open), table))
    print(robot.scene()["cables"]["rope"]["nodes"][-1])
"""

FREE_TWO_ARM_EXAMPLE = """
Example (two arms: take both ends of a cable and lift them together):
    s = robot.scene(); table = robot.scene_table()
    n = s["cables"]["rope"]["nodes"]
    plans = {}
    for arm, i, j in (("left", 0, 1), ("right", -1, -2)):
        d = n[i] - n[j]; yaw = math.atan2(d[1], d[0]) + math.pi / 2
        p = n[i].copy(); p[2] = max(p[2], table.height + 0.017)
        plans[arm] = robot.plan(robot.grasp_at(robot.top_down(p, yaw), arm=arm), table)
        if plans[arm] is None: print(arm, robot.why())
    out = robot.run(plans); print(out.get("error"), out["tcp_error"])
    st = robot.status(); ups = {}
    for arm in ("left", "right"):
        t = st["tcp"][arm].copy(); t[2, 3] += 0.15
        ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t), table)
    out = robot.run(ups, sync=True)
    print(robot.scene()["cables"]["rope"]["nodes"][[0, -1]])
"""



__all__ = ["FREE_ONE_ARM_EXAMPLE", "FREE_REFERENCE", "FREE_TWO_ARM_EXAMPLE", "PRIVILEGED_FREE_TOOLS",
           "PrivilegedFreeTools"]
