"""The `Tools` facade (RFC-0006 part 2): the verbs a program uses as `robot`. Stateless: the program keeps its own
list of objects and of what each hand holds; every verb reads the robot afresh. The implementations live in the
modules beside it (kinematics, collision, planning, perception, grasping); `api_reference(tools)` renders the verbs'
docstrings as the reference a code-writing model reads.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from hylo.core.robot import RobotSpec

from .datatypes import Box, CameraView, Detection, Fit, Goal, Mask, Plan, Table, Verb, View
from .executor import Executor
from .geometry import angle, apply, inv, rotz, yaw_of


class Tools:
    """`robot` in programs. `executor` moves the robot; `segmenter` answers `find` (default: the environment's
    segmentation when the observation has it, else colour names); `show` collects images for the model. `max_vel`
    and `max_acc` (per joint or scalar, rad/s and rad/s^2) bound planned motions; `seed` makes planning repeatable."""

    def __init__(self, spec: RobotSpec, executor: Executor, *, segmenter: Any | None = None,
                 control_hz: float = 20.0, show: Callable[[np.ndarray, str], None] | None = None,
                 max_vel: float | Sequence[float] = 1.0, max_acc: float | Sequence[float] = 2.0, seed: int = 0) -> None:
        from .planning import PlannerConfig

        self.spec, self.executor, self.segmenter, self.control_hz = spec, executor, segmenter, control_hz
        self._show = show
        self.config = PlannerConfig(control_hz=control_hz, max_vel=max_vel, max_acc=max_acc, seed=seed)
        self._kin = self._colliders = self._planner = None
        self._hand_cmd: dict[str, np.ndarray] = {}
        self._hand_pts: dict[str, np.ndarray] = {}
        self._dex_cache: dict[str, Any] = {}                 # arm -> (HandModel, DexGraspPlanner)
        self._why = ""
        self._looks: list[tuple[Any, dict, dict]] = []   # (weakref to view.points, q, hand) of recent looks

    Goal = Goal                         # robot.Goal(...) in programs

    # ------------------------------------------------------------------ internals
    def _models(self):
        if self._kin is None:
            from .collision import Collider, _subtree
            from .kinematics import Kinematics
            from .planning import Planner

            kin = Kinematics(self.spec)
            for a in kin.arms.values():                  # every arm at home while each collider records the
                a.set_q(a.home)                          # contacts that are normal (arms sharing a model)
                a.set_hand(a.hand_open)
            bodies: dict[int, set[int]] = {}
            for a in kin.arms.values():
                own = bodies.setdefault(id(a.m), set())
                for j in a.jids:
                    b = int(a.m.jnt_bodyid[j])
                    own |= _subtree(a.m, b)
                    while b > 0:
                        own.add(b)
                        b = int(a.m.body_parentid[b])
            self._colliders = {n: Collider(a, bodies[id(a.m)], seed=self.config.seed) for n, a in kin.arms.items()}
            self._planner = Planner(kin, self._colliders, self.config)
            self._kin = kin
        return self._kin, self._colliders, self._planner

    def _arm(self, arm: str):
        return self._models()[0][arm]

    def _name(self, arm: str | None) -> str:
        """The arm a verb acts on: `arm`, or the only arm when None."""
        if arm is None:
            if len(self.spec.arms) != 1:
                raise ValueError(f"this robot has arms {[a.name for a in self.spec.arms]}: pass arm=")
            return self.spec.arms[0].name
        self.spec.arm(arm)                               # KeyError naming the arms when unknown
        return arm

    def _goal_arm(self, g: Goal) -> Goal:
        if g.arm is None:
            g.arm = self._name(None)
        return g

    def _state(self) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
        obs = self.executor.observation()
        q = {a.name: np.asarray(obs.state[f"q.{a.name}"], float).ravel() for a in self.spec.arms}
        hand = {a.name: np.asarray(obs.state.get(f"hand.{a.name}", a.hand.open), float).ravel() for a in self.spec.arms}
        return obs, q, hand

    def _hand_frac(self, arm: str, measured: np.ndarray) -> float:
        h = self.spec.arm(arm).hand
        span = h.open[0] - h.closed[0]
        return float((measured[0] - h.closed[0]) / span) if span else 0.0

    def _keep(self, hand: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Hand command each arm keeps while another motion runs: the last one this toolkit sent, else closed when
        holding, else the measured state snapped to open / closed when near them."""
        out = {}
        for a in self.spec.arms:
            if a.name in self._hand_cmd:
                out[a.name] = self._hand_cmd[a.name]
            elif a.hand.kind == "parallel":
                f = self._hand_frac(a.name, hand[a.name])
                out[a.name] = np.array(a.hand.open if f > 0.9 else a.hand.closed if f < 0.1 else
                                       a.hand.closed if self.holding(a.name) else hand[a.name], float)
            else:
                out[a.name] = hand[a.name]
        return out

    def _scene(self, scene: np.ndarray | None, table: Table | None, q: dict[str, np.ndarray],
               hand: dict[str, np.ndarray]) -> np.ndarray | None:
        """Observed points without the table and the robot."""
        if scene is None:
            return None
        pts = np.asarray(scene, float).reshape(-1, 3)
        if table is not None:
            pts = pts[pts[:, 2] > table.height + 0.01]
        _, _, planner = self._models()
        robot = [p for qq, hh in [(q, hand), *self._seen_at(scene)] for p in planner.robot_points(qq, hh).values()]
        R = np.concatenate(robot)
        if len(R) and len(pts):
            d, _ = cKDTree(R).query(pts, distance_upper_bound=self.config.self_filter)
            pts = pts[~np.isfinite(d)]
        return pts

    # ------------------------------------------------------------------ state
    def status(self) -> dict[str, Any]:
        """Robot and episode now: {"step", "max_steps", "instruction", "ended", "q": {arm: (n,) joint positions},
        "hand": {arm: measured hand state, 0 closed .. 1 open}, "tcp": {arm: (4, 4) TCP pose in the world}}."""
        st = dict(self.executor.status())
        _, q, hand = self._state()
        st.update(q=q, hand=hand, tcp={a: self._arm(a).fk(q[a]) for a in q})
        return st

    def look(self) -> View:
        """The latest observation: per-camera RGB, depth, intrinsics, pose and segmentation (`view.cams`), fused world
        points of every depth camera (`view.points`, 5 mm voxels; includes table and robot), `view.segments`."""
        from .perception import fuse

        obs = self.executor.observation()
        cams_meta = (obs.extra or {}).get("cameras", {}) or {}
        cams = {}
        for name, img in obs.images.items():
            if "." in name:
                continue
            meta = cams_meta.get(name, {}) or {}
            K, T = meta.get("K"), meta.get("T_world_cam")
            cams[name] = CameraView(rgb=np.asarray(img), depth=_arr(obs.images.get(f"{name}.depth")),
                                    K=_arr(K), T_world_cam=_arr(T), seg=_arr(obs.images.get(f"{name}.seg")))
        segs = {int(k): str(v) for k, v in ((obs.extra or {}).get("segments", {}) or {}).items()}
        step = int(self.executor.status().get("step", 0))
        view = View(cams=cams, points=fuse(cams), segments=segs, step=step)
        _, q, hand = self._state()
        self._looks = [x for x in self._looks if x[0]() is not None][-7:] + [(weakref.ref(view.points), q, hand)]
        return view

    def _seen_at(self, scene) -> list[tuple[dict, dict]]:
        """The robot configuration when `scene` (a view's points) was observed, so its robot points can go."""
        return [(q, h) for ref, q, h in self._looks if scene is not None and ref() is scene]

    # ------------------------------------------------------------------ perception
    def find(self, view: View, labels: str | Sequence[str]) -> list[Mask]:
        """Instance masks for text labels (one Mask per instance per camera). With environment segmentation a label
        matches segment labels with the same content words (score 1), else those sharing the most words (score =
        word overlap); with the colour segmenter only the colour word counts. Empty list: nothing matched."""
        from .perception import ColourSegmenter, EnvSegmenter

        labels = [labels] if isinstance(labels, str) else list(labels)
        seg = self.segmenter
        if seg is None:
            has_seg = view.segments and any(c.seg is not None for c in view.cams.values())
            seg = EnvSegmenter() if has_seg else ColourSegmenter()
        return seg.segment(view, labels)

    def table(self, view: View) -> Table:
        """The support plane from `view.points` by RANSAC (near-horizontal planes only; near
        `RobotSpec.table_height` when given): height, normal, xy extent of its inliers."""
        from .perception import ransac_table

        return ransac_table(view.points, np.random.default_rng(self.config.seed), hint=self.spec.table_height)

    def objects(self, view: View, masks: Sequence[Mask], table: Table) -> list[Detection]:
        """Per label (and instance when known): the masked depth as world points, table points and outliers removed
        (flat markings on the table, e.g. zones, keep their points at table height), cameras merged, split into
        separate 3D clusters (one Detection each, largest first)."""
        from .perception import detect_objects

        return detect_objects(view, masks, table)

    def box(self, points: np.ndarray, table: Table | None = None) -> Box:
        """Oriented box around object points: minimum-area xy rectangle (size[0] >= size[1], yaw in (-pi/2, pi/2]),
        top from the points, bottom at the table when given (hidden sides are filled down to it). `box.pose`: 4x4."""
        from .perception import fit_box

        return fit_box(points, table)

    def fit(self, model_points: np.ndarray, observed: np.ndarray, init: np.ndarray | None = None) -> Fit:
        """Rigid ICP of a known shape (model points, its own frame) onto observed world points; partial views fit.
        Without `init` it tries 12 yaws about +z from the centroids. `fit.T` maps model to world."""
        from .perception import icp

        return icp(model_points, observed, init)

    # ------------------------------------------------------------------ goals
    def grasps(self, points: np.ndarray, table: Table, scene: np.ndarray | None = None,
               arm: str | None = None) -> list[Goal]:
        """Ranked grasp goals (best first) for a parallel hand from an object's points: top-down across the narrow
        side, across the long side if it fits, side grasps from the robot's side for tall objects. TCP depth is up
        to 3 cm below the top (half the height if shorter); width must fit the open hand; candidates whose open hand
        (at the goal or the 10 cm pre-pose) hits the table, the object or `scene` points are dropped. Each Goal has
        pre, hand_before=open, hand=closed. Empty: no grasp fits. arm: which arm (None: the only one)."""
        from .grasping import grasp_candidates
        from .perception import fit_box

        arm = self._name(arm)
        spec = self.spec.arm(arm)
        if spec.hand.kind != "parallel":
            raise ValueError(f"grasps: arm {arm} has a {spec.hand.kind} hand; use grasp_at")
        _, cols, _ = self._models()
        if arm not in self._hand_pts:
            self._hand_pts[arm] = cols[arm].hand_points_in_tcp(spec.hand.open)
        _, q, hand = self._state()
        R_now = self._arm(arm).fk(q[arm])[:3, :3]
        pts = np.asarray(points, float)
        cands = grasp_candidates(fit_box(pts, table), pts, table, self._hand_pts[arm], spec.hand.max_width,
                                 np.asarray(spec.base_in_world)[:2, 3], R_now, self._scene(scene, table, q, hand))
        return [Goal("grasp", arm, tcp=c.tcp, pre=c.pre, hand_before=spec.hand.open, hand=spec.hand.closed,
                     rank=c.score) for c in cands]

    def grasp_at(self, tcp: np.ndarray, hand: Sequence[float] | None = None, arm: str | None = None) -> Goal:
        """A grasp goal at a TCP pose you chose: pre-pose 10 cm back along the TCP's +x (approach), hand opens
        before moving and closes (or goes to `hand`) at `tcp`."""
        arm = self._name(arm)
        spec = self.spec.arm(arm)
        tcp = np.asarray(tcp, float)
        pre = tcp.copy()
        pre[:3, 3] -= tcp[:3, 0] * 0.10
        return Goal("grasp", arm, tcp=tcp, pre=pre, hand_before=spec.hand.open,
                    hand=tuple(hand) if hand is not None else spec.hand.closed)

    def hand_grasps(self, points: np.ndarray, table: Table, kind: str = "power", arm: str | None = None) -> list[Goal]:
        """Dexterous hand: top-down grasp goals, TCP at the object's mid height, closing axis across its box faces
        (four directions; first the one nearest the arm base frame's +x, e.g. a humanoid's forward: thumb on the
        near side, fingers on the far side).
        Opens to HandSpec.open on the way, closes to preset `kind`."""
        from .geometry import pose
        from .perception import fit_box

        arm = self._name(arm)
        h = self.spec.arm(arm).hand
        if h.kind != "dexterous":
            raise ValueError(f"hand_grasps: arm {arm} has a {h.kind} hand; use grasps")
        close = h.closed if kind == "closed" else h.presets.get(kind)
        if close is None:
            raise KeyError(f"no hand preset {kind!r}; presets: {sorted(h.presets)}")
        box = fit_box(np.asarray(points, float), table)
        bx = np.asarray(self.spec.arm(arm).base_in_world, float)[:3, 0]
        away = float(np.arctan2(bx[1], bx[0]))                          # the base frame's forward direction
        z = max(box.bottom + box.size[2] / 2, table.height + 0.02)
        down = np.array([0.0, 0.0, -1.0])
        out = []
        for k in range(4):
            psi = box.yaw + k * np.pi / 2
            width = box.size[1] if k % 2 == 0 else box.size[0]       # extent along the closing axis
            if h.max_width and width > h.max_width - 0.01:
                continue
            y = np.array([np.cos(psi), np.sin(psi), 0.0])
            R = np.stack([down, y, np.cross(down, y)], axis=1)
            T = pose(R, [box.center[0], box.center[1], z])
            pre = T.copy()
            pre[2, 3] += 0.10
            turn = abs((psi - away + np.pi) % (2 * np.pi) - np.pi)
            out.append(Goal("grasp", arm, tcp=T, pre=pre, hand_before=h.open, hand=tuple(close), rank=float(turn)))
        return sorted(out, key=lambda g: g.rank)

    def _dex(self, arm: str):
        """The arm's hand as a HandModel and its grasp planner (built once per arm)."""
        if arm not in self._dex_cache:
            from .dexgrasp import DexGraspPlanner, PlannerConfig
            from .handmodel import HandModel

            hm = HandModel.from_arm(self.spec.arm(arm))
            self._dex_cache[arm] = (hm, DexGraspPlanner(hm, PlannerConfig(seed=self.config.seed)))
        return self._dex_cache[arm]

    def dex_grasps(self, points: np.ndarray, table: Table, hand: str | None = None, scene: np.ndarray | None = None,
                   arm: str | None = None, mesh: tuple[np.ndarray, np.ndarray] | None = None,
                   top: int = 8) -> list[Goal]:
        """Multi-finger grasp goals (best first) for a dexterous hand from an object's points (or `mesh` = (V, F)):
        hand poses sampled around the object (approach directions times rolls, none from below the table), the open
        hand slid in until it touches, each finger closed until it touches, scored by the friction-cone wrench hull
        (force closure), the force needed to hold the object against gravity, penetration and fingers in contact,
        then refined by a short search over the pose. hand: closing style, "power" (all fingers wrap, palm may
        touch), "pinch" (the pinch preset) or None for both. Approaches from the arm's side and from above rank
        higher. Each Goal: pre 10 cm back along the approach with hand_before = the open pre-shape, tcp, hand = a
        squeeze command past the contact angles. Empty: no grasp found. arm: which arm (None: the only one)."""
        arm = self._name(arm)
        spec = self.spec.arm(arm)
        if spec.hand.kind != "dexterous":
            raise ValueError(f"dex_grasps: arm {arm} has a {spec.hand.kind} hand; use grasps")
        from .dexgrasp import ObjectGeom, to_goals

        _, planner = self._dex(arm)
        if mesh is not None:
            obj = ObjectGeom.from_mesh(*mesh)
        else:
            obj = ObjectGeom.from_points(np.asarray(points, float), table.height if table is not None else None)
        _, q, hnow = self._state()
        base = np.asarray(spec.base_in_world, float)[:3, 3]
        out = obj.centre[:2] - base[:2]
        out = np.r_[out / (np.linalg.norm(out) or 1.0), 0.0]
        prefer = {"approach": (np.array([0.0, 0.0, -1.0]) + 0.6 * out) / np.linalg.norm([0.6, 1.0]), "w_approach": 0.8}
        obstacles = self._scene(scene, table, q, hnow)
        if obstacles is not None and len(obstacles):
            from scipy.spatial import cKDTree

            d, _ = cKDTree(obj.points).query(obstacles, distance_upper_bound=0.01)
            obstacles = obstacles[~np.isfinite(d)]                 # the object's own points are not obstacles
        old_keep = planner.cfg.keep
        planner.cfg.keep = top
        try:
            found = planner.plan(obj, table.height if table is not None else None, style=hand, scene=obstacles,
                                 prefer=prefer)
        finally:
            planner.cfg.keep = old_keep
        return to_goals(found, arm)

    def retarget(self, keypoints: np.ndarray, arm: str | None = None, valid: np.ndarray | None = None,
                 side: str | None = None, scale: float | None = None) -> dict[str, Any]:
        """Human hand keypoints (T, 21, 3) in the MediaPipe layout (wrist 0, thumb 1-4, index 5-8, middle 9-12,
        ring 13-16, little 17-20; metres, world frame) -> this arm's hand: {"hand": (T, n) finger commands,
        "tcp": (T, 4, 4) TCP poses that put the robot palm where the human palm is, "ee": (T, 4, 4),
        "tip_error_mm": (T,) fingertip error per frame, "summary": {...}}. Fingertip and thumb-to-finger vectors
        are matched (scaled to the robot hand's size; `scale` overrides it) within joint limits, smooth across
        frames. side: "left"/"right" human hand (a mismatch mirrors it); valid: frames with keypoints."""
        arm = self._name(arm)
        if self.spec.arm(arm).hand.kind != "dexterous":
            raise ValueError(f"retarget: arm {arm} has a {self.spec.arm(arm).hand.kind} hand")
        from .retarget import Retargeter

        hm, _ = self._dex(arm)
        res = Retargeter(hm, scale=scale).run(np.asarray(keypoints, float), valid=valid, side=side)
        return {"hand": res.q, "tcp": res.tcp, "ee": res.ee, "tip_error_mm": res.tip_error_mm,
                "summary": res.summary()}

    def hand_points(self, hand: Sequence[float] | str | None = None, arm: str | None = None) -> np.ndarray:
        """Hand surface points (N, 3) in the TCP frame at command `hand` (a preset name; None: open). [:, 0].max():
        how far the hand reaches past the TCP along +x (e.g. to push a button with it)."""
        arm = self._name(arm)
        h = self.spec.arm(arm).hand
        cmd = h.open if hand is None else (h.presets[hand] if isinstance(hand, str) else hand)
        _, cols, _ = self._models()
        return cols[arm].hand_points_in_tcp(np.asarray(cmd, float))

    def place_at(self, obj_in_tcp: np.ndarray, obj_points: np.ndarray | Box, xy: Sequence[float],
                 yaw: float | None = None, table: Table | None = None, hover: float = 0.03,
                 scene: np.ndarray | None = None, support: float | None = None, arm: str | None = None) -> Goal:
        """Where the TCP must go so the held object rests at `xy`. obj_in_tcp: the object frame in the TCP frame at
        grasp time (inv(grasp_tcp) @ box.pose). obj_points: the object's Box (its pose is the object frame; best:
        the hidden bottom counts) or its points in that frame (apply inv(box.pose) to its world points). The object
        keeps its current tilt; its yaw becomes `yaw` (world, None: unchanged); its lowest point ends `hover`
        metres above the support: `support` (a height), else the top of `scene` points (e.g. view.points; robot,
        table and the held object ignored) under the object's footprint at `xy` (a plate, a block to stack on),
        else the table (`table`, else the robot's table height). Pre-pose 10 cm above; hand opens at the goal; the
        object rides along as `attached` geometry (checked against the table and observed points)."""
        from .collision import geom_surface_box

        arm = self._name(arm)
        spec = self.spec.arm(arm)
        _, q, hand = self._state()
        T_tcp_now = self._arm(arm).fk(q[arm])
        T_oit = np.asarray(obj_in_tcp, float)
        if isinstance(obj_points, Box):
            pts = geom_surface_box(np.zeros(3), np.asarray(obj_points.size, float) / 2, 0.01)
        else:
            pts = np.asarray(obj_points, float).reshape(-1, 3)
        R_obj = T_tcp_now[:3, :3] @ T_oit[:3, :3]
        if yaw is not None:
            R_obj = rotz(yaw - yaw_of(R_obj)) @ R_obj
        rotated = pts @ R_obj.T
        height = table.height if table is not None else (self.spec.table_height or 0.0)
        if support is None and scene is not None:
            under = self._scene(scene, table, q, hand)
            if under is not None and len(under):
                held = apply(T_tcp_now @ T_oit, pts)          # the object where it is now, and the hand above it
                near = cKDTree(held).query(under, distance_upper_bound=self.config.near_hand / 2)[0]
                under = under[~np.isfinite(near) & (np.linalg.norm(under - T_tcp_now[:3, 3], axis=1) >
                                                    self.config.near_hand)]
                c, s_ = np.cos(yaw_of(R_obj)), np.sin(yaw_of(R_obj))
                Rz = np.array([[c, s_], [-s_, c]])             # world xy -> object-yaw frame
                lo, hi = (rotated[:, :2] @ Rz.T).min(0) - 0.003, (rotated[:, :2] @ Rz.T).max(0) + 0.003
                rel = (under[:, :2] - np.asarray(xy, float)[:2]) @ Rz.T
                inside = under[np.all((rel >= lo) & (rel <= hi), axis=1)]
                if len(inside) >= 5:
                    support = max(height, float(np.percentile(inside[:, 2], 95)))
        if support is None:
            support = height
        low = float(rotated[:, 2].min())
        T_obj = np.eye(4)
        T_obj[:3, :3] = R_obj
        T_obj[:3, 3] = [xy[0], xy[1], support + hover - low]
        tcp = T_obj @ inv(T_oit)
        pre = tcp.copy()
        pre[2, 3] += 0.10
        return Goal("place", arm, tcp=tcp, pre=pre, hand=spec.hand.open, attached=apply(T_oit, pts))

    # ------------------------------------------------------------------ motion
    def plan(self, goals: Goal | Sequence[Goal], table: Table | None = None,
             scene: np.ndarray | None = None) -> Plan | None:
        """The first reachable goal by rank: IK (seeds: current joints, home, random), a collision-free joint path to
        `pre` (or `tcp`) avoiding the table and `scene` points (pass view.points: the robot, the table and points
        within 8 cm of the hand are ignored), the straight approach to `tcp`, then the final hand command. Timed to
        control steps within velocity and acceleration limits; other arms hold. None: no goal reachable
        (robot.why() says why). Deterministic for a given seed."""
        import time

        goals = [goals] if isinstance(goals, Goal) else sorted(goals, key=lambda g: g.rank)
        goals = [self._goal_arm(g) for g in goals]
        kin, _, planner = self._models()
        _, q, hand = self._state()
        keep = self._keep(hand)
        height = table.height if table is not None else self.spec.table_height
        reasons = []
        t0 = time.monotonic()
        for i, g in enumerate(goals):
            if g.arm not in kin.arms:
                raise KeyError(f"no arm {g.arm!r}; arms: {list(kin.arms)}")
            att = planner.plan_goal(g, q, hand, height, scene, seen_at=self._seen_at(scene))
            if att.path:
                actions, info = planner.actions(g, att.path, q, keep, self.spec.action_dim)
                info.update(att.info, goal_index=i, planner_time=time.monotonic() - t0)
                self._why = ""
                return Plan(goal=g, actions=actions, start_q={k: v.copy() for k, v in q.items()}, info=info,
                            path={g.arm: [np.asarray(p, float) for p in att.path]})
            reasons.append(f"goal {i} ({g.verb}): {att.why}")
        self._why = "; ".join(reasons) or "no goals"
        return None

    def why(self) -> str:
        """Why the last plan() returned None (empty after a success)."""
        return self._why

    def hold(self, steps: int, hand: dict[str, Sequence[float]] | None = None) -> np.ndarray:
        """(steps, action_dim) joint targets that keep every arm where it is now; `hand` {arm: command} sets those
        hands (others keep their last command). Run it with run_actions (a gripper command, a pause)."""
        _, q, h = self._state()
        keep = self._keep(h)
        row = np.zeros(self.spec.action_dim)
        for a in self.spec.arms:
            row[a.action_slice[0]:a.action_slice[1]] = q[a.name]
            cmd = np.asarray((hand or {}).get(a.name, keep[a.name]), float)
            row[a.hand_slice[0]:a.hand_slice[1]] = cmd
        return np.repeat(row[None], max(1, int(steps)), axis=0)

    def run_actions(self, actions: np.ndarray) -> dict[str, Any]:
        """Execute raw joint targets (from hold) through the kernel and the shield; remembers the hand commands."""
        res = dict(self.executor.run(np.asarray(actions, float)))
        last = np.asarray(actions, float)[-1]
        for a in self.spec.arms:
            self._hand_cmd[a.name] = np.array(last[a.hand_slice[0]:a.hand_slice[1]], float)
        _, q, hand = self._state()
        res.update(q=q, hand=hand, tcp={a: self._arm(a).fk(q[a]) for a in q})
        return res

    def reach(self, tcp: np.ndarray, arm: str | None = None, free: str | None = None) -> dict[str, Any]:
        """Can the arm put its TCP at `tcp` (4x4)? IK only (no collision check, no path): {"ok", "pos_err" (m),
        "rot_err" (rad), "q"}."""
        arm = self._name(arm)
        _, q, _ = self._state()
        r = self._arm(arm).ik(np.asarray(tcp, float), q[arm], free=free,
                              rng=np.random.default_rng(self.config.seed))
        return {"ok": bool(r.ok), "pos_err": float(r.pos_err), "rot_err": float(r.rot_err), "q": r.q}

    def together(self, plans: dict[str, Plan] | Sequence[Plan], sync: bool = False) -> Plan | None:
        """One plan moving several arms at once: each arm follows its own plan from the same moment (a shorter one
        then holds its last command). sync=True re-times the arms to start and stop together with their TCPs in step
        (both hands on one object: plan each arm's motion, e.g. two "lift" goals, then together(..., sync=True)).
        Every step is checked for arms touching each other or the robot; None when they would (robot.why())."""
        from .planning import _limits, retime_together

        items = list(plans.values()) if isinstance(plans, dict) else list(plans)
        if not items or any(p is None for p in items):
            self._why = "together: every plan must be a Plan (one of them is None)"
            return None
        arms = [p.goal.arm for p in items]
        if len(set(arms)) != len(arms):
            self._why = f"together: one plan per arm (got arms {arms})"
            return None
        kin, _, planner = self._models()
        _, q, hand = self._state()
        keep = self._keep(hand)
        cols: dict[str, np.ndarray] = {}
        hands: dict[str, np.ndarray] = {}
        dt = 1.0 / self.control_hz
        if sync:
            paths = {p.goal.arm: p.path.get(p.goal.arm) or [np.asarray(p.start_q[p.goal.arm])] for p in items}
            n = {a: len(kin[a].home) for a in paths}
            vmax = {a: _limits(self.config.max_vel, n[a]) for a in paths}
            amax = {a: _limits(self.config.max_acc, n[a]) for a in paths}
            trajs = retime_together(paths, kin, vmax, amax, dt)
        for p in items:
            a = p.goal.arm
            spec = self.spec.arm(a)
            rows = np.asarray(p.actions, float)
            moving = int(p.info.get("moving_steps", len(rows)))
            arm_rows = rows[:, spec.action_slice[0]:spec.action_slice[1]]
            hand_rows = rows[:, spec.hand_slice[0]:spec.hand_slice[1]]
            if sync:
                tr = trajs[a]
                hold = len(rows) - moving
                arm_rows = np.vstack([tr, np.repeat(tr[-1:], hold, axis=0)])
                hand_rows = np.vstack([np.repeat(hand_rows[:1], len(tr), axis=0), hand_rows[moving:]])
            cols[a], hands[a] = arm_rows, hand_rows
        T = max(len(v) for v in cols.values())
        actions = np.zeros((T, self.spec.action_dim))
        for spec in self.spec.arms:
            a = spec.name
            if a in cols:
                pad = T - len(cols[a])
                actions[:, spec.action_slice[0]:spec.action_slice[1]] = np.vstack([cols[a], np.repeat(cols[a][-1:], pad, 0)])
                actions[:, spec.hand_slice[0]:spec.hand_slice[1]] = np.vstack([hands[a], np.repeat(hands[a][-1:], pad, 0)])
            else:
                actions[:, spec.action_slice[0]:spec.action_slice[1]] = q[a]
                actions[:, spec.hand_slice[0]:spec.hand_slice[1]] = keep[a]
        if len(self.spec.arms) > 1:
            for k in range(0, T, 2):
                qk = {s.name: actions[k, s.action_slice[0]:s.action_slice[1]] for s in self.spec.arms}
                hk = {s.name: actions[k, s.hand_slice[0]:s.hand_slice[1]] for s in self.spec.arms}
                hit = planner.arms_clear(qk, hk)
                if hit:
                    self._why = f"together: at step {k} ({k * dt:.2f} s) {hit}"
                    return None
        start = {a: np.asarray(v, float).copy() for a, v in q.items()}
        info = {"arms": arms, "goals": [p.goal for p in items], "sync": bool(sync), "duration_s": T * dt,
                "moving_steps": T}
        self._why = ""
        return Plan(goal=items[0].goal, actions=actions, start_q=start, info=info,
                    path={a: v for p in items for a, v in p.path.items()})

    def run(self, plan: Plan | dict[str, Plan] | Sequence[Plan], sync: bool = False) -> dict[str, Any]:
        """Execute a plan (through the kernel and the safety shield) and wait. Returns {"steps", "ended",
        "success", "reason", "q", "hand", "tcp": {arm: 4x4}, "tcp_error": {"pos": m, "rot": rad} vs the goal's
        tcp, None for a joints goal}. Refuses ("error", steps 0) if the robot moved more than 0.05 rad since
        planning: plan again. Several arms at once: run({"left": plan_l, "right": plan_r}, sync=...) runs
        together(...) first (refused with "error" when the arms would touch); tcp_error is then {arm: ...}."""
        if not isinstance(plan, Plan):
            merged = self.together(plan, sync=sync)
            if merged is None:
                st = self.executor.status()
                return {"steps": 0, "ended": bool(st.get("ended")), "success": None, "reason": None,
                        "tcp_error": None, "error": self._why}
            plan = merged
        _, q, _ = self._state()
        dev = max(float(np.max(np.abs(q[a] - np.asarray(plan.start_q[a])))) for a in plan.start_q)
        if dev > 0.05:
            st = self.executor.status()
            return {"steps": 0, "ended": bool(st.get("ended")), "success": None, "reason": None, "tcp_error": None,
                    "error": f"robot moved {dev:.3f} rad since planning; plan again"}
        res = dict(self.executor.run(np.asarray(plan.actions, float)))
        last = plan.actions[-1]
        for a in self.spec.arms:
            self._hand_cmd[a.name] = np.array(last[a.hand_slice[0]:a.hand_slice[1]], float)
        _, q, hand = self._state()
        tcp = {a: self._arm(a).fk(q[a]) for a in q}
        res.update(q=q, hand=hand, tcp=tcp, tcp_error=None)   # None: the goal had no tcp (e.g. a joints goal)
        goals = plan.info.get("goals")
        if goals:
            res["tcp_error"] = {g.arm: _tcp_error(tcp[g.arm], g) for g in goals}
        else:
            res["tcp_error"] = _tcp_error(tcp[plan.goal.arm], plan.goal)
        return res

    # ------------------------------------------------------------------ checks
    def holding(self, arm: str | None = None) -> bool | None:
        """Parallel hand: True when the measured pads stopped between open and closed after a close command
        (something is between them); False when open, opening or fully closed. Dexterous hand: True when, after a
        command nearer closed than open, at least two finger joints stopped more than 0.12 rad short of it."""
        arm = self._name(arm)
        spec = self.spec.arm(arm)
        if spec.hand.kind != "parallel":
            cmd = self._hand_cmd.get(arm)
            if cmd is None:
                return False
            op, cl = np.asarray(spec.hand.open, float), np.asarray(spec.hand.closed, float)
            if np.linalg.norm(cmd - cl) >= np.linalg.norm(cmd - op):
                return False
            _, _, hand = self._state()
            short = np.abs(np.asarray(hand[arm], float) - cmd) > 0.12
            return bool(short.sum() >= 2)
        cmd = self._hand_cmd.get(arm)
        if cmd is not None and self._hand_frac(arm, cmd) > 0.5:
            return False
        _, _, hand = self._state()
        f = self._hand_frac(arm, hand[arm])
        return bool(0.03 < f < 0.97)

    def carried(self, view: View, points: np.ndarray, tcp_before: np.ndarray, tcp_now: np.ndarray,
                table: Table) -> dict[str, Any]:
        """Did the object move with the hand? `points`: its world points seen before the motion; `view`: a look()
        after it. Returns {"carried": bool, "match": fraction of rigidly-moved points seen in the view, "left_behind":
        fraction still seen at the old place, "lifted": TCP z change (m), "offset": (3,) mean observed - expected}."""
        from .perception import above_table

        pts = np.asarray(points, float)
        expected = apply(np.asarray(tcp_now) @ inv(np.asarray(tcp_before)), pts)
        seen = above_table(view.points, table, 0.008)
        if len(seen) == 0:
            return {"carried": False, "match": 0.0, "left_behind": 0.0,
                    "lifted": float(tcp_now[2, 3] - tcp_before[2, 3]), "offset": np.zeros(3)}
        tree = cKDTree(seen)
        d, j = tree.query(expected)
        match = float(np.mean(d < 0.01))
        moved_away = np.linalg.norm(expected - pts, axis=1) > 0.02
        old = pts[moved_away]
        left = float(np.mean(tree.query(old)[0] < 0.01)) if len(old) else 1.0
        near = d < 0.03
        offset = (seen[j[near]] - expected[near]).mean(0) if near.any() else np.zeros(3)
        return {"carried": bool(match >= 0.5 and left < 0.3), "match": match, "left_behind": left,
                "lifted": float(tcp_now[2, 3] - tcp_before[2, 3]), "offset": offset}

    def show(self, image: View | np.ndarray, note: str = "") -> None:
        """Attach an image (an RGB array, or a View: its cameras side by side) to your next turn, with a note."""
        if isinstance(image, View):
            imgs = [c.rgb for c in image.cams.values()]
            if not imgs:
                return
            h = max(i.shape[0] for i in imgs)
            image = np.concatenate([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs], axis=1)
        if self._show is not None:
            self._show(np.asarray(image), note)


def _arr(x) -> np.ndarray | None:
    return None if x is None else np.asarray(x)


def _tcp_error(T: np.ndarray, g: Goal) -> dict[str, float] | None:
    if g.tcp is None:
        return None
    G = np.asarray(g.tcp, float)
    if g.free:
        k = {"x": 0, "z": 2}[g.free]
        rot = float(np.arccos(np.clip(T[:3, k] @ G[:3, k], -1, 1)))
    else:
        rot = angle(T[:3, :3].T @ G[:3, :3])
    return {"pos": float(np.linalg.norm(T[:3, 3] - G[:3, 3])), "rot": rot}


def _arm_first(fn):
    """Accept the older calling form with the arm name first (grasps("arm", points, table, ...))."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        if args and isinstance(args[0], str):
            if "arm" in kw:
                raise TypeError(f"{fn.__name__}: arm given twice")
            kw["arm"], args = args[0], args[1:]
        return fn(self, *args, **kw)

    return wrapper


for _verb in ("grasps", "grasp_at", "place_at"):            # (the verbs added later take arm= only)
    setattr(Tools, _verb, _arm_first(getattr(Tools, _verb)))


VERBS = ("status", "look", "find", "table", "objects", "box", "fit", "grasps", "hand_grasps", "dex_grasps", "grasp_at",
         "place_at", "hand_points", "retarget", "Goal", "plan", "why", "run", "together", "holding", "carried", "show")
TYPES = (View, CameraView, Mask, Table, Detection, Box, Fit, Plan)
DEXTEROUS_ONLY = ("dex_grasps", "retarget")


def verbs_for(tools: Tools | None = None) -> tuple[str, ...]:
    """The verbs the reference lists for a robot: the multi-finger verbs only when one of its hands is dexterous
    (all verbs without a robot)."""
    if tools is None or any(a.hand.kind == "dexterous" for a in tools.spec.arms):
        return VERBS
    return tuple(v for v in VERBS if v not in DEXTEROUS_ONLY)


def api_reference(tools: Tools | None = None) -> str:
    """The toolkit's API reference for a code-writing model: the robot (from the spec), each verb's signature and
    docstring, and the data types."""
    lines = ["# robot: the manipulation toolkit", ""]
    if tools is not None:
        s = tools.spec
        for a in s.arms:
            h = a.hand
            lines.append(f"arm {a.name!r}: {len(a.joints)} joints, {h.kind} hand (open {list(h.open)}, closed "
                         f"{list(h.closed)}" + (f", max width {h.max_width:.3f} m" if h.max_width else "") + ")")
        if s.cameras:
            lines.append("cameras: " + ", ".join(c.name + (f" (on {c.mounted_on})" if c.mounted_on else "")
                                                 for c in s.cameras))
        if s.table_height is not None:
            lines.append(f"table height about {s.table_height:.3f} m")
        if s.notes:
            lines.append(s.notes.strip())
        lines.append("")
    lines += ["Frames: world, metres, radians, +z up; poses are 4x4 numpy arrays. TCP: origin between the pads (a "
              "dexterous hand: among the fingers), +x approach, +y closing axis (thumb to fingers), +z = x cross y.", ""]
    several = tools is not None and len(tools.spec.arms) > 1
    if several:
        lines += ["Several arms: pass arm= to grasps, grasp_at, place_at and holding, and set Goal.arm. plan() moves one "
                  "arm while the others hold still (they are obstacles); to move arms at the same time plan each, then "
                  "robot.run({arm: plan, ...}) (sync=True when both hold one object). Arms are checked against each "
                  "other at every step.", ""]
    for name in verbs_for(tools):
        obj = Goal if name == "Goal" else getattr(Tools, name)
        lines += [f"robot.{name}{_signature(obj, name != 'Goal')}", _indent(inspect.getdoc(obj) or ""), ""]
    lines += ["## Example", _indent(TWO_ARM_EXAMPLE if several else EXAMPLE), ""]
    lines.append("## Types")
    for t in TYPES:
        lines += [f"{t.__name__}: {inspect.getdoc(t)}", ""]
    return "\n".join(lines).rstrip() + "\n"


def _signature(obj, returns: bool = True) -> str:
    sig = inspect.signature(obj)
    parts = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        txt = p.name
        if p.annotation is not inspect.Parameter.empty:
            txt += f": {p.annotation if isinstance(p.annotation, str) else inspect.formatannotation(p.annotation)}"
        if p.default is not inspect.Parameter.empty:
            txt += f" = {p.default!r}"
        parts.append(txt)
    ret = sig.return_annotation
    tail = f" -> {ret}" if returns and ret is not inspect.Signature.empty else ""
    return f"({', '.join(parts)}){tail}"


EXAMPLE = """view = robot.look(); table = robot.table(view)
obj = robot.objects(view, robot.find(view, "red block"), table)[0]
box = robot.box(obj.points, table)
plan = robot.plan(robot.grasps(obj.points, table, scene=view.points), table, view.points)
if plan is None: print(robot.why())
else: out = robot.run(plan); print(out["tcp_error"], robot.holding())
up = out["tcp"]["arm"].copy(); up[2, 3] += 0.10
robot.run(robot.plan(robot.Goal("lift", tcp=up), table, view.points))
obj_in_tcp = np.linalg.inv(plan.goal.tcp) @ box.pose
view = robot.look()                                # rests on whatever is under (0.5, 0.2): table, plate, block
goal = robot.place_at(obj_in_tcp, box, xy=(0.5, 0.2), yaw=0.0, table=table, hover=0.01, scene=view.points)
robot.run(robot.plan(goal, table, view.points))"""

TWO_ARM_EXAMPLE = """view = robot.look(); table = robot.table(view)
a = robot.objects(view, robot.find(view, "red block"), table)[0]
b = robot.objects(view, robot.find(view, "blue block"), table)[0]
pl = robot.plan(robot.grasps(a.points, table, scene=view.points, arm="left"), table, view.points)
pr = robot.plan(robot.grasps(b.points, table, scene=view.points, arm="right"), table, view.points)
out = robot.run({"left": pl, "right": pr})         # both arms at once; out["error"] if they would touch
print(out.get("error"), robot.holding(arm="left"), robot.holding(arm="right"))
st = robot.status(); ups = {}
for arm in ("left", "right"):
    t = st["tcp"][arm].copy(); t[2, 3] += 0.10
    ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t), table)
robot.run(ups, sync=True)                          # start and stop together, TCPs in step"""


def _indent(text: str) -> str:
    return "\n".join("    " + ln for ln in text.splitlines())


__all__ = ["Box", "CameraView", "Detection", "Fit", "Goal", "Mask", "Plan", "Table", "Tools", "Verb", "View",
           "api_reference", "verbs_for"]
