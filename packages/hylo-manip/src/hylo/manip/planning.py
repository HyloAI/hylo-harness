"""Motion planning for goals: IK over seeds, a collision-free joint path (straight interpolation, else RRT-Connect with
shortcut smoothing), a straight Cartesian final approach, hand commands, and time parameterisation to control steps
under per-joint velocity and acceleration limits."""

from __future__ import annotations

import time
from itertools import islice
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .collision import Collider, Obstacles
from .geometry import angle, apply
from .kinematics import ArmModel, Kinematics

LINE_VERBS = ("lift", "retreat", "push", "press")


@dataclass
class PlannerConfig:
    control_hz: float = 20.0
    max_vel: Any = 1.0               # rad/s (or m/s for slide joints), scalar or per joint
    max_acc: Any = 2.0               # rad/s^2
    hand_time: float = 1.0           # seconds the final hand command is held
    margin: float = 0.01             # clearance to observed points
    table_clearance: float = 0.01    # clearance to the table plane while moving freely
    approach_clearance: float = 0.002  # clearance to the table on the straight final approach
    self_filter: float = 0.03        # observed points this close to the robot are the robot
    near_hand: float = 0.08          # observed points this close to the start TCP are ignored (a held object)
    line_step: float = 0.01          # Cartesian line resolution (m)
    line_jump: float = 0.25          # max joint change between line samples (rad)
    edge_res: float = 0.03           # joint-space collision check resolution (rad)
    rrt_step: float = 0.2
    rrt_iters: int = 3000
    rrt_time: float = 10.0
    shortcut_iters: int = 60
    ik_random: int = 12
    ik_random_more: int = 48         # more random seeds when the first ones find no solution (narrow workspaces)
    ik_solutions: int = 4
    seed: int = 0


@dataclass
class Attempt:
    path: list[np.ndarray] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    why: str = ""


def _limits(value, n: int) -> np.ndarray:
    v = np.asarray(value, float)
    return np.full(n, float(v)) if v.ndim == 0 else v.reshape(n)


def time_parameterize(path: Sequence[np.ndarray], vmax: np.ndarray, amax: np.ndarray, dt: float) -> np.ndarray:
    """Positions at t = dt, 2 dt, ... along a piecewise-linear joint path, starting and ending at rest, whose finite
    differences stay within `vmax` and `amax` (the profile is slowed down until they do)."""
    vmax, amax = np.asarray(vmax, float), np.asarray(amax, float)
    v, a = vmax, amax
    for _ in range(20):
        traj = _profile(path, v, a, dt)
        rv, ra = check_limits(traj, np.asarray(path[0], float), vmax, amax, dt)
        if rv <= 1.0 + 1e-6 and ra <= 1.0 + 1e-6:
            break
        v, a = v * 0.95, a * 0.85
    return traj


def _profile(path: Sequence[np.ndarray], vmax: np.ndarray, amax: np.ndarray, dt: float) -> np.ndarray:
    """Trapezoidal speed along each segment; at a corner the speed is limited so the velocity change per step
    stays within half of `amax * dt`."""
    P = [np.asarray(path[0], float)]
    for q in path[1:]:
        if np.max(np.abs(np.asarray(q) - P[-1])) > 1e-9:
            P.append(np.asarray(q, float))
    if len(P) < 2:
        return P[0][None].copy()
    P = np.array(P)
    D = np.diff(P, axis=0)
    L = np.linalg.norm(D, axis=1)
    U = D / L[:, None]
    with np.errstate(divide="ignore"):
        V = np.min(vmax / np.abs(U), axis=1)
        A = np.min(amax / np.abs(U), axis=1)
    k = len(L)
    vj = np.zeros(k + 1)                      # speed at the junctions
    for j in range(1, k):
        du = np.abs(U[j] - U[j - 1])
        with np.errstate(divide="ignore"):
            vj[j] = min(V[j - 1], V[j], np.min(0.5 * amax * dt / np.maximum(du, 1e-12)))
    for j in range(k - 1, -1, -1):           # backward: able to slow down in time
        vj[j] = min(vj[j], np.sqrt(vj[j + 1] ** 2 + 2 * A[j] * L[j]))
    for j in range(k):                        # forward: able to speed up in time
        vj[j + 1] = min(vj[j + 1], np.sqrt(vj[j] ** 2 + 2 * A[j] * L[j]))
    segs, t0 = [], 0.0
    for j in range(k):
        v0, v1, a, ln = vj[j], vj[j + 1], A[j], L[j]
        vp = min(V[j], np.sqrt(max(a * ln + 0.5 * (v0 ** 2 + v1 ** 2), 0.0)))
        vp = max(vp, v0, v1, 1e-9)
        ta, td = (vp - v0) / a, (vp - v1) / a
        da, dd = (v0 + vp) / 2 * ta, (vp + v1) / 2 * td
        tc = max(ln - da - dd, 0.0) / vp
        segs.append((t0, v0, vp, v1, a, ta, tc, td, da, ln))
        t0 += ta + tc + td
    times = np.arange(1, int(np.ceil(t0 / dt - 1e-9)) + 1) * dt
    out = np.empty((len(times), P.shape[1]))
    j = 0
    for i, t in enumerate(times):
        while j < k - 1 and t > segs[j][0] + sum(segs[j][5:8]):
            j += 1
        s0, v0, vp, v1, a, ta, tc, td, da, ln = segs[j]
        tau = min(t - s0, ta + tc + td)
        if tau <= ta:
            s = v0 * tau + 0.5 * a * tau ** 2
        elif tau <= ta + tc:
            s = da + vp * (tau - ta)
        else:
            r = tau - ta - tc
            s = da + vp * tc + vp * r - 0.5 * a * r ** 2
        out[i] = P[j] + U[j] * min(max(s, 0.0), ln)
    out[-1] = P[-1]
    return out


def check_limits(traj: np.ndarray, q0: np.ndarray, vmax: np.ndarray, amax: np.ndarray,
                 dt: float) -> tuple[float, float]:
    """Largest ratio of finite-difference velocity and acceleration to the limits (starting and ending at rest)."""
    X = np.vstack([q0, traj, traj[-1]])
    v = np.diff(X, axis=0) / dt
    a = np.diff(np.vstack([np.zeros(X.shape[1]), v]), axis=0) / dt
    return float(np.max(np.abs(v) / vmax)), float(np.max(np.abs(a) / amax))


def retime_together(paths: dict[str, Sequence[np.ndarray]], kin: Kinematics, vmax: dict[str, np.ndarray],
                    amax: dict[str, np.ndarray], dt: float, min_steps: int = 2) -> dict[str, np.ndarray]:
    """Joint trajectories of several arms that start and stop together and keep their TCPs in step: each path is
    parameterised by its TCP travel (plus a little joint travel, for turns on the spot), and every arm follows one
    shared minimum-jerk progress profile, lengthened until every arm keeps its velocity and acceleration limits."""
    prog = {}
    for name, path in paths.items():
        P = np.asarray(path, float)
        arm = kin[name]
        tcp = np.array([arm.fk(q)[:3, 3] for q in P])
        step = np.linalg.norm(np.diff(tcp, axis=0), axis=1) + 0.05 * np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(step)]
        prog[name] = (s / s[-1] if s[-1] > 1e-12 else np.linspace(0.0, 1.0, len(P)), P)
    n = max(min_steps, 2)
    for name, path in paths.items():
        n = max(n, len(time_parameterize(list(path), vmax[name], amax[name], dt)))
    for _ in range(40):
        t = np.arange(1, n + 1) / n
        sigma = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
        out, ok = {}, True
        for name, (s, P) in prog.items():
            traj = np.stack([np.interp(sigma, s, P[:, j]) for j in range(P.shape[1])], axis=1)
            rv, ra = check_limits(traj, P[0], vmax[name], amax[name], dt)
            ok &= rv <= 1.0 + 1e-6 and ra <= 1.0 + 1e-6
            out[name] = traj
        if ok:
            return out
        n = int(np.ceil(n * 1.1))
    return out


class Planner:
    def __init__(self, kin: Kinematics, colliders: dict[str, Collider], cfg: PlannerConfig) -> None:
        self.kin, self.col, self.cfg = kin, colliders, cfg

    # ------------------------------------------------------------------ obstacles
    def robot_points(self, q: dict[str, np.ndarray], hand: dict[str, Any]) -> dict[str, np.ndarray]:
        """What cameras see of each robot model (every geom of its bodies, fixed base included) at the given joints,
        per arm; arms sharing a model are posed together and reported once."""
        out, done = {}, set()
        for name, c in self.col.items():
            c.arm.set_q(q[name])
            c.arm.set_hand(hand.get(name))
        for name, c in self.col.items():
            if id(c.arm.m) in done:
                continue
            done.add(id(c.arm.m))
            c.arm.forward()
            out[name] = c.seen_world()
        return out

    def obstacles(self, arm: str, q_now: dict[str, np.ndarray], hand_now: dict[str, Any], table: float | None,
                  scene: np.ndarray | None, seen_at: Sequence[tuple[dict, dict]] = (),
                  attached: np.ndarray | None = None) -> Obstacles:
        """Observed points minus the table, the robot (now and at `seen_at`, the configurations the points were
        observed in) and what the hand holds (points near the TCP; with `attached`, points held in the TCP frame,
        within reach of those points where they are now); arms of other models added as points."""
        cfg = self.cfg
        pts = None if scene is None else np.asarray(scene, float).reshape(-1, 3)
        now = self.robot_points(q_now, hand_now)
        robot = list(now.values()) + [p for q, h in seen_at for p in self.robot_points(q, h).values()]
        me = self.kin[arm]
        me.set_q(q_now[arm])
        me.forward()
        T_tcp = me.tcp()
        tcp = T_tcp[:3, 3]
        held = None
        if attached is not None and len(attached):
            held = apply(T_tcp, attached)
        if pts is not None and len(pts):
            keep = np.ones(len(pts), bool)
            if table is not None:
                keep &= pts[:, 2] > table + 0.01
            R = np.concatenate(robot)
            if len(R):
                d, _ = cKDTree(R).query(pts, distance_upper_bound=cfg.self_filter)
                keep &= ~np.isfinite(d)
            keep &= np.linalg.norm(pts - tcp, axis=1) > cfg.near_hand
            if held is not None:                       # the held object as seen now (it may sit a little off)
                d, _ = cKDTree(held).query(pts, distance_upper_bound=cfg.near_hand / 2)
                keep &= ~np.isfinite(d)
            pts = pts[keep]
        others = [r for name, r in now.items() if name != arm and self.kin[name].m is not me.m]
        if others:
            pts = np.concatenate([p for p in [pts, *others] if p is not None])
        return Obstacles(table=table, clearance=cfg.table_clearance, points=pts, margin=cfg.margin)

    def arms_clear(self, q: dict[str, np.ndarray], hand: dict[str, Any]) -> str | None:
        """None when the arms at joints `q` touch neither each other nor the rest of the robot, else a description.
        Arms in one model: MuJoCo contacts among robot geoms (pairs touching at home and pairs within one hand
        excepted); arms in different models: each arm's collision samples against the others' surfaces."""
        for name, c in self.col.items():
            c.arm.set_q(q[name])
            c.arm.set_hand(hand.get(name))
        done: set[int] = set()
        for name, c in self.col.items():
            if id(c.arm.m) in done:
                continue
            done.add(id(c.arm.m))
            c.arm.forward()
            pairs = set()
            for other in self.col.values():
                if other.arm.m is c.arm.m:
                    pairs.update(other._contacts())
            if pairs:
                m = c.arm.m
                g1, g2 = sorted(pairs)[0]
                return f"{_body_name(m, g1)} touches {_body_name(m, g2)}"
        names = list(self.col)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ca, cb = self.col[a], self.col[b]
                if ca.arm.m is cb.arm.m:
                    continue
                ca.arm.forward()
                cb.arm.forward()
                obs = Obstacles(points=cb.seen_world(), margin=self.cfg.margin / 2)
                if obs.hits(ca.points_world()):
                    return f"arm {a} touches arm {b}"
        return None

    # ------------------------------------------------------------------ paths
    def edge_free(self, c: Collider, a, b, obs: Obstacles) -> bool:
        n = int(np.ceil(np.max(np.abs(b - a)) / self.cfg.edge_res))
        for i in range(1, n + 1):
            if c.check(a + (b - a) * (i / n), obs):
                return False
        return True

    def connect(self, c: Collider, q0, q1, obs: Obstacles, rng) -> tuple[list[np.ndarray] | None, str]:
        if self.edge_free(c, q0, q1, obs):
            return [q0, q1], "direct"
        path = self.rrt_connect(c, q0, q1, obs, rng)
        if path is None:
            return None, "rrt failed"
        return self.shortcut(c, path, obs, rng), "rrt"

    def rrt_connect(self, c: Collider, q0, q1, obs: Obstacles, rng) -> list[np.ndarray] | None:
        cfg, arm = self.cfg, c.arm
        cap = cfg.rrt_iters * 4 + 2
        trees = []
        for q in (q0, q1):
            nodes = np.empty((cap, arm.n))
            nodes[0] = q
            trees.append({"nodes": nodes, "parent": [-1], "n": 1})
        t_start = time.monotonic()

        def extend(tree, target):
            nodes, n = tree["nodes"], tree["n"]
            i = int(np.argmin(np.sum((nodes[:n] - target) ** 2, axis=1)))
            d = target - nodes[i]
            dist = np.linalg.norm(d)
            reached = dist <= cfg.rrt_step
            new = target if reached else nodes[i] + d * (cfg.rrt_step / dist)
            if tree["n"] >= cap or not self.edge_free(c, nodes[i], new, obs):
                return None, False
            nodes[n] = new
            tree["parent"].append(i)
            tree["n"] = n + 1
            return n, reached

        def branch(tree, i):
            out = []
            while i >= 0:
                out.append(tree["nodes"][i].copy())
                i = tree["parent"][i]
            return out

        a, b = 0, 1
        for _ in range(cfg.rrt_iters):
            if time.monotonic() - t_start > cfg.rrt_time:
                break
            target = rng.uniform(arm.lower, arm.upper)
            ia, _ = extend(trees[a], target)
            if ia is None:
                a, b = b, a
                continue
            qa = trees[a]["nodes"][ia]
            while True:                       # greedy connect of the other tree
                ib, reached = extend(trees[b], qa)
                if ib is None:
                    break
                if reached:
                    pa, pb = branch(trees[a], ia), branch(trees[b], ib)
                    path = pa[::-1] + pb[1:]
                    return path if a == 0 else path[::-1]
            a, b = b, a
        return None

    def shortcut(self, c: Collider, path: list[np.ndarray], obs: Obstacles, rng) -> list[np.ndarray]:
        out = [path[0]]                      # greedy: from each kept node, jump to the farthest visible one
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.edge_free(c, path[i], path[j], obs):
                j -= 1
            out.append(path[j])
            i = j
        for _ in range(self.cfg.shortcut_iters):
            if len(out) < 3:
                break
            i, j = sorted(rng.choice(len(out), 2, replace=False))
            if j - i > 1 and self.edge_free(c, out[i], out[j], obs):
                out = out[:i + 1] + out[j:]
        return out

    def line(self, c: Collider, q_start, T0: np.ndarray, T1: np.ndarray, obs: Obstacles) -> list[np.ndarray] | None:
        """Joint path following the straight TCP line T0 -> T1 (position lerp, rotation slerp) from q_start. A step
        whose IK solution jumps more than `line_jump` is subdivided (up to 16 times finer) before giving up: near
        joint limits and singular postures the joints move fast per centimetre of TCP travel."""
        cfg, arm = self.cfg, c.arm
        dist = np.linalg.norm(T1[:3, 3] - T0[:3, 3])
        n = max(2, int(np.ceil(max(dist / cfg.line_step, angle(T0[:3, :3].T @ T1[:3, :3]) / 0.05))))
        slerp = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))

        def pose_at(s: float) -> np.ndarray:
            T = np.eye(4)
            T[:3, :3] = slerp([s]).as_matrix()[0]
            T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
            return T

        def step(q: np.ndarray, s0: float, s1: float, depth: int) -> list[np.ndarray] | None:
            r = arm.solve(pose_at(s1), q, iters=60)
            if not r.ok:
                return None
            if np.max(np.abs(r.q - q)) > cfg.line_jump:
                if depth >= 4:
                    return None
                mid = (s0 + s1) / 2
                a = step(q, s0, mid, depth + 1)
                b = step(a[-1], mid, s1, depth + 1) if a else None
                return a + b if b else None
            if c.check(r.q, obs):
                return None
            return [r.q]

        q, out = np.asarray(q_start, float), [np.asarray(q_start, float)]
        grid = np.linspace(0, 1, n + 1)
        for s0, s1 in zip(grid[:-1], grid[1:]):
            seg = step(q, s0, s1, 0)
            if seg is None:
                return None
            out += seg
            q = seg[-1]
        return out

    # ------------------------------------------------------------------ goals
    def plan_goal(self, goal, q_now: dict[str, np.ndarray], hand_now: dict[str, Any], table: float | None,
                  scene: np.ndarray | None, attached: np.ndarray | None = None,
                  seen_at: Sequence[tuple[dict, dict]] = ()) -> Attempt:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        arm: ArmModel = self.kin[goal.arm]
        c = self.col[goal.arm]
        q0 = np.asarray(q_now[goal.arm], float)
        hand_move = goal.hand_before if goal.hand_before is not None else hand_now.get(goal.arm)
        if attached is None and getattr(goal, "attached", None) is not None:
            attached = np.asarray(goal.attached, float).reshape(-1, 3)
        obs = self.obstacles(goal.arm, q_now, hand_now, table, scene, seen_at, attached)
        if attached is not None and len(attached):
            obs = obs.relaxed(attached=attached)
        for name, other in self.col.items():          # other arms of the same model hold still
            if name != goal.arm:
                other.arm.set_q(q_now[name])
        arm.set_hand(hand_move)
        start_hit = c.check(q0, obs, attached)
        if start_hit == "self":
            return Attempt(why="start configuration is in self-collision")
        if start_hit == "table":                       # start below the clearance: allow that height
            arm.set_q(q0)
            arm.forward()
            low_pts = c.points_world()
            if obs.attached is not None and len(obs.attached):
                low_pts = np.vstack([low_pts, apply(arm.tcp(), obs.attached)])
            low = low_pts[:, 2].min() - table - 0.002
            obs = obs.relaxed(clearance=min(obs.clearance, low))
        elif start_hit == "points":
            obs = obs.relaxed(points=None)
        near = obs.relaxed(points=None, clearance=min(cfg.approach_clearance, obs.clearance),
                           soft_fingers=arm.spec.hand.kind == "dexterous")   # fingers give way on the last cm

        if goal.joints is not None:
            qg = np.clip(np.asarray(goal.joints, float), arm.lower, arm.upper)
            if c.check(qg, obs, attached):
                return Attempt(why=f"joint goal is in collision ({c.check(qg, obs, attached)})")
            path, method = self.connect(c, q0, qg, obs, rng)
            return Attempt(path, {"method": method}) if path else Attempt(why=method)
        if goal.tcp is None:
            return Attempt(why="goal has neither tcp nor joints")
        T_goal = np.asarray(goal.tcp, float)

        if goal.pre is None and goal.verb in LINE_VERBS:
            T0 = arm.fk(q0)
            target = T_goal.copy()
            if goal.free:                               # keep the current rotation about the free axis
                target[:3, :3] = _align_free(T0[:3, :3], T_goal[:3, :3], goal.free)
            line = self.line(c, q0, T0, target, near if goal.verb in ("push", "press") else obs)
            if line is not None:
                r = arm.solve(target, line[-1])
                return Attempt(line, {"method": "line", "ik_pos": r.pos_err, "ik_rot": r.rot_err})
            if goal.verb in ("push", "press"):
                return Attempt(why="straight line from the current TCP is unreachable or collides")

        reasons: list[str] = []
        def solutions(T):
            """Converged IK solutions: the usual seeds, then many more random ones when those found nothing."""
            found = 0
            for r in islice(arm.ik_solutions(T, q0, goal.free, rng, cfg.ik_random), cfg.ik_solutions):
                found += 1
                yield r
            if not found and cfg.ik_random_more:
                more = np.random.default_rng(cfg.seed + 1)
                yield from islice(arm.ik_solutions(T, None, goal.free, more, cfg.ik_random_more), cfg.ik_solutions)

        if goal.pre is not None:
            T_pre = np.asarray(goal.pre, float)
            for r in solutions(T_pre):
                hit = c.check(r.q, obs, attached)
                if hit:
                    reasons.append(f"pre-pose in collision ({hit})")
                    continue
                T_a = arm.fk(r.q)
                T_b = T_goal.copy()
                if goal.free:
                    T_b[:3, :3] = _align_free(T_a[:3, :3], T_goal[:3, :3], goal.free)
                line = self.line(c, r.q, T_a, T_b, near)
                if line is None:
                    reasons.append("approach line from the pre-pose unreachable or hits the table")
                    continue
                path, method = self.connect(c, q0, r.q, obs, rng)
                if path is None:
                    reasons.append("no collision-free path to the pre-pose")
                    continue
                rr = arm.solve(T_b, line[-1])
                return Attempt(path + line[1:], {"method": method + "+line", "ik_pos": rr.pos_err,
                                                 "ik_rot": rr.rot_err})
            if not reasons:
                best = arm.ik(T_pre, q0, goal.free, rng, cfg.ik_random)
                reasons.append(f"pre-pose unreachable (ik residual {best.pos_err:.3f} m, {best.rot_err:.2f} rad)")
        else:
            for r in solutions(T_goal):
                hit = c.check(r.q, obs, attached)
                if hit:
                    reasons.append(f"goal in collision ({hit})")
                    continue
                path, method = self.connect(c, q0, r.q, obs, rng)
                if path is None:
                    reasons.append("no collision-free path to the goal")
                    continue
                return Attempt(path, {"method": method, "ik_pos": r.pos_err, "ik_rot": r.rot_err})
            if not reasons:
                best = arm.ik(T_goal, q0, goal.free, rng, cfg.ik_random)
                reasons.append(f"tcp unreachable (ik residual {best.pos_err:.3f} m, {best.rot_err:.2f} rad)")
        return Attempt(why="; ".join(dict.fromkeys(reasons)))

    def actions(self, goal, path: list[np.ndarray], q_now: dict[str, np.ndarray], hand_keep: dict[str, Any],
                action_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Timed (T, action_dim) rows: the arm follows the path with `hand_before`, then holds with `hand`."""
        cfg = self.cfg
        arm = self.kin[goal.arm]
        dt = 1.0 / cfg.control_hz
        vmax, amax = _limits(cfg.max_vel, arm.n), _limits(cfg.max_acc, arm.n)
        q0 = np.asarray(q_now[goal.arm], float)
        traj = time_parameterize([q0, *path[1:]], vmax, amax, dt)
        moving = len(traj)
        hold = int(np.ceil(cfg.hand_time * cfg.control_hz)) if goal.hand is not None else 0
        rows = np.zeros((moving + hold, action_dim))
        for spec in self.kin.spec.arms:
            s0, s1 = spec.action_slice
            h0, h1 = spec.hand_slice
            keep = hand_keep.get(spec.name)
            if spec.name == goal.arm:
                rows[:moving, s0:s1] = traj
                rows[moving:, s0:s1] = traj[-1]
                before = goal.hand_before if goal.hand_before is not None else keep
                rows[:moving, h0:h1] = np.asarray(before, float)
                rows[moving:, h0:h1] = np.asarray(goal.hand if goal.hand is not None else before, float)
            else:
                rows[:, s0:s1] = q_now[spec.name]
                rows[:, h0:h1] = np.asarray(keep, float)
        length = float(sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))) if len(path) > 1 else 0.0
        return rows, {"path_length": length, "waypoints": len(path), "moving_steps": moving, "hand_steps": hold,
                      "duration_s": len(rows) * dt}


def _body_name(m, g: int) -> str:
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or f"geom {g}"


def _align_free(R_actual: np.ndarray, R_goal: np.ndarray, free: str) -> np.ndarray:
    """R_actual rotated by the smallest rotation taking its free axis onto R_goal's."""
    k = {"x": 0, "z": 2}[free]
    a, b = R_actual[:, k], R_goal[:, k]
    v, cth = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    if np.linalg.norm(v) < 1e-9:
        return R_actual.copy() if cth > 0 else R_goal.copy()
    rot = Rotation.from_rotvec(v / np.linalg.norm(v) * np.arccos(cth)).as_matrix()
    return rot @ R_actual


__all__ = ["Planner", "PlannerConfig", "check_limits", "retime_together", "time_parameterize"]
