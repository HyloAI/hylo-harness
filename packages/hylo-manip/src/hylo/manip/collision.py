"""Collision checking of an arm and its hand: the model's collision geoms (as surface samples) against the table plane
and observed points (KD-tree, a margin), and self-collision from MuJoCo contacts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply, inv, voxel_down
from .kinematics import ArmModel

G = mujoco.mjtGeom  # compared as ints below


def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _grid(a: float, b: float, h: float) -> np.ndarray:
    u = np.linspace(-a, a, max(2, int(np.ceil(2 * a / h)) + 1))
    v = np.linspace(-b, b, max(2, int(np.ceil(2 * b / h)) + 1))
    return np.stack(np.meshgrid(u, v), -1).reshape(-1, 2)


def geom_surface(m: mujoco.MjModel, g: int, h: float, rng: np.random.Generator) -> np.ndarray:
    """Points on the surface of geom g in its own frame, about `h` apart."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == G.mjGEOM_BOX:
        pts = []
        for ax in range(3):
            o = [k for k in range(3) if k != ax]
            uv = _grid(s[o[0]], s[o[1]], h)
            for sign in (-1, 1):
                p = np.zeros((len(uv), 3))
                p[:, o] = uv
                p[:, ax] = sign * s[ax]
                pts.append(p)
        return np.concatenate(pts)
    if t in (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        r = s if t == G.mjGEOM_ELLIPSOID else np.full(3, s[0])
        n = max(12, int(4 * np.pi * r.max() ** 2 / h ** 2))
        return _fib_sphere(n) * r
    if t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        r, hl = s[0], s[1]
        nt = max(8, int(2 * np.pi * r / h))
        nz = max(2, int(np.ceil(2 * hl / h)) + 1)
        th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        z = np.linspace(-hl, hl, nz)
        side = np.stack([np.repeat(np.cos(th)[None] * r, nz, 0).ravel(), np.repeat(np.sin(th)[None] * r, nz, 0).ravel(),
                         np.repeat(z[:, None], nt, 1).ravel()], axis=1)
        if t == G.mjGEOM_CAPSULE:
            cap = _fib_sphere(max(12, int(4 * np.pi * r * r / h ** 2))) * r
            cap[:, 2] += np.where(cap[:, 2] >= 0, hl, -hl)
        else:
            disc = _grid(r, r, h)
            disc = disc[np.linalg.norm(disc, axis=1) <= r]
            cap = np.concatenate([np.c_[disc, np.full(len(disc), hl)], np.c_[disc, np.full(len(disc), -hl)]])
        return np.concatenate([side, cap])
    if t == G.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[va:va + vn].astype(float)
        F = m.mesh_face[fa:fa + fn]
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = int(min(20000, max(50, 3 * area.sum() / h ** 2)))
        tri = rng.choice(len(F), size=n, p=area / area.sum()) if area.sum() > 0 else np.zeros(n, int)
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        p = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
        return voxel_down(np.concatenate([p, V]), h)
    # planes, height fields, SDFs: the local bounding box
    aabb = m.geom_aabb[g]
    return geom_surface_box(aabb[:3], aabb[3:], h)


def geom_surface_box(center, half, h) -> np.ndarray:
    pts = []
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        uv = _grid(half[o[0]], half[o[1]], h)
        for sign in (-1, 1):
            p = np.zeros((len(uv), 3))
            p[:, o] = uv
            p[:, ax] = sign * half[ax]
            pts.append(p + center)
    return np.concatenate(pts)


def _subtree(m: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.add(b)
    return out


@dataclass
class Obstacles:
    """What an arm must not touch: the table plane (z, with a clearance) and points (KD-tree, a margin)."""
    table: float | None = None
    clearance: float = 0.01
    points: np.ndarray | None = None
    margin: float = 0.01
    check_self: bool = True
    attached: np.ndarray | None = None      # points held in the TCP frame, checked like the hand
    soft_fingers: bool = False              # finger links are not checked (compliant, torque-limited fingers)

    def __post_init__(self) -> None:
        pts = self.points
        self.tree = cKDTree(pts) if pts is not None and len(pts) else None
        if self.tree is not None:
            self.lo, self.hi = pts.min(0) - self.margin, pts.max(0) + self.margin

    def relaxed(self, **kw) -> "Obstacles":
        return replace(self, **kw)

    def hits(self, pts: np.ndarray) -> str | None:
        if self.table is not None and len(pts) and pts[:, 2].min() < self.table + self.clearance:
            return "table"
        if self.tree is not None and len(pts):
            sel = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
            if sel.any():
                d, _ = self.tree.query(pts[sel], distance_upper_bound=self.margin)
                if np.isfinite(d).any():
                    return "points"
        return None


class Collider:
    """Collision geoms of one arm's model: those of the bodies its joints move (with the hand) are checked against
    obstacles; self-collision uses MuJoCo contacts between robot geoms, minus pairs already touching at home and
    pairs within the hand."""

    def __init__(self, arm: ArmModel, robot_bodies: set[int], spacing: float = 0.01, seed: int = 0) -> None:
        self.arm = arm
        m = arm.m
        joint_bodies = {int(m.jnt_bodyid[j]) for j in arm.jids}
        self.moving = set().union(*(_subtree(m, b) for b in joint_bodies))
        self.hand_bodies = _subtree(m, arm.ee)
        self.robot_bodies = robot_bodies | self.moving
        collidable = (m.geom_contype | m.geom_conaffinity) != 0
        rng = np.random.default_rng(seed)
        geoms, samples, owner = [], [], []
        for g in range(m.ngeom):
            if collidable[g] and m.geom_bodyid[g] in self.moving:
                s = geom_surface(m, g, spacing, rng)
                samples.append(s)
                owner.append(np.full(len(s), len(geoms)))
                geoms.append(g)
        self.geoms = np.array(geoms, dtype=int)
        self.samples = np.concatenate(samples) if samples else np.zeros((0, 3))
        self.owner = np.concatenate(owner) if owner else np.zeros(0, int)
        self.hand_sample = np.isin(m.geom_bodyid[self.geoms][self.owner], list(self.hand_bodies))
        finger_bodies = set()
        for j in arm.spec.hand.joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                finger_bodies |= _subtree(m, int(m.jnt_bodyid[jid]))
        self.finger_sample = np.isin(m.geom_bodyid[self.geoms][self.owner], list(finger_bodies))
        self.robot_geoms = {g for g in range(m.ngeom) if collidable[g] and m.geom_bodyid[g] in self.robot_bodies}
        # what cameras see of the robot: every geom of its bodies (visual ones included, fixed base included), for
        # removing the robot from observed points; collision geoms alone can sit centimetres inside the visual
        # surface, and a fixed base has no moving collision geoms at all
        seen, seen_owner = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.robot_bodies and int(m.geom_type[g]) not in (G.mjGEOM_PLANE, G.mjGEOM_HFIELD):
                s = geom_surface(m, g, max(spacing, 0.01), rng)
                seen.append(s)
                seen_owner.append(np.full(len(s), g))
        self.seen_samples = np.concatenate(seen) if seen else np.zeros((0, 3))
        self.seen_geom = np.concatenate(seen_owner) if seen_owner else np.zeros(0, int)
        self.allowed: set[tuple[int, int]] = set()
        arm.set_q(arm.home)
        arm.set_hand(arm.hand_open)
        arm.forward()
        self.allowed = set(self._contacts())

    def _contacts(self) -> list[tuple[int, int]]:
        m, d = self.arm.m, self.arm.d
        mujoco.mj_collision(m, d)
        pairs = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist > 0 or g1 not in self.robot_geoms or g2 not in self.robot_geoms:
                continue
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 not in self.moving and b2 not in self.moving:
                continue
            if b1 in self.hand_bodies and b2 in self.hand_bodies:
                continue
            pair = (min(g1, g2), max(g1, g2))
            if pair not in self.allowed:
                pairs.append(pair)
        return pairs

    def points_world(self, hand_only: bool = False) -> np.ndarray:
        """Surface samples in the world at the current model state (after `arm.forward()`)."""
        d, base = self.arm.d, self.arm.base
        gidx = self.geoms[self.owner]
        S = self.samples
        if hand_only:
            gidx, S = gidx[self.hand_sample], S[self.hand_sample]
        R = d.geom_xmat[gidx].reshape(-1, 3, 3)
        p = np.einsum("nij,nj->ni", R, S) + d.geom_xpos[gidx]
        return p @ base[:3, :3].T + base[:3, 3]

    def seen_world(self) -> np.ndarray:
        """Surface samples of every robot geom (visual and collision, moving and fixed) in the world at the current
        model state (after `arm.forward()`): what cameras see of the robot."""
        d, base = self.arm.d, self.arm.base
        R = d.geom_xmat[self.seen_geom].reshape(-1, 3, 3)
        p = np.einsum("nij,nj->ni", R, self.seen_samples) + d.geom_xpos[self.seen_geom]
        return p @ base[:3, :3].T + base[:3, 3]

    def check(self, q, obstacles: Obstacles, attached: np.ndarray | None = None) -> str | None:
        """None when q is free, else "self", "table" or "points". `attached`: points held in the TCP frame (default:
        the obstacles' `attached`)."""
        if attached is None:
            attached = obstacles.attached
        self.arm.set_q(q)
        self.arm.forward()
        if obstacles.check_self and self._contacts():
            return "self"
        pts = self.points_world()
        if obstacles.soft_fingers:
            pts = pts[~self.finger_sample]
        hit = obstacles.hits(pts)
        if hit is None and attached is not None and len(attached):
            hit = obstacles.hits(apply(self.arm.tcp(), attached))
        return hit

    def hand_points_in_tcp(self, cmd, spacing: float = 0.005) -> np.ndarray:
        """Hand surface samples (about `spacing` apart) in the TCP frame with the fingers at command `cmd`."""
        arm, m = self.arm, self.arm.m
        arm.set_q(arm.home)
        arm.set_hand(cmd)
        arm.forward()
        rng = np.random.default_rng(0)
        pts = []
        for g in self.geoms:
            if m.geom_bodyid[g] in self.hand_bodies:
                R = arm.d.geom_xmat[g].reshape(3, 3)
                pts.append(geom_surface(m, g, spacing, rng) @ R.T + arm.d.geom_xpos[g])
        if not pts:
            return np.zeros((0, 3))
        return apply(inv(arm.ee_model() @ arm.tcp_in_ee), np.concatenate(pts))


__all__ = ["Collider", "Obstacles", "geom_surface"]
