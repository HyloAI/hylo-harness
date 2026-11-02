"""Multi-finger grasp planning for any dexterous hand (a HandModel: HandSpec plus its MJCF) on an object given as
points or a mesh.

Object: one or more convex parts (the hull of the points, of the mesh, or of each given part), so the signed
distance to the object is the smallest of the parts' hull distances and a ray's entry into the object is exact.
A partial point cloud seen from above is extruded down to the table first (the hidden underside is solid).

Planning, per candidate hand pose:
  1. sample: approach directions (+x of the TCP) over the sphere, minus those from below the table, times rolls
     about the approach (the +y closing axis); the TCP aims at the object's centre (plus offsets during refinement);
  2. approach: the open hand (a rigid point set) slides along the approach until its first point would enter the
     object or the table (ray-hull entry distances, exact), never past the centre;
  3. close: each finger moves from the open command towards the closing command (HandSpec.closed for "power",
     the "pinch" preset for "pinch") until its links touch the object (a sweep, then bisection per finger);
  4. score: contacts are the finger and palm samples within a few millimetres of the surface (one per link, the
     hull facet's normal); the epsilon measure of their friction-cone wrenches (force closure when positive), the
     total normal force needed to hold the object against gravity, penetration, how many fingers touch, and an
     optional preferred approach direction or pose;
  5. refine: the best few are perturbed (offset, tilt, roll) for a couple of rounds and re-scored (a short
     cross-entropy search over the pose; the fingers re-close each time).

`plan()` returns DexGrasp records, best first; `to_goals()` turns them into toolkit Goals (pre-pose back along the
approach with the hand open, then the hand closing to a squeeze command past the contact angles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry import angle, frame, pose, voxel_down
from .grasp_quality import grasp_epsilon, holding_effort
from .handmodel import HandModel


# ---------------------------------------------------------------------------------------------------- object
def _hull(points: np.ndarray, max_vertices: int = 120) -> np.ndarray:
    """Facet planes (F, 4) [n, b] of the convex hull, n outward unit, n.x + b <= 0 inside. A hull with many
    vertices (a sampled curved surface) is rebuilt from a spread-out subset of its vertices (one per grid cell),
    which keeps it on the surface (sagitta well under a millimetre for centimetre cells on curved objects)."""
    P = np.asarray(points, float)

    def hull(X):
        try:
            return ConvexHull(X)
        except QhullError:
            return ConvexHull(X, qhull_options="QJ")

    h = hull(P)
    V = P[h.vertices]
    if len(V) > max_vertices:
        ext = float(np.ptp(V, axis=0).max())
        cell = ext / 10
        for _ in range(6):
            keys = np.floor(V / cell).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            if len(first) <= max_vertices:
                break
            cell *= 1.3
        h = hull(V[np.sort(first)])
    return h.equations.copy()


def mesh_surface(V: np.ndarray, F: np.ndarray, spacing: float = 0.004, seed: int = 0) -> np.ndarray:
    """Points on a triangle mesh, about `spacing` apart (plus the vertices)."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(min(40000, max(100, 3 * area.sum() / spacing ** 2)))
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(F), size=n, p=area / area.sum()) if area.sum() > 0 else np.zeros(n, int)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    p = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    return voxel_down(np.vstack([p, V]), spacing * 0.75)


@dataclass
class ObjectGeom:
    """An object as convex parts (list of facet arrays) with its surface samples."""
    planes: list[np.ndarray]
    points: np.ndarray
    centre: np.ndarray
    radius: float
    mass: float = 0.1
    part_spheres: list | None = None     # (centre, radius + margin) per part, for the broad phase

    @classmethod
    def from_parts(cls, parts: Sequence[np.ndarray], mass: float = 0.1, spacing: float = 0.004) -> "ObjectGeom":
        """`parts`: point sets (world), each one convex part (its hull)."""
        parts = [np.asarray(p, float).reshape(-1, 3) for p in parts if len(p) >= 4]
        if not parts:
            raise ValueError("an object needs at least one part with 4 or more points")
        planes = [_hull(p) for p in parts]
        pts = voxel_down(np.vstack(parts), spacing * 0.75)
        # keep surface samples only (inside points of one part may lie inside another)
        d = _sdf_planes(planes, pts)[0]
        surf = pts[d > -0.002] if (d > -0.002).sum() >= 10 else pts
        allp = np.vstack(parts)
        centre = allp.mean(0) if len(parts) > 1 else _hull_centroid(parts[0])
        radius = float(np.linalg.norm(allp - centre, axis=1).max())
        spheres = []
        for p in parts:
            c = (p.min(0) + p.max(0)) / 2
            spheres.append((c, float(np.linalg.norm(p - c, axis=1).max())))
        return cls(planes, surf, centre, radius, mass, spheres if len(parts) > 1 else None)

    @classmethod
    def from_points(cls, points: np.ndarray, table_z: float | None = None, mass: float = 0.1) -> "ObjectGeom":
        """One convex part from observed points; with a table height the points are extruded down to it."""
        P = np.asarray(points, float).reshape(-1, 3)
        if table_z is not None:
            P = P[P[:, 2] > table_z - 0.005]
            low = P.copy()
            low[:, 2] = table_z + 0.001
            P = np.vstack([P, voxel_down(low, 0.006)])
        return cls.from_parts([P], mass)

    @classmethod
    def from_mesh(cls, V: np.ndarray, F: np.ndarray, parts: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
                  mass: float = 0.1, spacing: float = 0.004) -> "ObjectGeom":
        """A mesh (its hull), or several convex parts of it (V, F each) for a concave object."""
        if parts:
            return cls.from_parts([mesh_surface(v, f, spacing) for v, f in parts], mass, spacing)
        return cls.from_parts([mesh_surface(V, F, spacing)], mass, spacing)

    def sdf(self, P: np.ndarray, far: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance (negative inside; outside a lower bound of the true distance) and the outward normal of
        the nearest facet. With `far`, points farther than radius + far from the centre are not evaluated (their
        distance is reported as `far`, normal zero)."""
        if far is None:
            return _sdf_planes(self.planes, P)            # exact everywhere (no broad phase)
        P = np.asarray(P, float).reshape(-1, 3)
        near = np.einsum("ij,ij->i", P - self.centre, P - self.centre) < (self.radius + far) ** 2
        d = np.full(len(P), float(far))
        n = np.zeros((len(P), 3))
        if near.any():
            sph = None if self.part_spheres is None else [(c, r + far) for c, r in self.part_spheres]
            d[near], n[near] = _sdf_planes(self.planes, P[near], sph)
            d = np.minimum(d, far)
        return d, n

    def entry(self, P: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Per point, the smallest advance s >= 0 along unit direction a at which it enters the object (inf: never)."""
        best = np.full(len(P), np.inf)
        rel = P - self.centre                       # broad phase: rays that pass within the bounding sphere
        along = rel @ a
        perp2 = np.einsum("ij,ij->i", rel, rel) - along ** 2
        cand = np.flatnonzero((perp2 < self.radius ** 2 * 1.0001) & (along < self.radius))
        if not len(cand):
            return best
        P = P[cand]
        sub = np.full(len(P), np.inf)
        for pi, E in enumerate(self.planes):
            sel = np.arange(len(P))
            if self.part_spheres is not None:
                c, r = self.part_spheres[pi]
                rl = P - c
                al = rl @ a
                sel = np.flatnonzero((np.einsum("ij,ij->i", rl, rl) - al ** 2 < r * r * 1.0001) & (al < r))
                if not len(sel):
                    continue
            n, b = E[:, :3], E[:, 3]
            g = P[sel] @ n.T + b                         # (N, F) plane values now
            na = n @ a                              # (F,)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -g / na
            lo = np.where(na < -1e-12, t, -np.inf)  # entering through facets the ray faces
            hi = np.where(na > 1e-12, t, np.inf)
            par = np.abs(na) <= 1e-12
            blocked = (par & (g > 0)).any(1)
            s_in, s_out = lo.max(1), hi.min(1)
            ok = (s_in <= s_out) & (s_out >= 0) & ~blocked
            s = np.where(ok, np.maximum(s_in, 0.0), np.inf)
            sub[sel] = np.minimum(sub[sel], s)
        best[cand] = sub
        return best


def _sdf_planes(planes: list[np.ndarray], P: np.ndarray,
                spheres: list[tuple[np.ndarray, float]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float).reshape(-1, 3)
    best = np.full(len(P), np.inf)
    normal = np.zeros((len(P), 3))
    for i, E in enumerate(planes):
        if spheres is not None:
            c, r = spheres[i]
            idx = np.flatnonzero(np.einsum("ij,ij->i", P - c, P - c) < r * r)
            if not len(idx):
                continue
            D = P[idx] @ E[:, :3].T + E[:, 3]
            k = D.argmax(1)
            d = D[np.arange(len(idx)), k]
            better = d < best[idx]
            best[idx[better]] = d[better]
            normal[idx[better]] = E[k[better], :3]
            continue
        D = P @ E[:, :3].T + E[:, 3]
        k = D.argmax(1)
        d = D[np.arange(len(P)), k]
        better = d < best
        best = np.where(better, d, best)
        normal[better] = E[k[better], :3]
    return best, normal


def _hull_centroid(P: np.ndarray) -> np.ndarray:
    try:
        h = ConvexHull(P)
    except QhullError:
        return P.mean(0)
    c0 = P[h.vertices].mean(0)
    vol, acc = 0.0, np.zeros(3)
    for s in h.simplices:
        a, b, c = P[s]
        v = abs(np.dot(a - c0, np.cross(b - c0, c - c0))) / 6
        vol += v
        acc += v * (a + b + c + c0) / 4
    return acc / vol if vol > 0 else c0


# ---------------------------------------------------------------------------------------------------- grasps
@dataclass
class DexGrasp:
    tcp: np.ndarray                 # (4, 4) TCP pose, world
    approach: np.ndarray            # unit approach direction (the TCP's +x)
    q_open: np.ndarray              # finger command while approaching
    q_contact: np.ndarray           # finger joints at first contact (or the closing target when a finger misses)
    q_squeeze: np.ndarray           # finger command to hold: past the contacts
    contacts: np.ndarray            # (m, 3) contact points
    normals: np.ndarray             # (m, 3) inward normals
    contact_fingers: list[str]      # finger (or "palm") of each contact
    epsilon: float                  # wrench-hull measure (> 0: force closure)
    effort: float                   # normal force to hold against gravity, in units of the object's weight
    penetration: float              # deepest hand sample inside the object (m)
    score: float                    # lower is better
    style: str
    refined: bool = False
    info: dict = field(default_factory=dict)

    @property
    def force_closure(self) -> bool:
        return self.epsilon > 1e-6


@dataclass
class PlannerConfig:
    mu: float = 0.5                 # friction coefficient assumed for scoring
    cone_edges: int = 8
    n_dirs: int = 32                # approach directions sampled on the sphere
    n_rolls: int = 6                # rolls about each approach
    min_down: float = -0.25         # approach z must be <= this ... (-1 straight down, 0 horizontal)
    max_up: float = 0.2             # ... or at most this much upward (approaches from below the table excluded)
    table_clearance: float = 0.004
    contact_tol: float = 0.004      # samples this close to the surface count as touching
    pen_tol: float = 0.0015         # a finger stops when a sample is this deep
    max_penetration: float = 0.004
    sweep: int = 14                 # closing sweep samples (then 4 bisection steps)
    squeeze: float = 0.35           # squeeze command: contact closure + this fraction of the closing motion
    pre_dist: float = 0.10
    keep: int = 10                  # grasps returned
    refine_top: int = 6
    refine_rounds: int = 2
    refine_samples: int = 10
    offset_sigma: float = 0.008
    tilt_sigma: float = 0.15
    seed: int = 0


class DexGraspPlanner:
    """Grasps of one hand. See the module docstring."""

    def __init__(self, hand: HandModel, config: PlannerConfig | None = None) -> None:
        self.hand, self.cfg = hand, config or PlannerConfig()
        self._fmask = hand.pt_finger >= 0                # finger samples (the palm stays where the approach left it)
        self._open_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ targets
    def closing_target(self, style: str) -> np.ndarray:
        h = self.hand
        if style == "power":
            return h.closed.copy()
        if style in h.hand.presets:
            return np.asarray(h.hand.presets[style], float)
        raise KeyError(f"no closing style {style!r}; use power or one of {sorted(h.hand.presets)}")

    def _open_points(self, q_open: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = np.round(q_open, 5).tobytes()
        if key not in self._open_cache:
            self._open_cache[key] = self.hand.points_in_tcp(q_open)
        return self._open_cache[key]

    # ------------------------------------------------------------------ one candidate
    def evaluate(self, obj: ObjectGeom, R: np.ndarray, aim: np.ndarray, table_z: float | None, style: str,
                 q_open: np.ndarray | None = None, scene: np.ndarray | None = None,
                 prefer: dict | None = None) -> DexGrasp | None:
        """Approach along R's x axis towards `aim`, close, score. None when the hand cannot get near. Without
        `q_open` the hand approaches open, and again with a wider pre-shape (half way to the "flat" preset) when the
        open hand stops short of the aim (fingertips on the object's top); the better grasp is kept."""
        if q_open is None:
            g = self._evaluate(obj, R, aim, table_z, style, self.hand.open, scene, prefer)
            wide = self.wide_preshape()
            if wide is not None and (g is None or g.info["stop_m"] > 0.5 * obj.radius):
                g2 = self._evaluate(obj, R, aim, table_z, style, wide, scene, prefer)
                if g2 is not None and (g is None or g2.score < g.score):
                    g = g2
            return g
        return self._evaluate(obj, R, aim, table_z, style, np.asarray(q_open, float), scene, prefer)

    def wide_preshape(self) -> np.ndarray | None:
        h = self.hand
        flat = h.hand.presets.get("flat")
        return None if flat is None else h.clip(h.open + 0.5 * (np.asarray(flat, float) - h.open))

    def _evaluate(self, obj: ObjectGeom, R: np.ndarray, aim: np.ndarray, table_z: float | None, style: str,
                  q_open: np.ndarray, scene: np.ndarray | None, prefer: dict | None) -> DexGrasp | None:
        cfg, h = self.cfg, self.hand
        target = self.closing_target(style)
        a = R[:, 0]
        P_local, _ = self._open_points(q_open)
        far = obj.radius + float(np.linalg.norm(P_local, axis=1).max()) + 0.02
        start = aim - a * far
        P0 = P_local @ R.T + start
        s = obj.entry(P0, a)
        adv = float(s.min()) - 0.001
        if table_z is not None and a[2] < -1e-6:
            room = (P0[:, 2] - table_z - cfg.table_clearance) / -a[2]
            adv = min(adv, float(room.min()))
        if table_z is not None and (P0[:, 2] < table_z + cfg.table_clearance).any():
            return None
        adv = min(adv, far)                                 # never past the aim point
        if not np.isfinite(adv) or adv < 0:
            return None
        T = pose(R, start + a * adv)
        if np.linalg.norm(T[:3, 3] - aim) > obj.radius + 0.03:   # stopped far from the object: nothing to grasp
            return None
        if scene is not None and len(scene):
            from scipy.spatial import cKDTree

            P = P_local @ R.T + T[:3, 3]
            if np.isfinite(cKDTree(scene).query(P, distance_upper_bound=0.004)[0]).any():
                return None
        q_c, closure = self._close(obj, T, q_open, target, table_z)
        stop = float(np.linalg.norm(T[:3, 3] - aim))
        return self._score(obj, T, q_open, q_c, closure, target, style, prefer, stop)

    def _close(self, obj: ObjectGeom, T: np.ndarray, q_open: np.ndarray, target: np.ndarray,
               table_z: float | None) -> tuple[np.ndarray, np.ndarray]:
        """Per-finger closure s in [0, 1] at first contact (1: never touched): (finger joints, s per finger)."""
        h, cfg = self.hand, self.cfg
        nf = len(h.fingers)
        jf = np.zeros(len(h.jids), int)
        for i, f in enumerate(h.fingers):
            jf[f.joints] = i

        def blocked(svec: np.ndarray) -> np.ndarray:
            q = q_open + svec[jf] * (target - q_open)
            h.set(T, q)
            P, _ = h.surface(self._fmask)
            d, _ = obj.sdf(P, far=0.01)
            bad = d < -cfg.pen_tol
            if table_z is not None:
                bad |= P[:, 2] < table_z + 0.5 * cfg.table_clearance
            out = np.zeros(nf, bool)
            fi = h.pt_finger[self._fmask]
            for i in range(nf):
                out[i] = bad[fi == i].any()
            return out

        grid = np.linspace(0.0, 1.0, cfg.sweep + 1)
        lo, hi = np.zeros(nf), np.ones(nf)
        done = np.zeros(nf, bool)
        start_blocked = blocked(np.zeros(nf))
        lo[start_blocked], hi[start_blocked], done[start_blocked] = 0.0, 0.0, True
        for k in range(1, len(grid)):
            if done.all():
                break
            b = blocked(np.where(done, lo, grid[k]))
            new = b & ~done
            lo[new], hi[new] = grid[k - 1], grid[k]
            done |= new
            lo[~done] = grid[k]
        hi[~done] = lo[~done] = 1.0
        for _ in range(4):
            mid = (lo + hi) / 2
            b = blocked(mid)
            hi = np.where(b, mid, hi)
            lo = np.where(b, lo, mid)
        return q_open + lo[jf] * (target - q_open), lo

    def _score(self, obj: ObjectGeom, T: np.ndarray, q_open: np.ndarray, q_c: np.ndarray, closure: np.ndarray,
               target: np.ndarray, style: str, prefer: dict | None, stop: float) -> DexGrasp | None:
        h, cfg = self.hand, self.cfg
        h.set(T, q_c)
        P, body = h.surface()
        d, n = obj.sdf(P, far=0.02)
        pen = float(max(0.0, -d.min()))
        if pen > cfg.max_penetration:
            return None
        touch = d < cfg.contact_tol
        cps, cns, who = [], [], []
        for b in np.unique(body[touch]):
            sel = touch & (body == b)
            # contact: the deepest few samples of this link, normal from the facets there
            idx = np.flatnonzero(sel)
            idx = idx[np.argsort(d[idx])[:6]]
            p = P[idx].mean(0)
            nn = n[idx].mean(0)
            if np.linalg.norm(nn) < 1e-6:
                continue
            nn /= np.linalg.norm(nn)
            p = p - nn * float(np.mean(d[idx]))            # onto the surface
            cps.append(p)
            cns.append(-nn)
            fi = h.pt_finger[np.flatnonzero(h.pt_body == b)[0]]
            who.append("palm" if fi < 0 else h.fingers[fi].name)
        fingers_touching = {w for w in who if w != "palm"}
        if not cps:
            eps, effort = 0.0, float("inf")
        else:
            C, N = np.asarray(cps), np.asarray(cns)
            eps = grasp_epsilon(C, N, obj.centre, cfg.mu, cfg.cone_edges, obj.radius)
            g = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
            effort = holding_effort(C, N, obj.centre, g, cfg.mu, cfg.cone_edges, obj.radius)
        thumb = h.fingers[h.thumb].name in fingers_touching
        nfing = len(h.fingers)
        score = (-6.0 * eps + 0.15 * min(effort, 10.0) + 0.6 * (1 - len(fingers_touching) / nfing)
                 + (0.0 if thumb else 0.5) + 30.0 * pen + 0.4 * np.linalg.norm(np.mean(cps, 0) - obj.centre) / obj.radius
                 if cps else 5.0)
        if eps <= 1e-6:
            score += 2.0
        score += 0.5 * stop / obj.radius                          # the hand around the object, not perched on it
        R = T[:3, :3]
        if prefer:
            if prefer.get("approach") is not None:
                pa = np.asarray(prefer["approach"], float)
                score += prefer.get("w_approach", 0.5) * (1 - float(R[:, 0] @ pa)) / 2
            if prefer.get("tcp") is not None:
                Tp = np.asarray(prefer["tcp"], float)
                score += prefer.get("w_rot", 0.6) * angle(Tp[:3, :3].T @ R) / np.pi
                score += prefer.get("w_pos", 0.5) * float(np.linalg.norm(Tp[:3, 3] - T[:3, 3])) / 0.05
        s = closure
        jf = np.zeros(len(h.jids), int)
        for i, f in enumerate(h.fingers):
            jf[f.joints] = i
        s_cmd = np.minimum(1.0, s + cfg.squeeze * self._balance(who, cns))
        q_sq = h.clip(q_open + s_cmd[jf] * (target - q_open))
        return DexGrasp(tcp=T.copy(), approach=R[:, 0].copy(), q_open=q_open.copy(), q_contact=q_c.copy(),
                        q_squeeze=q_sq, contacts=np.asarray(cps).reshape(-1, 3), normals=np.asarray(cns).reshape(-1, 3),
                        contact_fingers=who, epsilon=float(eps), effort=float(effort), penetration=pen,
                        score=float(score), style=style,
                        info={"closure": np.round(s, 3).tolist(), "stop_m": round(stop, 4)})

    def _balance(self, who: list[str], normals: list[np.ndarray]) -> np.ndarray:
        """Per-finger squeeze weights whose pushes (along each finger's mean contact normal) cancel, so position
        servos of equal stiffness do not shove the object out towards the weaker side (e.g. three fingers against
        one thumb). Palm contacts push back as much as needed. Weights average 1 over the touching fingers."""
        from scipy.optimize import nnls

        h = self.hand
        w = np.ones(len(h.fingers))
        names = [f.name for f in h.fingers]
        touching = [i for i, n in enumerate(names) if n in who]
        if len(touching) < 2:
            return w
        cols, sumrow = [], []
        for i in touching:
            nn = np.mean([normals[k] for k, x in enumerate(who) if x == names[i]], 0)
            cols.append(nn / max(np.linalg.norm(nn), 1e-9))
            sumrow.append(1.0)
        palm = [normals[k] for k, x in enumerate(who) if x == "palm"]
        if palm:
            nn = np.mean(palm, 0)
            cols.append(nn / max(np.linalg.norm(nn), 1e-9))
            sumrow.append(0.0)
        lam = 0.3
        A = np.vstack([np.array(cols).T, np.sqrt(lam) * np.array(sumrow)[None]])
        b = np.r_[np.zeros(3), np.sqrt(lam) * len(touching)]
        sol, _ = nnls(A, b)
        for k, i in enumerate(touching):
            w[i] = float(np.clip(sol[k], 0.3, 2.5))
        return w

    # ------------------------------------------------------------------ search
    def candidates(self, table_z: float | None) -> list[np.ndarray]:
        cfg = self.cfg
        n = cfg.n_dirs
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = np.pi * (1 + 5 ** 0.5) * i
        dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        dirs = np.vstack([dirs, [[0.0, 0.0, -1.0]]])
        if table_z is not None:
            dirs = dirs[dirs[:, 2] <= cfg.max_up]
        out = []
        for a in dirs:
            ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
            y0 = np.cross(ref, a)
            y0 /= np.linalg.norm(y0)
            for k in range(cfg.n_rolls):
                t = 2 * np.pi * k / cfg.n_rolls
                y = np.cos(t) * y0 + np.sin(t) * np.cross(a, y0)
                out.append(frame(a, y))
        return out

    def plan(self, obj: ObjectGeom, table_z: float | None = None, style: str | None = None,
             scene: np.ndarray | None = None, prefer: dict | None = None,
             rotations: Sequence[np.ndarray] | None = None) -> list[DexGrasp]:
        """Ranked grasps (best first). style: "power", "pinch" (a preset name) or None for both. prefer: {"approach":
        unit vector, "tcp": 4x4 pose, "w_approach", "w_rot", "w_pos"} biases the ranking towards it. rotations:
        candidate TCP rotations instead of the sampled sphere."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        styles = [style] if style else ["power"] + (["pinch"] if "pinch" in self.hand.hand.presets else [])
        rots = list(rotations) if rotations is not None else self.candidates(table_z)
        found: list[DexGrasp] = []
        for st in styles:
            for R in rots:
                g = self.evaluate(obj, R, obj.centre, table_z, st, scene=scene, prefer=prefer)
                if g is not None:
                    found.append(g)
        found.sort(key=lambda g: g.score)
        # refine: a short cross-entropy search over offset, tilt and roll around the best few
        refined = []
        for g in found[:cfg.refine_top]:
            best = g
            sig_o, sig_r = cfg.offset_sigma, cfg.tilt_sigma
            for _ in range(cfg.refine_rounds):
                R0 = best.tcp[:3, :3]
                aim0 = best.info.get("aim", obj.centre)
                trial = []
                for _ in range(cfg.refine_samples):
                    w = rng.normal(0, sig_r, 3)
                    R = _expm(w) @ R0
                    off = rng.normal(0, sig_o, 3)
                    off -= (off @ R[:, 0]) * R[:, 0]
                    aim = np.asarray(aim0) + off
                    c = self.evaluate(obj, R, aim, table_z, best.style, scene=scene, prefer=prefer)
                    if c is not None:
                        c.info["aim"] = aim
                        trial.append(c)
                if trial:
                    top = min(trial, key=lambda c: c.score)
                    if top.score < best.score:
                        best = top
                        best.refined = True
                sig_o, sig_r = sig_o * 0.6, sig_r * 0.6
            refined.append(best)
        out = refined + found[cfg.refine_top:]
        out.sort(key=lambda g: g.score)
        # distinct grasps only (pose differs by 1 cm or 0.2 rad)
        keep: list[DexGrasp] = []
        for g in out:
            if all(np.linalg.norm(g.tcp[:3, 3] - k.tcp[:3, 3]) > 0.01 or angle(g.tcp[:3, :3].T @ k.tcp[:3, :3]) > 0.2
                   for k in keep):
                keep.append(g)
            if len(keep) >= cfg.keep:
                break
        return keep

    def pre_pose(self, g: DexGrasp, dist: float | None = None) -> np.ndarray:
        pre = g.tcp.copy()
        pre[:3, 3] -= g.approach * (self.cfg.pre_dist if dist is None else dist)
        return pre


def _expm(w: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(w).as_matrix()


def to_goals(grasps: Sequence[DexGrasp], arm: str | None, pre_dist: float = 0.10) -> list:
    """Toolkit Goals (verb "grasp"): pre-pose back along the approach, hand open while moving, then the squeeze."""
    from .datatypes import Goal

    out = []
    for g in grasps:
        pre = g.tcp.copy()
        pre[:3, 3] -= g.approach * pre_dist
        out.append(Goal("grasp", arm, tcp=g.tcp.copy(), pre=pre, hand_before=tuple(np.round(g.q_open, 4)),
                        hand=tuple(np.round(g.q_squeeze, 4)), rank=float(g.score)))
    return out


__all__ = ["DexGrasp", "DexGraspPlanner", "ObjectGeom", "PlannerConfig", "mesh_surface", "to_goals"]
