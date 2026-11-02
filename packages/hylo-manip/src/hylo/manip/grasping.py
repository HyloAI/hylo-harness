"""Grasp candidates for a parallel hand from an object's points: top-down across the narrow side (and the long side
when it fits), side grasps from the robot's side, each checked for width, antipodal support and hand collisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .datatypes import Box, Table
from .geometry import angle, apply, frame, inv, pose


@dataclass
class Candidate:
    tcp: np.ndarray
    pre: np.ndarray
    score: float          # lower is better
    kind: str
    width: float
    support: float


def hand_collides(T: np.ndarray, hand_pts: np.ndarray, table: Table, tree: cKDTree | None,
                  margin: float = 0.004, table_clearance: float = 0.002) -> str | None:
    pts = apply(T, hand_pts)
    if pts[:, 2].min() < table.height + table_clearance:
        return "table"
    if tree is not None and np.isfinite(tree.query(pts, distance_upper_bound=margin)[0]).any():
        return "points"
    return None


def grasp_candidates(box: Box, points: np.ndarray, table: Table, hand_pts: np.ndarray, max_width: float,
                     base_xy: np.ndarray, R_now: np.ndarray | None = None, scene: np.ndarray | None = None,
                     depth: float = 0.03, pre_dist: float = 0.10, clearance: float = 0.008,
                     pad_half: float = 0.015) -> list[Candidate]:
    """Ranked candidates. `hand_pts`: hand surface in the TCP frame with the fingers open. `depth`: how far below the
    object's top the TCP goes at most (half the height for short objects)."""
    pts = np.asarray(points, float)
    # how far the hand reaches past the TCP before the palm: hand points between the open pads lie behind the TCP
    # (-x); the TCP goes no deeper than that into an object (top grasps) and backs off for side grasps
    between = hand_pts[np.abs(hand_pts[:, 1]) < max_width / 2 - 0.002] if len(hand_pts) else hand_pts
    behind = -between[:, 0][between[:, 0] < 0]
    palm = float(behind.min()) if len(behind) else np.inf
    depth = min(depth, palm - 0.012)              # 8 mm under the palm, plus the collision margin
    obstacles = pts if scene is None or len(scene) == 0 else np.vstack([pts, np.asarray(scene, float)])
    tree = cKDTree(obstacles) if len(obstacles) else None
    top = box.bottom + box.size[2]
    c = box.center
    cy, sy = np.cos(box.yaw), np.sin(box.yaw)
    ax_len, ax_wid = np.array([cy, sy, 0.0]), np.array([-sy, cy, 0.0])
    to_obj = c[:2] - np.asarray(base_xy, float)[:2]
    to_obj = np.r_[to_obj / (np.linalg.norm(to_obj) or 1.0), 0.0]
    down = np.array([0.0, 0.0, -1.0])
    proposals = []                                     # (approach, closing axis, kind, kind cost)
    for closing, kind, cost in ((ax_wid, "top", 0.0), (ax_len, "top_long", 0.5)):
        for sgn in (1, -1):
            proposals.append((down, sgn * closing, kind, cost))
    for normal, extent in ((ax_len, box.size[0]), (-ax_len, box.size[0]), (ax_wid, box.size[1]), (-ax_wid, box.size[1])):
        a = -normal
        if a @ to_obj > 0.3:
            for sgn in (1, -1):
                proposals.append((a, sgn * np.cross([0.0, 0.0, 1.0], a), "side", 1.0, extent))
    out = []
    for a, y, kind, cost, *ext in proposals:
        R = frame(a, y)
        lowest = table.height + 0.004 - (hand_pts @ R.T)[:, 2].min()   # hand clears the table
        if kind == "side":
            z = max(box.bottom + box.size[2] / 2, lowest)
            if z > top - 0.01:
                continue
        else:
            z = max(top - min(box.size[2] / 2, depth), lowest)
        T = pose(R, [c[0], c[1], z])
        if kind == "side":                             # the palm stays clear of the object's near side
            T[:3, 3] -= a * max(0.0, ext[0] / 2 + 0.008 - palm)
        local = apply(inv(T), pts)
        slab = local[(np.abs(local[:, 0]) < pad_half + 0.01) & (np.abs(local[:, 2]) < pad_half + 0.01)]
        use = slab if len(slab) >= 5 else local
        lo, hi = np.percentile(use[:, 1], [2, 98])
        width = float(hi - lo)
        if width > max_width - 2 * clearance:
            continue
        mid = (lo + hi) / 2
        sides = (slab[:, 1] < mid - 0.3 * width).any() + (slab[:, 1] > mid + 0.3 * width).any() if len(slab) else 0
        support = sides / 2
        T[:3, 3] += R[:, 1] * mid                      # centre the pads on the measured extent
        pre = T.copy()
        pre[:3, 3] -= a * pre_dist
        if hand_collides(T, hand_pts, table, tree) or hand_collides(pre, hand_pts, table, tree):
            continue
        turn = angle(R_now.T @ R) / np.pi if R_now is not None else 0.0
        score = cost + 0.3 * (1 - support) + 0.1 * turn + 0.2 * width / max_width
        out.append(Candidate(T, pre, float(score), kind, width, float(support)))
    return sorted(out, key=lambda k: k.score)


__all__ = ["Candidate", "grasp_candidates", "hand_collides"]
