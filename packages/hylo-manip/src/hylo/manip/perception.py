"""Perception from RGB-D: back-projection and fusion, the table plane (RANSAC), segmenters behind one interface
(environment segmentation, named colours, VLM marks), per-object point clouds, oriented boxes and ICP."""

from __future__ import annotations

import re
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from .datatypes import Box, CameraView, Detection, Fit, Mask, Table, View
from .geometry import apply, pose, rotz, voxel_down


# ---------------------------------------------------------------------- points
def backproject(cam: CameraView, mask: np.ndarray | None = None, max_depth: float = 4.0) -> np.ndarray:
    """World points of the camera's valid depth pixels (optionally only inside `mask`)."""
    if cam.depth is None or cam.K is None or cam.T_world_cam is None:
        return np.zeros((0, 3))
    z = np.asarray(cam.depth, float)
    valid = np.isfinite(z) & (z > 0) & (z < max_depth)
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = z[v, u]
    K = np.asarray(cam.K, float)
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return apply(np.asarray(cam.T_world_cam, float), p)


def fuse(cams: dict[str, CameraView], voxel: float = 0.005) -> np.ndarray:
    pts = [backproject(c) for c in cams.values()]
    pts = [p for p in pts if len(p)]
    return voxel_down(np.concatenate(pts), voxel) if pts else np.zeros((0, 3))


def remove_outliers(points: np.ndarray, k: int = 8, std: float = 2.0) -> np.ndarray:
    """Statistical outlier removal: drop points whose mean distance to k neighbours exceeds mean + std * sigma."""
    if len(points) <= k:
        return points
    d, _ = cKDTree(points).query(points, k=k + 1)
    m = d[:, 1:].mean(1)
    return points[m <= m.mean() + std * m.std()]


def components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Index arrays of the groups of points connected by steps shorter than `radius`, largest first."""
    if len(points) == 0:
        return []
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    n = len(points)
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        g = coo_matrix((n, n))
    _, labels = connected_components(g, directed=False)
    groups = [np.flatnonzero(labels == i) for i in np.unique(labels)]
    return sorted(groups, key=len, reverse=True)


def ransac_table(points: np.ndarray, rng: np.random.Generator, iters: int = 300, thresh: float = 0.008,
                 max_tilt_deg: float = 15.0, hint: float | None = None) -> Table:
    """The near-horizontal plane with the most inliers (within 0.15 m of `hint` when given), refined by least
    squares; height is the inliers' mean z."""
    pts = points
    if hint is not None:
        pts = pts[np.abs(pts[:, 2] - hint) < 0.15]
    if len(pts) < 3:
        raise ValueError("table: not enough points")
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm * np.sign(n[2] or 1.0)
        if n[2] < cos_tilt:
            continue
        inl = np.abs((pts - a) @ n) < thresh
        cnt = int(inl.sum())
        if cnt > best_n:
            best, best_n = inl, cnt
    if best is None:
        raise ValueError("table: no near-horizontal plane found")
    P = pts[best]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] * np.sign(vt[2, 2] or 1.0)
    inl = np.abs((pts - c) @ n) < thresh
    P = pts[inl]
    ext = np.array([np.percentile(P[:, 0], [1, 99]), np.percentile(P[:, 1], [1, 99])])
    return Table(height=float(P[:, 2].mean()), normal=n, extent=ext)


def above_table(points: np.ndarray, table: Table, min_height: float = 0.006) -> np.ndarray:
    """Points higher than `min_height` above the table plane."""
    c = np.array([0.0, 0.0, table.height])
    return points[(points - c) @ np.asarray(table.normal, float) > min_height]


# ---------------------------------------------------------------------- segmenters
class Segmenter(Protocol):
    def segment(self, view: View, labels: Sequence[str]) -> list[Mask]:
        """Instance masks answering each text label."""
        ...


_STOP = {"the", "a", "an", "of", "on", "in", "to", "and", "with", "object", "thing"}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))
    out = []
    for w in words:
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def label_score(query: str, label: str) -> float:
    """1.0 when the content words are equal; else their Jaccard overlap (0 when none shared)."""
    a, b = tokens(query), tokens(label)
    if not a or not b:
        return 0.0
    if a == b or set(a) == set(b):
        return 1.0
    return len(set(a) & set(b)) / len(set(a) | set(b))


class EnvSegmenter:
    """The environment's instance segmentation (`<cam>.seg` with `extra["segments"]` labels). A query matches the
    segment labels with equal content words; failing that, those sharing the most words (score = overlap)."""

    def __init__(self, min_pixels: int = 15) -> None:
        self.min_pixels = min_pixels

    def segment(self, view: View, labels: Sequence[str]) -> list[Mask]:
        out = []
        for q in labels:
            scores = {i: label_score(q, name) for i, name in view.segments.items()}
            top = max(scores.values(), default=0.0)
            if top <= 0:
                continue
            for name, cam in view.cams.items():
                if cam.seg is None:
                    continue
                for i, s in scores.items():
                    if s == top:
                        m = cam.seg == i
                        if m.sum() >= self.min_pixels:
                            out.append(Mask(label=q, camera=name, mask=m, score=float(s), instance=int(i)))
        return out


# hue ranges in degrees, with saturation / value bounds: (h_lo, h_hi, s_min, s_max, v_min, v_max)
COLOURS: dict[str, tuple[float, float, float, float, float, float]] = {
    "red": (345, 15, 0.45, 1, 0.25, 1), "orange": (15, 40, 0.45, 1, 0.6, 1), "brown": (10, 45, 0.35, 1, 0.15, 0.6),
    "yellow": (40, 70, 0.45, 1, 0.35, 1), "green": (70, 165, 0.35, 1, 0.2, 1), "cyan": (165, 195, 0.35, 1, 0.3, 1),
    "blue": (195, 255, 0.35, 1, 0.2, 1), "purple": (255, 290, 0.3, 1, 0.2, 1), "pink": (290, 345, 0.3, 1, 0.3, 1),
    "magenta": (290, 345, 0.3, 1, 0.3, 1), "white": (0, 360, 0, 0.15, 0.8, 1), "black": (0, 360, 0, 1, 0, 0.18),
    "gray": (0, 360, 0, 0.15, 0.25, 0.8), "grey": (0, 360, 0, 0.15, 0.25, 0.8),
}


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rm, gm = nz & (mx == r), nz & (mx == g) & (mx != r)
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    return h * 60.0, np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0), mx


class ColourSegmenter:
    """Named colours in HSV (red, orange, brown, yellow, green, cyan, blue, purple, pink, magenta, white, black,
    gray): a label's colour word selects pixels, split into connected components (one Mask each, score 1.0). Other
    words of the label are ignored, so "red block" and "red bowl" both match every red region."""

    def __init__(self, min_pixels: int = 30) -> None:
        self.min_pixels = min_pixels

    def segment(self, view: View, labels: Sequence[str]) -> list[Mask]:
        out = []
        for q in labels:
            colour = next((w for w in tokens(q) if w in COLOURS), None)
            if colour is None:
                continue
            h0, h1, s0, s1, v0, v1 = COLOURS[colour]
            for name, cam in view.cams.items():
                h, s, v = hsv(cam.rgb)
                hm = (h >= h0) | (h <= h1) if h0 > h1 else (h >= h0) & (h <= h1)
                sel = hm & (s >= s0) & (s <= s1) & (v >= v0) & (v <= v1)
                lab, n = ndimage.label(sel)
                for i in range(1, n + 1):
                    m = lab == i
                    if m.sum() >= self.min_pixels:
                        out.append(Mask(label=q, camera=name, mask=m, score=1.0))
        return out


class MarkSegmenter:
    """A vision-language model picks among numbered regions (set-of-marks). `propose(view)` gives candidate regions
    per camera ({camera: [mask, ...]}; default: connected regions standing above the table in depth); `choose(rgb,
    regions, label)` is the model adapter: it draws the numbers, asks the model and returns the chosen indices."""

    def __init__(self, choose: Callable[[np.ndarray, list[np.ndarray], str], list[int]],
                 propose: Callable[[View], dict[str, list[np.ndarray]]] | None = None, min_pixels: int = 20) -> None:
        self.choose, self.propose, self.min_pixels = choose, propose or self.depth_regions, min_pixels

    def depth_regions(self, view: View) -> dict[str, list[np.ndarray]]:
        table = ransac_table(view.points, np.random.default_rng(0))
        out = {}
        for name, cam in view.cams.items():
            if cam.depth is None:
                continue
            H, W = cam.depth.shape
            pix = np.zeros((H, W), bool)
            z = np.asarray(cam.depth, float)
            v, u = np.nonzero(np.isfinite(z) & (z > 0) & (z < 4.0))
            pts = backproject(cam, max_depth=4.0)
            high = (pts - [0, 0, table.height]) @ table.normal > 0.01
            pix[v[high], u[high]] = True
            lab, n = ndimage.label(pix)
            out[name] = [lab == i for i in range(1, n + 1) if (lab == i).sum() >= self.min_pixels]
        return out

    def segment(self, view: View, labels: Sequence[str]) -> list[Mask]:
        regions = self.propose(view)
        out = []
        for q in labels:
            for name, masks in regions.items():
                for i in self.choose(view.cams[name].rgb, masks, q):
                    if 0 <= i < len(masks):
                        out.append(Mask(label=q, camera=name, mask=masks[i], score=1.0))
        return out


# ---------------------------------------------------------------------- objects and shapes
def detect_objects(view: View, masks: Sequence[Mask], table: Table, voxel: float = 0.004,
                   merge_radius: float = 0.015, min_points: int = 15) -> list[Detection]:
    """Per label (and instance when known): masked depth back-projected (mask eroded by one pixel against edge
    pixels), points on or under the table and statistical outliers dropped (a flat marking on the table keeps its
    points within 1 cm of the plane), cameras merged, then split into 3D connected components: one Detection each,
    largest first."""
    groups: dict[tuple[str, int | None], list[tuple[np.ndarray, float]]] = {}
    for m in masks:
        cam = view.cams[m.camera]
        mk = m.mask
        if mk.sum() > 40:
            mk = ndimage.binary_erosion(mk)
        raw = backproject(cam, mk)
        pts = above_table(raw, table)
        if len(pts) < min_points:                    # a flat marking on the table (a zone, a label): its own points
            h = (raw - [0.0, 0.0, table.height]) @ np.asarray(table.normal, float)
            flat = raw[(h > -0.004) & (h < 0.01)]
            pts = flat if len(flat) >= min_points else pts
        if len(pts):
            groups.setdefault((m.label, m.instance), []).append((remove_outliers(pts), m.score))
    out = []
    for (label, _), parts in groups.items():
        pts = voxel_down(np.concatenate([p for p, _ in parts]), voxel)
        score = float(np.mean([s for _, s in parts]))
        for idx in components(pts, merge_radius):
            if len(idx) >= min_points:
                out.append(Detection(label=label, score=score, points=pts[idx]))
    return sorted(out, key=lambda d: (d.label, -len(d.points)))


def fit_box(points: np.ndarray, table: Table | None = None) -> Box:
    """Minimum-area rectangle of the xy footprint (convex hull, rotating edges; extents from the 1st and 99th
    percentiles), z from the top of the points down to the table (or the lowest point without one)."""
    pts = np.asarray(points, float)
    xy = pts[:, :2]
    best = (np.inf, 0.0)
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        hull = xy
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    for th in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6)):
        r = hull @ rotz(-th)[:2, :2].T
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best[0]:
            best = (area, th)
    th = best[1]
    r = xy @ rotz(-th)[:2, :2].T
    lo, hi = np.percentile(r, 1, axis=0), np.percentile(r, 99, axis=0)      # robust to depth noise
    size_xy, mid = hi - lo, (lo + hi) / 2
    if size_xy[1] > size_xy[0]:
        th += np.pi / 2
        size_xy = size_xy[::-1]
    th = (th + np.pi / 2) % np.pi - np.pi / 2
    if th <= -np.pi / 2 + 1e-9:
        th += np.pi
    c_xy = rotz(best[1])[:2, :2] @ mid
    top = float(np.percentile(pts[:, 2], 99))
    bottom = float(table.height) if table is not None else float(pts[:, 2].min())
    return Box(center=np.array([c_xy[0], c_xy[1], (top + bottom) / 2]), size=np.array([*size_xy, top - bottom]),
               yaw=float(th), bottom=bottom)


def _kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rigid T minimising |T A - B|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return pose(R, cb - R @ ca)


def icp(model: np.ndarray, observed: np.ndarray, init: np.ndarray | None = None, iters: int = 40,
        tol: float = 0.005, yaw_steps: int = 12) -> Fit:
    """Point-to-point ICP of `model` onto `observed` (each observed point to its nearest model point, so a partial
    view fits). Without `init`: centroids aligned and `yaw_steps` yaws about +z tried, the best refined."""
    model, observed = np.asarray(model, float), np.asarray(observed, float)

    def run(T: np.ndarray, n: int) -> tuple[np.ndarray, float]:
        err = np.inf
        for _ in range(n):
            moved = apply(T, model)
            d, j = cKDTree(moved).query(observed)
            keep = d < max(np.percentile(d, 90), tol)
            T = _kabsch(model[j[keep]], observed[keep])
            new = float(np.sqrt(np.mean(d[keep] ** 2)))
            if abs(err - new) < 1e-7:
                break
            err = new
        return T, err

    if init is not None:
        starts = [np.asarray(init, float)]
    else:
        cm, co = model.mean(0), observed.mean(0)
        starts = [pose(rotz(a), co - rotz(a) @ cm) for a in np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False)]
    scored = sorted((run(T, 8) for T in starts), key=lambda r: r[1])
    T, _ = run(scored[0][0], iters)
    d, _ = cKDTree(apply(T, model)).query(observed)
    inl = d < tol
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf")
    return Fit(T=T, rmse=rmse, inliers=float(inl.mean()))


__all__ = ["ColourSegmenter", "EnvSegmenter", "MarkSegmenter", "Segmenter", "backproject", "components",
           "detect_objects", "fit_box", "fuse", "icp", "label_score", "ransac_table", "remove_outliers"]
