"""Perception for replay-to-sim (RFC-0006 part 8): lift a recording to 3D, track objects, detect contact events,
register the camera to the robot and retarget a human hand to a parallel gripper.

Frames. Tracking runs in a "frame" F with +z up and the table at z = table_z: the replay world itself when the
recording knows its camera pose (our episodes), otherwise the gravity-levelled camera frame (GeoCalib roll / pitch,
refined by the RANSAC table normal) shifted so the table is at z = 0. `register` then finds the similarity
(scale, yaw about +z, translation) from F to the robot's world by aligning the carried object's track with the TCP
path while the gripper holds it (the constant hand-object offset is absorbed by the translation).

Tracking (per object): a model point set from the first frame's mask; per frame the mask is propagated by
projecting the model at the predicted pose (a box around it), keeping pixels that match the object's colour
histogram (when it is discriminative), lie above the table and near the predicted surface, and are closer to this
object than to any other; the pose is refined by trimmed ICP (translation + yaw by default, or full 6-DoF). A frame
whose fit fails keeps the prediction and is marked lost.

Events: closed-gripper intervals from the opening signal (hysteresis relative to the open level); grasp (with the
object that moves in the interval), lift (2 cm above its rest height), release, place (at rest after the release).
The same detector runs on the simulation, so recorded and simulated events compare like for like.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from hylo.manip.perception import components, ransac_table, remove_outliers

UP = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------------------------------------ small geometry
def pose(R: np.ndarray | None = None, p: Any = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return pose(R.T, -R.T @ p)


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, float).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * (1.0 / (1.0 + c))


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None,
                max_depth: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points of the valid depth pixels (inside `mask`) and their (row, col) indices."""
    z = np.asarray(depth, float)
    valid = np.isfinite(z) & (z > 1e-4) & (z < max_depth)
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = z[v, u]
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return p, np.stack([v, u], axis=1)


def project(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.maximum(pts_cam[:, 2], 1e-6)
    return np.stack([K[0, 0] * pts_cam[:, 0] / z + K[0, 2], K[1, 1] * pts_cam[:, 1] / z + K[1, 2]], 1), pts_cam[:, 2]


# ------------------------------------------------------------------------------------------------ labels (VLM)
LABEL_PROMPT = """This is the first frame of a video in which {who} manipulates objects on a table.
{task}List the distinct objects on the table that are handled in the video or are the place where something is put
(at most {n}). Use short noun phrases with a colour or material when it helps an open-vocabulary segmenter tell them
apart (for example "red block", "white plate", "black bowl"). Do not list the robot, the gripper, the hand, the arm,
the table or the background. Reply with a JSON list of strings only."""

STOP_WORDS = {"the", "a", "an", "it", "on", "in", "into", "inside", "of", "to", "and", "then", "put", "place", "pick",
              "up", "move", "take", "from", "top", "left", "right", "front", "back", "next", "both", "its", "onto"}


def labels_from_text(text: str, n: int = 4) -> list[str]:
    """Deterministic fallback: noun phrases of the instruction, the words after 'the' up to a preposition."""
    out: list[str] = []
    for m in re.finditer(r"\bthe ([a-z][a-z ]*?)(?= (?:on|in|into|inside|to|and|from|next|onto|of|then)\b|[.,]|$)",
                         text.lower()):
        phrase = " ".join(w for w in m.group(1).split() if w not in STOP_WORDS)
        if phrase and phrase not in out:
            out.append(phrase)
    return out[:n]


def propose_labels(model: Any, rgb: np.ndarray, instruction: str, who: str = "a robot", n: int = 4) -> dict[str, Any]:
    """Ask a vision chat model for segmenter labels; falls back to the instruction's noun phrases."""
    fallback = labels_from_text(instruction, n)
    if model is None:
        return {"labels": fallback, "by": "instruction"}
    task = f'The task is: "{instruction}".\n' if instruction else ""
    try:
        reply = model.chat([{"role": "user", "content": LABEL_PROMPT.format(who=who, task=task, n=n),
                             "images": [np.asarray(rgb)]}], max_tokens=400)
    except Exception as exc:  # noqa: BLE001 - the deterministic labels still work
        return {"labels": fallback, "by": "instruction", "error": str(exc)[:300]}
    text = re.sub(r"<think>.*?</think>", "", reply.text or "", flags=re.S)
    m = re.search(r"\[.*?\]", text, re.S)
    labels: list[str] = []
    if m:
        import ast

        for parse in (json.loads, ast.literal_eval):          # models also answer with Python-style quotes
            try:
                labels = [str(x).strip().lower() for x in parse(m.group(0)) if str(x).strip()]
                break
            except (ValueError, SyntaxError, TypeError):
                labels = []
    bad = ("robot", "gripper", "hand", "arm", "table", "background")
    labels = [lab for lab in labels if not any(b == w for w in lab.split() for b in bad)][:n]
    # objects the instruction names are relevant by definition: add those the model left out
    extra = [f for f in fallback if labels and not any(set(f.split()) & set(lab.split()) for lab in labels)]
    labels = labels + extra
    return {"labels": labels or fallback, "from_instruction": extra, "by": "vlm" if labels else "instruction", "reply": text[:500],
            **reply.accounting()}


ALT_PROMPT = """An open-vocabulary segmenter found nothing for these labels in this image: {missing}.
For each, give up to three other short names a segmenter might know for the same object as it looks here (for
example "dish" or "pot" for a bowl seen from above). Reply with a JSON object mapping each label to a list."""


def alternative_labels(model: Any, rgb: np.ndarray, missing: Sequence[str]) -> dict[str, list[str]]:
    """Other names for labels the segmenter missed: from the vision model, else the label's last word and generic
    container / object words."""
    generic = {lab: [lab.split()[-1], "object"] if len(lab.split()) > 1 else ["object"] for lab in missing}
    if model is None or not missing:
        return generic
    try:
        reply = model.chat([{"role": "user", "content": ALT_PROMPT.format(missing=json.dumps(list(missing))),
                             "images": [np.asarray(rgb)]}], max_tokens=400)
        text = re.sub(r"<think>.*?</think>", "", reply.text or "", flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            import ast

            for parse in (json.loads, ast.literal_eval):
                try:
                    alts = parse(m.group(0))
                    return {lab: [str(x).lower() for x in alts.get(lab, [])][:3] or generic[lab] for lab in missing}
                except (ValueError, SyntaxError, TypeError, AttributeError):
                    continue
    except Exception:  # noqa: BLE001 - generic alternatives still work
        pass
    return generic


# ------------------------------------------------------------------------------------------------ colour model
class ColourModel:
    """Object-versus-background RGB histograms (8 bins per channel) from the first frame."""

    def __init__(self, rgb: np.ndarray, mask: np.ndarray, background: np.ndarray, bins: int = 8) -> None:
        self.bins = bins
        q = self._q(rgb)
        n = bins ** 3
        h_obj = np.bincount(q[mask], minlength=n).astype(float) + 0.5
        h_bg = np.bincount(q[background], minlength=n).astype(float) + 0.5
        self.p = (h_obj / h_obj.sum()) / (h_obj / h_obj.sum() + h_bg / h_bg.sum())
        self.discriminative = float(np.mean(self.p[q[mask]] > 0.5)) > 0.6 and float(np.mean(self.p[q[background]] > 0.5)) < 0.3
        self.mean = np.asarray(rgb[mask].mean(0), float) if mask.any() else np.full(3, 128.0)

    def _q(self, rgb: np.ndarray) -> np.ndarray:
        c = (np.asarray(rgb, np.int32) * self.bins) // 256
        return (c[..., 0] * self.bins + c[..., 1]) * self.bins + c[..., 2]

    def prob(self, rgb: np.ndarray) -> np.ndarray:
        return self.p[self._q(rgb)]


# ------------------------------------------------------------------------------------------------ lifting
@dataclass
class Lift:
    """Depth and camera in the tracking frame F (+z up). depth (T, H, W) camera metres; K (3, 3); T_frame_cam
    rigid; table_z, table_extent [[x0, x1], [y0, y1]] in F; frame "world" (F is the replay world) or "level"."""
    depth: np.ndarray
    K: np.ndarray
    T_frame_cam: np.ndarray
    table_z: float
    table_extent: np.ndarray
    frame: str
    info: dict[str, Any] = field(default_factory=dict)

    def points(self, t: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        p, pix = backproject(self.depth[t], self.K, mask)
        return apply(self.T_frame_cam, p), pix


def normalise_depth_scale(depths: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame monocular depth rescaled to frame 0 on `static` pixels (median ratio, robust to moving parts)."""
    out = depths.astype(np.float32).copy()
    ratios = np.ones(len(depths))
    d0 = depths[0]
    for t in range(1, len(depths)):
        ok = static & (d0 > 0) & (depths[t] > 0) & np.isfinite(depths[t])
        if ok.sum() < 50:
            continue
        r = d0[ok] / depths[t][ok]
        med = np.median(r)
        r = r[np.abs(r - med) < 0.1 * med]
        ratios[t] = float(np.median(r)) if len(r) > 20 else float(med)
        out[t] *= ratios[t]
    return out, ratios


def fit_plane(pts: np.ndarray, rng: np.random.Generator, thresh: float, up_hint: np.ndarray | None = None,
              iters: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dominant plane facing the camera (at the origin): unit normal towards the camera, a point on it and the
    inlier mask.
    Planes far from `up_hint` (> 40 degrees) count 0.6 of their inliers: gravity from one image is a hint (it can be
    far off for a camera looking straight down), the table is the big plane."""
    best, best_score = None, -1.0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if n @ (-a) < 0:
            n = -n
        inl = np.abs((pts - a) @ n) < thresh
        score = float(inl.sum())
        if up_hint is not None and float(n @ up_hint) < math.cos(math.radians(40)):
            score *= 0.6
        if score > best_score:
            best, best_score = (n, inl), score
    n, inl = best
    P = pts[inl]
    c = P.mean(0)
    n2 = np.linalg.svd(P - c, full_matrices=False)[2][2]
    n2 = n2 if n2 @ (-c) > 0 else -n2
    return n2, c, np.abs((pts - c) @ n2) < thresh


def level_frame(depth0: np.ndarray, K: np.ndarray, R_level_cam: np.ndarray | None,
                rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    """T_frame_cam for a camera with unknown pose: the table plane's normal is +z (GeoCalib's gravity picks the
    plane and fixes the heading), the table at z = 0. Returns (T_frame_cam, table_z, extent, info)."""
    R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]) if R_level_cam is None else np.asarray(R_level_cam, float)
    p, _ = backproject(depth0, K)
    sel = p[rng.choice(len(p), min(len(p), 30000), replace=False)]
    thresh = 0.01 * max(1.0, float(np.median(p[:, 2])))
    n_cam, c_cam, _ = fit_plane(sel, rng, thresh, up_hint=R.T @ UP)
    n_level = R @ n_cam
    tilt = math.degrees(math.acos(np.clip(float(n_level @ UP), -1, 1)))
    R2 = rot_between(n_level, UP) @ R
    pl = p @ R2.T
    inl = np.abs((p - c_cam) @ n_cam) < thresh
    zt = float(np.median(pl[inl, 2])) if inl.sum() > 30 else float(np.percentile(pl[:, 2], 10))
    T = pose(R2, [0.0, 0.0, -zt])
    pts = apply(T, p)
    on = np.abs(pts[:, 2]) < 0.015
    ext = np.array([np.percentile(pts[on, 0], [2, 98]), np.percentile(pts[on, 1], [2, 98])]) if on.sum() > 30 else \
        np.array([[-0.5, 0.5], [-0.5, 0.5]])
    return T, 0.0, ext, {"gravity_refined_by_table_deg": round(tilt, 2), "table_inliers": int(on.sum()),
                         "camera_height": round(-zt, 4)}


def table_in_world(depth0: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, rng: np.random.Generator,
                   hint: float | None = None) -> tuple[float, np.ndarray]:
    p, _ = backproject(depth0, K)
    pts = apply(T_world_cam, p)
    sel = pts[rng.choice(len(pts), min(len(pts), 40000), replace=False)]
    table = ransac_table(sel, rng, hint=hint)
    return float(table.height), np.asarray(table.extent if table.extent is not None else [[-0.5, 0.5], [-0.5, 0.5]])


# ------------------------------------------------------------------------------------------------ object models
def object_points(lift: Lift, mask: np.ndarray, t: int = 0, min_height: float = 0.004) -> np.ndarray:
    """Masked points of one object in F (mask eroded, table and outliers removed, largest 3D component)."""
    m = ndimage.binary_erosion(mask, iterations=1) if mask.sum() > 60 else mask
    pts, _ = lift.points(t, m)
    pts = pts[pts[:, 2] > lift.table_z + min_height]
    if len(pts) < 10:
        return pts
    pts = remove_outliers(pts)
    groups = components(pts, 0.012)
    return pts[groups[0]] if groups else pts


def box_extent(points: np.ndarray, table_z: float) -> dict[str, Any]:
    """Footprint (minimum-area rectangle of the xy hull) and height down to the table."""
    xy = points[:, :2]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:  # noqa: BLE001 - degenerate footprint
        hull = xy
    best = (np.inf, 0.0, None)
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    for th in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6)):
        r = hull @ rotz(-th)[:2, :2].T
        area = float(np.ptp(r[:, 0]) * np.ptp(r[:, 1]))
        if area < best[0]:
            best = (area, th, r)
    th = best[1]
    r = xy @ rotz(-th)[:2, :2].T
    lo, hi = np.percentile(r, 2, axis=0), np.percentile(r, 98, axis=0)
    centre_xy = rotz(th)[:2, :2] @ ((lo + hi) / 2)
    top = float(np.percentile(points[:, 2], 98))
    return {"size": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), top - table_z], "yaw": float(th),
            "centre": [float(centre_xy[0]), float(centre_xy[1]), (top + table_z) / 2]}


# ------------------------------------------------------------------------------------------------ ICP
def icp(model: np.ndarray, observed: np.ndarray, init: np.ndarray, mode: str = "yaw", iters: int = 25,
        max_dist: float = 0.03, trim: float = 85.0) -> tuple[np.ndarray, float, float]:
    """Trimmed point-to-point ICP of `model` (object frame) onto `observed`, starting at `init` (object -> F).
    mode "xyz" (translation), "yaw" (translation + rotation about +z) or "full". Returns (T, rmse, inlier fraction
    within 1 cm)."""
    T = np.asarray(init, float).copy()
    tree = None
    for _ in range(iters):
        moved = apply(T, model)
        tree = cKDTree(moved)
        d, j = tree.query(observed)
        keep = d < min(max_dist, max(np.percentile(d, trim), 0.004))
        if keep.sum() < 6:
            break
        A, B = moved[j[keep]], observed[keep]
        ca, cb = A.mean(0), B.mean(0)
        if mode == "xyz":
            dR = np.eye(3)
        elif mode == "yaw":
            a, b = A[:, :2] - ca[:2], B[:, :2] - cb[:2]
            ang = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a * b)))
            dR = rotz(ang)
        else:
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            dR = Vt.T @ D @ U.T
        step = pose(dR, cb - dR @ ca)
        T = step @ T
        if np.linalg.norm(step[:3, 3]) < 1e-5 and abs(np.trace(dR) - 3) < 1e-8:
            break
    d, _ = cKDTree(apply(T, model)).query(observed)
    inl = d < 0.01
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf")
    return T, rmse, float(inl.mean())


# ------------------------------------------------------------------------------------------------ tracking
@dataclass
class Track:
    name: str
    label: str
    model: np.ndarray                 # (N, 3) object frame (origin at the first-frame centroid, axes of F)
    poses: np.ndarray                 # (T, 4, 4) object -> F
    lost: np.ndarray                  # (T,) bool
    inliers: np.ndarray               # (T,)
    npts: np.ndarray                  # (T,)
    colour: list[float]
    mask0: np.ndarray | None = None
    extent: dict[str, Any] = field(default_factory=dict)

    def positions(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    def summary(self) -> dict[str, Any]:
        p = self.positions()
        return {"name": self.name, "label": self.label, "model_points": int(len(self.model)),
                "moved_m": round(float(np.max(np.linalg.norm(p - p[0], axis=1))), 4),
                "lost_frames": int(self.lost.sum()), "frames": int(len(p)),
                "mean_inliers": round(float(np.mean(self.inliers[~self.lost])) if (~self.lost).any() else 0.0, 3),
                "extent": self.extent, "colour": [round(c) for c in self.colour]}


def init_tracks(lift: Lift, rgb0: np.ndarray, masks: Sequence[tuple[str, np.ndarray]],
                min_points: int = 25) -> list[Track]:
    allm = np.zeros(rgb0.shape[:2], bool)
    for _, m in masks:
        allm |= m
    bg = ~ndimage.binary_dilation(allm, iterations=3)
    tracks, used = [], set()
    for label, m in masks:
        pts = object_points(lift, m)
        if len(pts) < min_points:
            continue
        name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"
        k, base = 1, name
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        c = pts.mean(0)
        T0 = pose(None, c)
        cm = ColourModel(rgb0, m, bg)
        tr = Track(name, label, pts - c, np.repeat(T0[None], len(lift.depth), 0), np.zeros(len(lift.depth), bool),
                   np.ones(len(lift.depth)), np.full(len(lift.depth), len(pts)), cm.mean.tolist(), m,
                   box_extent(pts, lift.table_z))
        tr._cm = cm  # type: ignore[attr-defined]
        tracks.append(tr)
    return tracks


def track_objects(lift: Lift, rgb: np.ndarray, tracks: list[Track], *, mode: str = "yaw",
                  prior: Callable[[str, int], np.ndarray | None] | None = None, gate: float = 0.04,
                  max_speed: float = 1.0, fps: float = 10.0, log: Callable[[str], None] | None = None) -> list[Track]:
    """Track every object through the video (see the module doc). `prior(name, t)` may give a predicted pose (e.g.
    attached to the gripper while it is closed)."""
    H, W = rgb.shape[1:3]
    Tcf = inv(lift.T_frame_cam)
    n = len(rgb)
    vel = {tr.name: np.zeros(3) for tr in tracks}
    for t in range(1, n):
        preds = {}
        for tr in tracks:
            P = tr.poses[t - 1].copy()
            pr = prior(tr.name, t) if prior is not None else None
            if pr is not None:
                P = pr
            else:
                P[:3, 3] += np.clip(vel[tr.name], -max_speed / fps, max_speed / fps)
            preds[tr.name] = P
        pts_all, pix_all = lift.points(t)
        keep_z = pts_all[:, 2] > lift.table_z + 0.004
        owners = None
        if len(tracks) > 1:
            trees = [cKDTree(apply(preds[tr.name], tr.model)) for tr in tracks]
            dists = np.stack([tree.query(pts_all, distance_upper_bound=0.2)[0] for tree in trees], 1)
            owners = np.argmin(dists, 1)
        for k, tr in enumerate(tracks):
            P = preds[tr.name]
            model_f = apply(P, tr.model)
            uv, z = project(apply(Tcf, model_f), lift.K)
            ok = (z > 0)
            if not ok.any():
                tr.poses[t], tr.lost[t] = P, True
                continue
            u0, v0 = uv[ok].min(0)
            u1, v1 = uv[ok].max(0)
            mgn = max(8.0, 0.4 * max(u1 - u0, v1 - v0))
            u0, v0, u1, v1 = u0 - mgn, v0 - mgn, u1 + mgn, v1 + mgn
            sel = keep_z & (pix_all[:, 1] >= u0) & (pix_all[:, 1] <= u1) & (pix_all[:, 0] >= v0) & (pix_all[:, 0] <= v1)
            if owners is not None:
                sel &= owners == k
            cm = getattr(tr, "_cm", None)
            if cm is not None and cm.discriminative and sel.any():
                pr = cm.prob(rgb[t][pix_all[sel, 0], pix_all[sel, 1]])
                idx = np.flatnonzero(sel)
                sel[idx[pr < 0.5]] = False
            obs = pts_all[sel]
            if len(obs):
                d, _ = cKDTree(model_f).query(obs)
                spd = float(np.linalg.norm(vel[tr.name]))
                obs = obs[d < gate + 1.5 * spd]
            need = max(12, int(0.15 * len(tr.model)))
            if len(obs) < need:
                tr.poses[t], tr.lost[t], tr.npts[t], tr.inliers[t] = P, True, len(obs), 0.0
                continue
            if len(obs) > 3000:
                obs = obs[np.random.default_rng(t).choice(len(obs), 3000, replace=False)]
            Tn, rmse, inl = icp(tr.model, obs, P, mode=mode, max_dist=gate)
            good = inl > 0.35 and rmse < 0.012 and np.linalg.norm(Tn[:3, 3] - P[:3, 3]) < gate + 0.05
            tr.poses[t] = Tn if good else P
            tr.lost[t], tr.npts[t], tr.inliers[t] = not good, len(obs), inl
            if good:
                v = Tn[:3, 3] - tr.poses[t - 1][:3, 3]
                vel[tr.name] = 0.5 * vel[tr.name] + 0.5 * v
        if log and t % 25 == 0:
            log(f"track: frame {t}/{n}")
    return tracks


def smooth_poses(poses: np.ndarray, lost: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving-average positions (rotations kept), lost frames interpolated between good ones."""
    out = poses.copy()
    p = poses[:, :3, 3].copy()
    good = np.flatnonzero(~lost)
    if len(good) >= 2:
        for k in range(3):
            p[:, k] = np.interp(np.arange(len(p)), good, p[good, k])
    if window > 1:
        ker = np.ones(window) / window
        pad = np.pad(p, ((window // 2, window // 2), (0, 0)), mode="edge")
        p = np.stack([np.convolve(pad[:, k], ker, mode="valid") for k in range(3)], 1)[: len(p)]
    out[:, :3, 3] = p
    return out


# ------------------------------------------------------------------------------------------------ events
def closed_intervals(gripper: np.ndarray, *, close_frac: float = 0.8, open_frac: float = 0.9,
                     min_len: int = 2, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Frames [start, end) where the gripper is closed, with hysteresis relative to the open level (90th
    percentile of the signal); intervals separated by at most `merge_gap` open frames are merged (noisy signals
    such as a hand's aperture)."""
    g = np.asarray(gripper, float)
    if not len(g):
        return []
    level = max(float(np.percentile(g, 90)), 1e-3)
    out, start, closed = [], 0, False
    for t, x in enumerate(g):
        if not closed and x < close_frac * level:
            closed, start = True, t
        elif closed and x > open_frac * level:
            closed = False
            if t - start >= min_len:
                out.append((start, t))
    if closed and len(g) - start >= min_len:
        out.append((start, len(g)))
    merged: list[tuple[int, int]] = []
    for a, b in out:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def detect_events(time: np.ndarray, gripper: np.ndarray | None, objects: dict[str, np.ndarray], *,
                  table_z: float = 0.0, lift_h: float = 0.02, move_eps: float = 0.015, rest_speed: float = 0.03,
                  intervals: list[tuple[int, int]] | None = None, closed_level: float = 0.04) -> list[dict[str, Any]]:
    """Contact events from the gripper opening and object positions {name: (T, 3)}: grasp, lift, release, place
    (and missed_grasp when the gripper closes on nothing)."""
    time = np.asarray(time, float)
    ivs = intervals if intervals is not None else (closed_intervals(gripper) if gripper is not None else [])
    events: list[dict[str, Any]] = []
    for a, b in ivs:
        b1 = min(b, len(time) - 1)
        moves = {n: float(np.max(np.linalg.norm(p[a:b1 + 1] - p[a], axis=1))) for n, p in objects.items()}
        name = max(moves, key=moves.get) if moves else None
        held = gripper is None or float(np.median(np.asarray(gripper)[a:b1 + 1])) > closed_level
        if name is None or moves[name] < move_eps or not held:
            events.append({"type": "missed_grasp", "t": float(time[a]), "frame": int(a), "object": name})
            continue
        p = objects[name]
        rest = float(np.median(p[max(0, a - 3):a + 1, 2]))
        events.append({"type": "grasp", "t": float(time[a]), "frame": int(a), "object": name})
        up = np.flatnonzero(p[a:b1 + 1, 2] > rest + lift_h)
        if len(up):
            events.append({"type": "lift", "t": float(time[a + up[0]]), "frame": int(a + up[0]), "object": name,
                           "height": round(float(p[a:b1 + 1, 2].max() - rest), 4)})
        events.append({"type": "release", "t": float(time[b1]), "frame": int(b1), "object": name})
        dt = np.diff(time).mean() if len(time) > 1 else 0.1
        spd = np.linalg.norm(np.diff(p, axis=0), axis=1) / max(dt, 1e-6)
        after = np.flatnonzero(spd[b1:] < rest_speed)
        if len(after):
            f = min(b1 + int(after[0]), len(time) - 1)
            events.append({"type": "place", "t": float(time[f]), "frame": int(f), "object": name,
                           "pos": [round(float(x), 4) for x in p[f]],
                           "moved": round(float(np.linalg.norm(p[f, :2] - p[a, :2])), 4)})
    order = {"missed_grasp": 0, "grasp": 1, "lift": 2, "release": 3, "place": 4}
    return sorted(events, key=lambda e: (e["t"], order[e["type"]]))


def match_events(ref: list[dict[str, Any]], sim: list[dict[str, Any]], tol_s: float = 1.0) -> dict[str, Any]:
    """Each reference grasp / lift / place matched to a simulated event of the same type and object within tol."""
    used, pairs = set(), []
    for e in ref:
        if e["type"] not in ("grasp", "lift", "place"):
            continue
        best = None
        for i, s in enumerate(sim):
            if i in used or s["type"] != e["type"] or s.get("object") != e.get("object"):
                continue
            dt = abs(s["t"] - e["t"])
            if dt <= tol_s and (best is None or dt < best[1]):
                best = (i, dt)
        if best is not None:
            used.add(best[0])
        pairs.append({"type": e["type"], "object": e.get("object"), "t": round(e["t"], 3),
                      "matched": best is not None, "dt": None if best is None else round(best[1], 3)})
    n = len(pairs)
    return {"matched": sum(p["matched"] for p in pairs), "of": n, "pairs": pairs,
            "extra_sim": [s for i, s in enumerate(sim) if i not in used and s["type"] in ("grasp", "lift", "place")]}


# ------------------------------------------------------------------------------------------------ registration
def register_similarity(obj_pos: np.ndarray, tcp_pos: np.ndarray, *, scale: bool = True) -> dict[str, Any]:
    """Least-squares s, yaw, t with tcp = s Rz(yaw) obj + t (both (N, 3), F and world)."""
    u, p = np.asarray(obj_pos, float), np.asarray(tcp_pos, float)
    uc, pc = u - u.mean(0), p - p.mean(0)
    th = math.atan2(float(np.sum(uc[:, 0] * pc[:, 1] - uc[:, 1] * pc[:, 0])), float(np.sum(uc[:, :2] * pc[:, :2])))
    R = rotz(th)
    s = float(np.sum((uc @ R.T) * pc) / max(np.sum(uc * uc), 1e-12)) if scale else 1.0
    s = max(s, 1e-3)
    t = p.mean(0) - s * R @ u.mean(0)
    res = p - (s * u @ R.T + t)
    return {"scale": s, "yaw": th, "t": t, "rmse": float(np.sqrt(np.mean(np.sum(res ** 2, 1)))), "pairs": int(len(u)),
            "path_m": float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0}


def similarity_matrix(reg: dict[str, Any]) -> np.ndarray:
    S = np.eye(4)
    S[:3, :3] = reg["scale"] * rotz(reg["yaw"])
    S[:3, 3] = reg["t"]
    return S


def carried_pairs(tracks: list[Track], tcp: np.ndarray, intervals: list[tuple[int, int]],
                  move_eps: float = 0.02) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(object positions in F, TCP positions) over the frames each closed interval carries an object."""
    objs, tcps, names = [], [], []
    for a, b in intervals:
        b1 = min(b, len(tcp))
        best, mv = None, move_eps
        for tr in tracks:
            p = tr.positions()[a:b1]
            m = float(np.max(np.linalg.norm(p - p[0], axis=1))) if len(p) else 0.0
            if m > mv:
                best, mv = tr, m
        if best is None:
            continue
        good = ~best.lost[a:b1]
        objs.append(best.positions()[a:b1][good])
        tcps.append(tcp[a:b1, :3, 3][good])
        names.append(best.name)
    if not objs:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    return np.concatenate(objs), np.concatenate(tcps), names


# ------------------------------------------------------------------------------------------------ hands
def anchor_grasps(tcp: np.ndarray, objects: dict[str, np.ndarray], events: list[dict[str, Any]],
                  intervals: list[tuple[int, int]], ramp: int = 5) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Grasp-anchored retargeting: a hand's thumb-index midpoint is not where a parallel gripper must close on the
    object (a wrap grasp holds the object beside the fingertips), so during each grasp the TCP path is shifted by the
    offset that puts it on the object's centre at the grasp moment, blended in and out over `ramp` frames. The hand's
    motion during the grasp is kept."""
    out = tcp.copy()
    T = len(tcp)
    notes = []
    for e in events:
        if e["type"] != "grasp" or e.get("object") not in objects:
            continue
        a = e["frame"]
        b = next((b for a0, b in intervals if a0 == a), min(T, a + 1))
        off = objects[e["object"]][a] - tcp[a][:3, 3]
        w = np.zeros(T)
        w[max(0, a - ramp):a] = np.linspace(0, 1, a - max(0, a - ramp) + 1)[1:] if a > 0 else []
        w[a:b] = 1.0
        w[b:min(T, b + ramp)] = np.linspace(1, 0, min(T, b + ramp) - b + 1)[:-1]
        out[:, :3, 3] += w[:, None] * off
        notes.append({"object": e["object"], "frame": int(a), "shift_m": np.round(off, 4).tolist()})
    return out, notes


THUMB_TIP, INDEX_TIP, WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 4, 8, 0, 5, 9, 17


def hand_scale(kp3d: np.ndarray, kp2d: np.ndarray, depth: np.ndarray, K: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Scale that makes the monocular depth agree with the hand's metric size (the hand model's palm, about
    9 cm wrist to middle knuckle): ratio of palm bone lengths, hand model over depth back-projection."""
    ratios = []
    for t in np.flatnonzero(valid):
        pts = []
        for j in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP):
            u, v = kp2d[t, j]
            vi, ui = int(round(v)), int(round(u))
            if not (0 <= vi < depth.shape[1] and 0 <= ui < depth.shape[2]):
                break
            win = depth[t, max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
            z = float(np.median(win[win > 0])) if (win > 0).any() else 0.0
            if z <= 0:
                break
            pts.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        if len(pts) < 4:
            continue
        pts = np.asarray(pts)
        bones = [(0, 1), (0, 2), (0, 3), (1, 3)]
        idx = [WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]
        for a, b in bones:
            lm = np.linalg.norm(kp3d[t, idx[a]] - kp3d[t, idx[b]])
            ld = np.linalg.norm(pts[a] - pts[b])
            if ld > 1e-4:
                ratios.append(lm / ld)
    if not ratios:
        return {"scale": 1.0, "samples": 0}
    r = np.asarray(ratios)
    med = float(np.median(r))
    return {"scale": med, "samples": int(len(r)), "iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))]}


def hand_keypoints_world(kp3d_cam: np.ndarray, kp2d: np.ndarray, valid: np.ndarray, depth: np.ndarray, K: np.ndarray,
                         T_frame_cam: np.ndarray, smooth: int = 5) -> dict[str, np.ndarray]:
    """The 21 hand keypoints per frame in F: each frame's metric hand (camera frame) shifted so its wrist and
    knuckles sit on the (scaled) depth map, as `hand_to_gripper` places it; missing frames interpolated, then a
    moving average over `smooth` frames. Returns {"kp": (T, 21, 3), "valid": (T,) frames with a detection}."""
    T = len(kp3d_cam)
    out = np.zeros((T, 21, 3))
    ok = np.zeros(T, bool)
    for t in range(T):
        if not valid[t]:
            continue
        k = np.asarray(kp3d_cam[t], float)
        anchor = []
        for j in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP):
            u, v = kp2d[t, j]
            vi, ui = int(round(v)), int(round(u))
            if 0 <= vi < depth.shape[1] and 0 <= ui < depth.shape[2]:
                win = depth[t, max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
                if (win > 0).any():
                    z = float(np.median(win[win > 0]))
                    anchor.append((j, np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])))
        if anchor:
            k = k + np.mean([p - k[j] for j, p in anchor], axis=0)
        out[t] = apply(T_frame_cam, k)
        ok[t] = np.isfinite(out[t]).all()
    good = np.flatnonzero(ok)
    if len(good) and len(good) < T:
        idx = np.arange(T)
        for j in range(21):
            for c in range(3):
                out[:, j, c] = np.interp(idx, good, out[good, j, c])
    if smooth > 1 and len(good):
        ker = np.ones(smooth) / smooth
        pad = np.pad(out, ((smooth // 2, smooth // 2), (0, 0), (0, 0)), mode="edge")
        out = np.stack([np.stack([np.convolve(pad[:, j, c], ker, "valid")[:T] for c in range(3)], 1)
                        for j in range(21)], 1)
    return {"kp": out, "valid": ok}


def hand_to_gripper(kp3d_cam: np.ndarray, kp2d: np.ndarray, valid: np.ndarray, depth: np.ndarray, K: np.ndarray,
                    T_frame_cam: np.ndarray, max_opening: float = 0.12, smooth: int = 5) -> dict[str, np.ndarray]:
    """Retarget one hand to a parallel-gripper TCP in F: origin between thumb and index tips, closing axis thumb to
    index, approach from the wrist towards the tips; opening = thumb-index aperture / max_opening. The hand is placed
    in depth by the (scaled) depth map at the wrist and knuckles; missing frames are interpolated."""
    T = len(kp3d_cam)
    tcp = np.repeat(np.eye(4)[None], T, 0)
    aperture = np.zeros(T)
    ok = np.zeros(T, bool)
    world = hand_keypoints_world(kp3d_cam, kp2d, valid, depth, K, T_frame_cam, smooth=1)
    for t in range(T):
        if not world["valid"][t]:
            continue
        kf = world["kp"][t]
        tip_t, tip_i, wrist = kf[THUMB_TIP], kf[INDEX_TIP], kf[WRIST]
        y = tip_i - tip_t
        ap = float(np.linalg.norm(y))
        y = y / max(ap, 1e-6)
        mid = (tip_t + tip_i) / 2
        x = mid - wrist
        x = x - (x @ y) * y
        if np.linalg.norm(x) < 1e-6:
            continue
        x /= np.linalg.norm(x)
        tcp[t] = pose(np.stack([x, y, np.cross(x, y)], 1), mid)
        aperture[t] = ap
        ok[t] = True
    good = np.flatnonzero(ok)
    if len(good) == 0:
        return {"tcp": tcp, "gripper": np.ones(T), "aperture": aperture, "valid": ok}
    idx = np.arange(T)
    for r in range(3):
        for c in range(4):
            tcp[:, r, c] = np.interp(idx, good, tcp[good, r, c])
    aperture = np.interp(idx, good, aperture[good])
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        for r in range(3):
            for c in range(4):
                tcp[:, r, c] = np.convolve(np.pad(tcp[:, r, c], smooth // 2, mode="edge"), ker, "valid")[:T]
        aperture = np.convolve(np.pad(aperture, smooth // 2, mode="edge"), ker, "valid")[:T]
    tcp[~np.isfinite(tcp)] = 0.0
    for t in range(T):                      # re-orthonormalise after averaging
        if abs(np.linalg.det(tcp[t, :3, :3])) < 1e-9:
            tcp[t, :3, :3] = np.eye(3)
        U, _, Vt = np.linalg.svd(tcp[t, :3, :3])
        tcp[t, :3, :3] = U @ Vt
    return {"tcp": tcp, "gripper": np.clip(aperture / max_opening, 0.0, 1.0), "aperture": aperture, "valid": ok}


__all__ = ["ColourModel", "Lift", "Track", "box_extent", "carried_pairs", "closed_intervals", "detect_events",
           "alternative_labels", "anchor_grasps", "hand_keypoints_world", "hand_scale", "hand_to_gripper", "icp", "init_tracks", "labels_from_text", "level_frame", "match_events",
           "normalise_depth_scale", "object_points", "propose_labels", "register_similarity", "similarity_matrix",
           "smooth_poses", "table_in_world", "track_objects"]
