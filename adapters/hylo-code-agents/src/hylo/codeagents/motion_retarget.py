"""Retargeting math shared by every target character (pure numpy, no Hylo imports): the MuJoCo worker
(motion_sim) uses it for robots, the Blender script (motion_blender) for rigged characters; both load this file by
path, Hylo code imports it as hylo.codeagents.motion_retarget.

Skeleton points are arrays (..., K, 3) indexed by `idx` (keypoint name -> column), metres, z up, facing +x.

- body_frames: the pelvis and chest frames of a skeleton (forward, left, up columns).
- proportional_targets: source keypoints -> target points with a character's proportions: the root path scaled by
  the leg-length ratio, each limb segment along the source's direction with the character's own length, hips and
  shoulders placed as on the character, turned with the source's pelvis and chest.
- lock_feet / smooth: clean-ups of the source before the fit (no sliding while a foot is down; video noise).
- segment_angles: limb direction error between two skeletons (a scale-free retargeting error).
- ground_correction: a per-frame height offset that keeps the lowest foot point on the floor while the source's
  feet are down, and never below it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

LIMBS = [("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeBase"), ("RightLeg", "RightShin"),
         ("RightShin", "RightFoot"), ("RightFoot", "RightToeBase"), ("LeftArm", "LeftForeArm"),
         ("LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand")]
# the target skeleton, parents first: (child, parent, how). "pelvis" / "chest": the character's rigid offset turned
# with that frame of the source; "dir": the source's direction with the character's segment length.
TREE = [("LeftLeg", "Hips", "pelvis"), ("RightLeg", "Hips", "pelvis"), ("Chest", "Hips", "dir"),
        ("Head", "Chest", "chest"), ("LeftArm", "Chest", "chest"), ("RightArm", "Chest", "chest"),
        *[(c, p, "dir") for p, c in LIMBS]]
FEET = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
Index = Mapping[str, int]


def unit(v: np.ndarray) -> np.ndarray:
    """v / |v| along the last axis (zero-safe)."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def body_frames(P: np.ndarray, idx: Index) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis and chest frames (..., 3, 3) of points P (..., K, 3); columns forward, left, up. Pelvis: left = right
    -> left hip, up = hips -> chest; chest: left = right -> left shoulder, up = chest -> mid-shoulders."""
    def frame(left: str, right: str, lo: str, hi: np.ndarray) -> np.ndarray:
        y = unit(P[..., idx[left], :] - P[..., idx[right], :])
        z = hi - P[..., idx[lo], :]
        z = unit(z - y * np.sum(z * y, -1, keepdims=True))
        return np.stack([np.cross(y, z), y, z], -1)
    mid = 0.5 * (P[..., idx["LeftArm"], :] + P[..., idx["RightArm"], :])
    return frame("LeftLeg", "RightLeg", "Hips", P[..., idx["Chest"], :]), frame("LeftArm", "RightArm", "Chest", mid)


def leg_length(P: np.ndarray, idx: Index) -> float:
    """Mean hip-knee-ankle length of both legs of P (K, 3) or (T, K, 3) (the median over frames)."""
    ls = [np.linalg.norm(P[..., idx[f"{s}Leg"], :] - P[..., idx[f"{s}Shin"], :], axis=-1)
          + np.linalg.norm(P[..., idx[f"{s}Shin"], :] - P[..., idx[f"{s}Foot"], :], axis=-1) for s in ("Left", "Right")]
    return float(np.median(0.5 * (ls[0] + ls[1])))


def proportional_targets(K: np.ndarray, idx: Index, rest: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Source keypoints K (T, K, 3) -> (target points with the character's proportions, root scale). `rest`:
    keypoint -> the character's rest position (facing +x), for every keypoint in idx."""
    R = np.stack([np.asarray(rest[n], float) for n in sorted(idx, key=idx.get)])
    s = leg_length(R, idx) / max(leg_length(K, idx), 1e-6)
    Fp, Fc = body_frames(K, idx)
    Rp, Rc = body_frames(R, idx)
    out = np.zeros_like(K, dtype=float)
    out[:, idx["Hips"]] = K[:, idx["Hips"]] * s
    for c, p, how in TREE:
        if c not in idx or p not in idx:
            continue
        off = R[idx[c]] - R[idx[p]]
        if how == "dir":
            out[:, idx[c]] = out[:, idx[p]] + np.linalg.norm(off) * unit(K[:, idx[c]] - K[:, idx[p]])
        else:
            F, F0 = (Fp, Rp) if how == "pelvis" else (Fc, Rc)
            out[:, idx[c]] = out[:, idx[p]] + np.einsum("tij,j->ti", F, F0.T @ off)
    return out, s


def uniform_scale(K: np.ndarray, idx: Index, rest: Mapping[str, np.ndarray]) -> float:
    """One scale for the whole skeleton: the character's leg + torso length over the source's (first frame)."""
    pairs = [("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("RightLeg", "RightShin"), ("RightShin", "RightFoot"),
             ("Hips", "Head")]
    h = sum(np.linalg.norm(K[0, idx[a]] - K[0, idx[b]]) for a, b in pairs)
    r = sum(np.linalg.norm(np.asarray(rest[a]) - np.asarray(rest[b])) for a, b in pairs)
    return float(r / h) if h > 1e-6 else 1.0


def smooth(K: np.ndarray, frames: int) -> np.ndarray:
    """A centred moving average over 2 * frames + 1 frames (edges padded); K unchanged when too short."""
    k = int(frames)
    if k < 1 or len(K) <= 2 * k + 1:
        return K
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    flat = np.pad(K.reshape(len(K), -1), ((k, k), (0, 0)), mode="edge")
    return np.stack([np.convolve(flat[:, c], ker, mode="valid") for c in range(flat.shape[1])], 1).reshape(K.shape)


def lock_feet(K: np.ndarray, idx: Index, tol: float = 0.025) -> np.ndarray:
    """While a foot keypoint stays near the ground (within `tol` of its low height), hold its horizontal position
    at the run's mean, eased over 3 frames at each end: removes the source's foot sliding."""
    K = K.copy()
    for n in FEET:
        if n not in idx:
            continue
        p = K[:, idx[n]]
        low = p[:, 2] < np.percentile(p[:, 2], 3) + tol
        t = 0
        while t < len(p):
            if not low[t]:
                t += 1
                continue
            e = t
            while e < len(p) and low[e]:
                e += 1
            if e - t >= 3:
                xy = p[t:e, :2].mean(0)
                w = np.ones(e - t)
                ramp = min(3, (e - t) // 2)
                if ramp:
                    w[:ramp] = w[-ramp:][::-1] = np.linspace(0.25, 1, ramp)
                p[t:e, :2] += w[:, None] * (xy - p[t:e, :2])
            t = e
    return K


def segment_angles(A: np.ndarray, B: np.ndarray, idx: Index) -> np.ndarray:
    """Angles in degrees (T, n) between the limb segments of point sets A and B (T, K, 3)."""
    ang = []
    for p, c in LIMBS:
        if p in idx and c in idx:
            u, v = unit(A[:, idx[c]] - A[:, idx[p]]), unit(B[:, idx[c]] - B[:, idx[p]])
            ang.append(np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1))))
    return np.stack(ang, 1)


def feet_height(P: np.ndarray, idx: Index) -> np.ndarray:
    """(T,) height of the lowest foot keypoint of P (T, K, 3)."""
    return np.stack([P[:, idx[n], 2] for n in FEET if n in idx], 1).min(1)


def ground_correction(low: np.ndarray, src_feet: np.ndarray, scale: float) -> np.ndarray:
    """(T,) height offsets: the lowest foot point (`low`) on the floor whenever the source's feet (`src_feet`, at
    target scale) are near the ground, interpolated in between (jumps keep their flight), never below the floor."""
    T = len(low)
    grounded = src_feet < np.percentile(src_feet, 10) + 0.04 * scale / 0.8
    corr = np.interp(np.arange(T), np.flatnonzero(grounded), -low[grounded])
    if T > 5:
        corr = np.convolve(np.pad(corr, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return np.maximum(corr, -low)


__all__ = ["FEET", "LIMBS", "TREE", "body_frames", "feet_height", "ground_correction", "leg_length", "lock_feet",
           "proportional_targets", "segment_angles", "smooth", "uniform_scale", "unit"]
