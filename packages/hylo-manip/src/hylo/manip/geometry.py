"""Small rigid-body helpers shared by the toolkit modules."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose(R: np.ndarray | None = None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return pose(R.T, -R.T @ p)


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


def rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis * angle of a rotation matrix."""
    return Rotation.from_matrix(R).as_rotvec()


def angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def yaw_of(R: np.ndarray) -> float:
    """Heading of the x axis projected on the xy plane (falls back to the y axis when x is vertical)."""
    v = R[:, 0] if np.linalg.norm(R[:2, 0]) > 1e-3 else R[:, 1]
    return float(np.arctan2(v[1], v[0]))


def frame(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rotation with x along `x` and y along the part of `y` orthogonal to it."""
    x = np.asarray(x, float) / np.linalg.norm(x)
    y = np.asarray(y, float) - np.dot(y, x) * x
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    """Mean of the points in each voxel of size `voxel` (voxels in order of their integer coordinates)."""
    if len(points) == 0 or voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    lo = keys.min(0)
    keys -= lo
    span = keys.max(0) + 1
    if float(span[0]) * float(span[1]) * float(span[2]) < 2.0 ** 62:   # one int64 per voxel: a 1-D unique
        flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, idx, counts = np.unique(flat, return_inverse=True, return_counts=True)
    else:
        _, idx, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    idx = idx.ravel()
    sums = np.stack([np.bincount(idx, weights=points[:, k], minlength=len(counts)) for k in range(3)], axis=1)
    return sums / counts[:, None]
