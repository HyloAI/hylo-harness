"""Editing a hand's motion so it holds what it should: a retargeted (human) trajectory of TCP poses and finger joints
bent onto a planned grasp around the grasp moment.

    tcp, fingers, info = bend_onto_grasp(tcp, fingers, grasp, start=a, end=b, fps=15.0)

Phases (frame indices; `grasp` is a DexGrasp from hylo.manip.dexgrasp):
  ramp     [a - ramp, a - approach): from the retargeted pose to the pre-pose (6 cm back along the approach),
           fingers to the grasp's open pre-shape;
  approach [a - approach, a): straight in along the approach to the grasp pose, fingers open;
  hold     [a, b): the grasp's offset from the retargeted TCP at frame a rides on the retargeted motion (the hand
           carries the object as the human hand does), fingers close to the squeeze command over `close_s`;
  release  [b, b + open_s): offset kept, fingers back to the pre-shape;
  return   then over `ramp`: back to the retargeted motion.
Frames outside these phases are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inv


def smoothstep(w: float) -> float:
    w = float(np.clip(w, 0.0, 1.0))
    return w * w * (3 - 2 * w)


def blend_pose(A: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
    """Pose between A (w = 0) and B (w = 1): positions linear, rotation along the shortest arc."""
    T = np.eye(4)
    T[:3, 3] = (1 - w) * A[:3, 3] + w * B[:3, 3]
    rel = Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec()
    T[:3, :3] = A[:3, :3] @ Rotation.from_rotvec(w * rel).as_matrix()
    return T


def bend_onto_grasp(tcp: np.ndarray, fingers: np.ndarray, grasp: Any, start: int, end: int, fps: float, *,
                    ramp_s: float = 0.6, approach_s: float = 0.3, close_s: float = 0.3, open_s: float = 0.4,
                    pre_dist: float = 0.06) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """New (T, 4, 4) TCP poses and (T, n) finger targets; see the module docstring. The inputs are not changed."""
    ref_tcp, ref_q = np.asarray(tcp, float), np.asarray(fingers, float)
    out_tcp, out_q = ref_tcp.copy(), ref_q.copy()
    T = len(ref_tcp)
    a, b = int(start), int(min(max(end, start + 1), T))
    n_ramp = max(3, int(round(ramp_s * fps)))
    n_app = max(2, int(round(approach_s * fps)))
    n_close = max(2, int(round(close_s * fps)))
    n_open = max(2, int(round(open_s * fps)))
    G = np.asarray(grasp.tcp, float)
    delta = inv(ref_tcp[a]) @ G                        # the grasp in the retargeted TCP frame at the grasp moment
    pre = G.copy()
    pre[:3, 3] -= np.asarray(grasp.approach, float) * pre_dist
    q_open, q_sq = np.asarray(grasp.q_open, float), np.asarray(grasp.q_squeeze, float)
    s0 = max(0, a - n_ramp)
    s1 = max(s0, a - n_app)
    for t in range(s0, s1):
        w = smoothstep((t - s0 + 1) / max(1, s1 - s0))
        out_tcp[t] = blend_pose(ref_tcp[t], pre, w)
        out_q[t] = (1 - w) * ref_q[t] + w * q_open
    for t in range(s1, a):
        out_tcp[t] = blend_pose(pre, G, smoothstep((t - s1 + 1) / max(1, a - s1)))
        out_q[t] = q_open
    for t in range(a, b):
        out_tcp[t] = ref_tcp[t] @ delta
        w = min(1.0, (t - a + 1) / n_close)
        out_q[t] = (1 - w) * q_open + w * q_sq
    r1 = min(T, b + n_open)
    for t in range(b, r1):
        out_tcp[t] = ref_tcp[t] @ delta
        w = (t - b + 1) / n_open
        out_q[t] = (1 - w) * q_sq + w * q_open
    for t in range(r1, min(T, r1 + n_ramp)):
        w = smoothstep((t - r1 + 1) / n_ramp)
        out_tcp[t] = blend_pose(ref_tcp[t] @ delta, ref_tcp[t], w)
        out_q[t] = (1 - w) * q_open + w * ref_q[t]
    turn = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    return out_tcp, out_q, {"shift_m": round(float(np.linalg.norm(G[:3, 3] - ref_tcp[a][:3, 3])), 4),
                            "turn_deg": round(turn, 1), "first_frame": s0, "last_frame": min(T, r1 + n_ramp) - 1}


__all__ = ["bend_onto_grasp", "blend_pose", "smoothstep"]
