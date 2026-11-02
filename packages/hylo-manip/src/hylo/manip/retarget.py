"""Hand motion retargeting: human hand keypoints (the 21-point MediaPipe layout) -> joint angles of a robot hand and
the pose of its palm.

Frames. Both hands get a palm frame from the same recipe: origin at the wrist, x towards the middle finger's base
(knuckle), y across the knuckles from the index towards the outermost finger, z = x cross y (out of the palm on a
right hand, anatomically the same side on both). The robot's wrist point, finger bases and fingertips come from its
model (HandModel); for a three-finger hand the outermost finger is its middle finger.

Per frame the finger joints q minimise, in the robot palm frame,
    sum_i w_tip  |tip_i(q) - o - a Rv(k_tip_i - k_wrist)|^2                          wrist-to-tip vectors
  + sum_j w_ij   |(tip_j(q) - tip_thumb(q)) - a Rv(k_tip_j - k_tip_thumb)|^2         thumb-to-finger vectors
  + w_smooth |q - q_prev|^2 + w_rest |q - q_open|^2
within the joint limits (bounded least squares, analytic Jacobians from MuJoCo). a scales the human hand to the
robot's (middle finger plus palm length, measured once per sequence), Rv expresses human palm-frame vectors in the
robot palm frame; w_ij grows as the human thumb and finger close in, so pinches stay pinches. Robot fingers map to
human fingers by name (thumb, index, middle, ring, little); a robot with fewer fingers uses the first ones.

The palm pose in the world aligns the robot palm frame with the human one, palm centres (wrist and knuckles)
together. A left human hand on a right robot hand (or the reverse) is mirrored: finger shapes reflected through the
palm plane, the palm's normal kept (the thumb then sits on the other side).

    rt = Retargeter(hand_model("allegro_right"))
    out = rt.run(keypoints)            # (T, 21, 3) metres, any frame (camera or world)
    out.q (T, n), out.ee (T, 4, 4) palm (ee frame) poses, out.tcp, out.tip_error_mm (T,), out.ms_per_frame
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .handmodel import HandModel

WRIST = 0
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "little": 17}
CHAIN = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
         "little": (17, 18, 19, 20)}
ORDER = ("thumb", "index", "middle", "ring", "little")


def palm_frame(wrist: np.ndarray, index: np.ndarray, middle: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """4x4 palm frame: origin wrist, x to the middle base, y index -> outer base (made orthogonal), z = x cross y."""
    x = middle - wrist
    x = x / max(np.linalg.norm(x), 1e-9)
    y = outer - index
    y = y - (y @ x) * x
    y = y / max(np.linalg.norm(y), 1e-9)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
    T[:3, 3] = wrist
    return T


def human_frame(k: np.ndarray) -> np.ndarray:
    return palm_frame(k[WRIST], k[BASE["index"]], k[BASE["middle"]], k[BASE["little"]])


@dataclass
class RetargetResult:
    q: np.ndarray                   # (T, n) finger joints (HandSpec.joints order)
    ee: np.ndarray                  # (T, 4, 4) robot ee (palm) pose in the keypoints' frame
    tcp: np.ndarray                 # (T, 4, 4)
    tip_error_mm: np.ndarray        # (T,) mean fingertip error, robot scale, in the palm frame
    pair_error_mm: np.ndarray       # (T,) mean thumb-to-finger vector error
    ms_per_frame: float
    scale: float                    # robot / human hand size
    fingers: dict[str, str]         # robot finger -> human finger
    mirrored: bool
    valid: np.ndarray               # frames with keypoints (others interpolated)
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        v = self.valid
        e = self.tip_error_mm[v] if v.any() else self.tip_error_mm
        return {"frames": int(len(self.q)), "valid": int(v.sum()), "scale": round(self.scale, 3),
                "tip_error_mm": {"mean": round(float(e.mean()), 2), "median": round(float(np.median(e)), 2),
                                 "p95": round(float(np.percentile(e, 95)), 2), "max": round(float(e.max()), 2)},
                "pair_error_mm_mean": round(float(self.pair_error_mm[v].mean()) if v.any() else 0.0, 2),
                "ms_per_frame": round(self.ms_per_frame, 2), "fingers": self.fingers, "mirrored": self.mirrored}


class Retargeter:
    """Keypoints -> robot hand. `scale`: robot/human size (None: measured per sequence). `side` of the human hand
    ("right"/"left"; None: same as the robot's) decides mirroring."""

    def __init__(self, hand: HandModel, *, scale: float | None = None, w_tip: float = 1.0, w_pair: float = 1.0,
                 w_smooth: float = 0.02, w_rest: float = 0.002, pinch_dist: float = 0.03,
                 robot_side: str | None = None) -> None:
        self.hand, self.fixed_scale = hand, scale
        self.w_tip, self.w_pair, self.w_smooth, self.w_rest, self.pinch = w_tip, w_pair, w_smooth, w_rest, pinch_dist
        self.robot_side = robot_side or ("left" if "left" in hand.name else "right")
        names = [f.name for f in hand.fingers]
        self.map = {n: n for n in names if n in TIP}
        others = [n for n in names if n not in TIP]
        free = [h for h in ORDER if h not in self.map.values()]
        for n, hname in zip(others, free):
            self.map[n] = hname
        self.robot_fingers = [n for n in ORDER if n in self.map] + [n for n in self.map if n not in ORDER]
        self.fidx = {f.name: i for i, f in enumerate(hand.fingers)}
        self.thumb = hand.fingers[hand.thumb].name
        # robot palm frame in the ee frame, at the open pose (finger bases are fixed in the palm)
        hand.set_ee(np.eye(4), hand.open)
        bases = {f.name: b for f, b in zip(hand.fingers, hand.finger_bases())}
        nonthumb = [n for n in ("index", "middle", "ring", "little") if n in bases]
        if len(nonthumb) < 2:
            raise ValueError(f"{hand.name}: retargeting needs two fingers besides the thumb")
        mid = bases["middle"] if "middle" in bases else np.mean([bases[n] for n in nonthumb], 0)
        self.wrist_r = hand.wrist()
        self.F_r = palm_frame(self.wrist_r, bases[nonthumb[0]], mid, bases[nonthumb[-1]])
        self.nonthumb = nonthumb
        self.palm_centre_r = np.mean([self.wrist_r] + [bases[n] for n in nonthumb], 0)
        mf = hand.finger("middle") if "middle" in bases else hand.fingers[self.fidx[nonthumb[0]]]
        self.robot_len = float(np.linalg.norm(bases[mf.name] - self.wrist_r)) + self._chain_length(mf)

    def _chain_length(self, f) -> float:
        h = self.hand
        h.set_ee(np.eye(4), h.open)
        anchors = [h.d.xanchor[h.jids[j]].copy() for j in f.joints]
        tip = h.tips()[self.fidx[f.name]]
        pts = anchors + [tip]
        return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))

    @staticmethod
    def human_length(k: np.ndarray) -> float:
        c = (WRIST,) + CHAIN["middle"]
        return float(sum(np.linalg.norm(k[c[i + 1]] - k[c[i]]) for i in range(len(c) - 1)))

    # ------------------------------------------------------------------ one frame
    def targets(self, k: np.ndarray, scale: float, mirror: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Robot-palm-frame (ee coordinates) targets: tips relative to nothing (absolute in ee), and thumb-to-finger
        vectors."""
        F_h = human_frame(k)
        M = np.diag([1.0, 1.0, -1.0]) if mirror else np.eye(3)
        Rv = self.F_r[:3, :3] @ M @ F_h[:3, :3].T            # human world vector -> robot ee vector
        o = self.wrist_r
        tips, pairs = {}, {}
        for rn in self.robot_fingers:
            hn = self.map[rn]
            tips[rn] = o + scale * Rv @ (k[TIP[hn]] - k[WRIST])
        th = self.map[self.thumb]
        for rn in self.robot_fingers:
            if rn != self.thumb:
                pairs[rn] = scale * Rv @ (k[TIP[self.map[rn]]] - k[TIP[th]])
        return tips, pairs

    def solve(self, k: np.ndarray, q0: np.ndarray, q_prev: np.ndarray | None, scale: float,
              mirror: bool = False) -> tuple[np.ndarray, float, float]:
        h = self.hand
        tips, pairs = self.targets(k, scale, mirror)
        fi = [self.fidx[n] for n in self.robot_fingers]
        ti = self.fidx[self.thumb]
        pnames = [n for n in self.robot_fingers if n != self.thumb]
        th = self.map[self.thumb]
        wp = {n: self.w_pair * (1 + 4 * np.exp(-np.linalg.norm(k[TIP[self.map[n]]] - k[TIP[th]]) / self.pinch))
              for n in pnames}
        lo, hi = h.lower + 1e-4, h.upper - 1e-4
        q_rest = h.clip(h.open)
        qp = q_prev if q_prev is not None else q0
        n = len(h.jids)
        u = 100.0                                   # position residuals in centimetres, angles in radians
        st, sp = u * np.sqrt(self.w_tip), {p: u * np.sqrt(wp[p]) for p in pnames}
        ss, sr = np.sqrt(self.w_smooth), np.sqrt(self.w_rest)

        cache: dict[str, np.ndarray] = {}

        def fk(q):
            key = q.tobytes()
            if cache.get("key") != key:
                h.set_ee(np.eye(4), q)
                cache.update(key=key, tips=h.tips(), J=h.tip_jacobians())
            return cache["tips"], cache["J"]

        def res(q):
            P, _ = fk(q)
            r = [st * (P[i] - tips[nm]) for i, nm in zip(fi, self.robot_fingers)]
            r += [sp[nm] * ((P[self.fidx[nm]] - P[ti]) - pairs[nm]) for nm in pnames]
            r += [ss * (q - qp), sr * (q - q_rest)]
            return np.concatenate(r)

        def jac(q):
            _, J = fk(q)
            rows = [st * J[i] for i in fi]
            rows += [sp[nm] * (J[self.fidx[nm]] - J[ti]) for nm in pnames]
            rows += [ss * np.eye(n), sr * np.eye(n)]
            return np.vstack(rows)

        x0 = np.clip(q0, lo, hi)
        sol = least_squares(res, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=40, xtol=1e-6, ftol=1e-8)
        q = sol.x
        P, _ = fk(q)
        tip_err = float(np.mean([np.linalg.norm(P[i] - tips[nm]) for i, nm in zip(fi, self.robot_fingers)]))
        pair_err = float(np.mean([np.linalg.norm((P[self.fidx[nm]] - P[ti]) - pairs[nm]) for nm in pnames]))
        return q, tip_err, pair_err

    # ------------------------------------------------------------------ sequences
    def run(self, keypoints: np.ndarray, valid: np.ndarray | None = None, side: str | None = None) -> RetargetResult:
        """Retarget a (T, 21, 3) sequence (or one (21, 3) frame). Invalid frames are interpolated afterwards."""
        K = np.asarray(keypoints, float)
        if K.ndim == 2:
            K = K[None]
        T = len(K)
        v = np.ones(T, bool) if valid is None else np.asarray(valid, bool).copy()
        v &= np.isfinite(K).all((1, 2))
        mirror = side is not None and side != self.robot_side
        h = self.hand
        if self.fixed_scale is not None:
            scale = float(self.fixed_scale)
        else:
            lens = [self.human_length(K[t]) for t in np.flatnonzero(v)]
            scale = self.robot_len / float(np.median(lens)) if lens else 1.0
        n = len(h.jids)
        Q = np.tile(h.clip(h.open), (T, 1))
        E = np.zeros((T, 4, 4))
        te, pe = np.zeros(T), np.zeros(T)
        q_prev = None
        t0 = time.perf_counter()
        solved = 0
        for t in range(T):
            if not v[t]:
                continue
            q0 = h.clip(h.open) if q_prev is None else q_prev
            Q[t], te[t], pe[t] = self.solve(K[t], q0, q_prev, scale, mirror)
            q_prev = Q[t]
            E[t] = self.palm_pose(K[t], mirror)
            solved += 1
        ms = 1000 * (time.perf_counter() - t0) / max(1, solved)
        good = np.flatnonzero(v)
        if len(good) and len(good) < T:
            idx = np.arange(T)
            for j in range(n):
                Q[:, j] = np.interp(idx, good, Q[good, j])
            for r in range(3):
                for c in range(4):
                    E[:, r, c] = np.interp(idx, good, E[good, r, c])
            for t in range(T):
                U, _, Vt = np.linalg.svd(E[t, :3, :3])
                E[t, :3, :3] = U @ Vt
            E[:, 3] = [0, 0, 0, 1]
        elif not len(good):
            E[:] = np.eye(4)
        tcp = E @ h.tcp_in_ee
        return RetargetResult(Q, E, tcp, 1000 * te, 1000 * pe, ms, scale,
                              {r: self.map[r] for r in self.robot_fingers}, mirror, v)

    def palm_pose(self, k: np.ndarray, mirror: bool = False) -> np.ndarray:
        """The robot ee pose putting its palm frame on the human palm frame, palm centres together."""
        F_h = human_frame(k)
        M = np.diag([1.0, -1.0, -1.0]) if mirror else np.eye(3)
        R = F_h[:3, :3] @ M @ self.F_r[:3, :3].T
        c_h = np.mean([k[WRIST]] + [k[BASE[self.map[f]]] for f in self.nonthumb], 0)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c_h - R @ self.palm_centre_r
        return T


def keypoints_from_robot(hand: HandModel, q: np.ndarray, T_world_ee: np.ndarray | None = None) -> np.ndarray:
    """A 21-point "human" hand made from the robot hand itself at joints q (tests and demos with a known answer):
    wrist at the robot's wrist point, each mapped finger's base, joint anchors and tip at the robot's; fingers the
    robot lacks are placed beside its outermost finger, straight."""
    Te = np.eye(4) if T_world_ee is None else np.asarray(T_world_ee, float)
    hand.set_ee(Te, q)
    k = np.zeros((21, 3))
    k[WRIST] = hand.wrist()
    tips = hand.tips()
    bases = hand.finger_bases()
    present = {}
    for i, f in enumerate(hand.fingers):
        if f.name not in CHAIN:
            continue
        anchors = [hand.d.xanchor[hand.jids[j]].copy() for j in f.joints]
        pts = anchors[-3:] if len(anchors) >= 3 else anchors
        chain = CHAIN[f.name]
        k[chain[0]] = bases[i]
        mids = np.linspace(0, 1, 4)[1:3]
        inner = pts[1:] if len(pts) >= 3 else [bases[i] + m * (tips[i] - bases[i]) for m in mids]
        if len(inner) < 2:
            inner = [bases[i] + m * (tips[i] - bases[i]) for m in mids]
        k[chain[1]], k[chain[2]] = inner[0], inner[1]
        k[chain[3]] = tips[i]
        present[f.name] = i
    order = [n for n in ("index", "middle", "ring", "little") if n in present]
    last = order[-1]
    step = k[BASE[order[-1]]] - k[BASE[order[-2]]]
    for j, name in enumerate([n for n in ("ring", "little") if n not in present]):
        off = step * (j + 1)
        for a, b in zip(CHAIN[name], CHAIN[last]):
            k[a] = k[b] + off
    return k


__all__ = ["BASE", "CHAIN", "RetargetResult", "Retargeter", "TIP", "human_frame", "keypoints_from_robot",
           "palm_frame"]
