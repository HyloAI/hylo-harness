"""Motion data and edits for the motion studio (pure numpy): one keypoint skeleton, converters from the motion
tools and edits on joint trajectories.

Skeleton: 17 keypoints (`NAMES`, SOMA joint names), metres, z up, ground z = 0. `canonical` puts a motion on the
ground, at the origin, facing +x at its first frame, so edits can align clips by root position and heading.
Edits (`concat`, `blend`, `loop`, `mirror`, `retime`, `trim`, `hold`, `keyframe`) work on positions: clips are
aligned (the next clip starts where and facing where the last one ended) and cross-faded over a short window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

NAMES = ["Hips", "Chest", "Head", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
         "RightFoot", "RightToeBase", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
PARENTS = [-1, 0, 1, 0, 3, 4, 5, 0, 7, 8, 9, 1, 11, 12, 1, 14, 15]
IDX = {n: i for i, n in enumerate(NAMES)}
FEET = [IDX[n] for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
# MediaPipe Pose landmark indices per keypoint (averaged when several)
MEDIAPIPE = {"Hips": (23, 24), "LeftLeg": (23,), "LeftShin": (25,), "LeftFoot": (27,), "LeftToeBase": (31,),
             "RightLeg": (24,), "RightShin": (26,), "RightFoot": (28,), "RightToeBase": (32,), "LeftArm": (11,),
             "LeftForeArm": (13,), "LeftHand": (15,), "RightArm": (12,), "RightForeArm": (14,), "RightHand": (16,),
             "Head": (7, 8)}
CHEST_FRACTION = 0.47     # SOMA's Chest sits about this far from the hips towards the shoulders


@dataclass
class Motion:
    """A skeleton motion: joints (T, 17, 3) in NAMES order, at fps. `source` says where it came from."""
    joints: np.ndarray
    fps: float = 30.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(len(self.joints))

    @property
    def seconds(self) -> float:
        return round(self.frames / self.fps, 3)

    @property
    def names(self) -> list[str]:
        return list(NAMES)

    def joint(self, name: str) -> np.ndarray:
        """(T, 3) trajectory of one keypoint."""
        return self.joints[:, IDX[name]].copy()

    def pose(self, t: float) -> "Motion":
        """The frame at time t (seconds) as a one-frame motion."""
        i = int(np.clip(round(t * self.fps), 0, self.frames - 1))
        return Motion(self.joints[i:i + 1].copy(), self.fps, f"pose({self.source}, {t:g}s)")

    def summary(self) -> dict[str, Any]:
        J = self.joints
        root = J[:, 0]
        head = heading(J)
        hands = {s: round(float(np.ptp(J[:, IDX[f"{s}Hand"], 2])), 2) for s in ("Left", "Right")}
        return {"source": self.source, "frames": self.frames, "fps": self.fps, "seconds": self.seconds,
                "root_path_m": round(float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()), 2),
                "displacement_xy_m": np.round(root[-1, :2] - root[0, :2], 2).tolist(),
                "turn_deg": round(float(np.degrees(np.unwrap(head)[-1] - head[0])), 1),
                "pelvis_height_m": [round(float(root[:, 2].min()), 2), round(float(root[:, 2].max()), 2)],
                "feet_max_height_m": round(float(J[:, FEET, 2].min(1).max()), 2),
                "hand_height_range_m": hands}

    def __repr__(self) -> str:
        s = self.summary()
        return (f"Motion({s['source'] or 'motion'}: {s['seconds']} s, {s['frames']} frames, path {s['root_path_m']} m, "
                f"turn {s['turn_deg']} deg)")


# ------------------------------------------------------------------------------------------------- converters
def from_soma(joints: np.ndarray, names: Sequence[str], fps: float, source: str = "text_motion") -> Motion:
    idx = {str(n): i for i, n in enumerate(names)}
    return canonical(Motion(np.asarray(joints, float)[:, [idx[n] for n in NAMES]], float(fps), source))


def from_mediapipe(joints_world: np.ndarray, valid: np.ndarray, fps: float, source: str = "video") -> Motion:
    """MediaPipe Pose world keypoints (T, 33, 3; z up) -> the skeleton; frames without a person are interpolated."""
    W = np.asarray(joints_world, float)
    valid = np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("no person was found in any frame")
    out = np.zeros((len(W), len(NAMES), 3))
    for n, ids in MEDIAPIPE.items():
        out[:, IDX[n]] = W[:, list(ids)].mean(1)
    sh = W[:, [11, 12]].mean(1)
    out[:, IDX["Chest"]] = out[:, 0] + CHEST_FRACTION * (sh - out[:, 0])
    t = np.arange(len(W))
    good = np.flatnonzero(valid)
    J = np.stack([[np.interp(t, good, out[good, j, k]) for k in range(3)] for j in range(len(NAMES))], 0)
    J = J.transpose(2, 0, 1)
    k = int(round(0.15 * fps))                    # monocular depth jitters the root: smooth its path (0.3 s)
    if len(J) > 2 * k + 1 and k:
        root = np.pad(J[:, 0], ((k, k), (0, 0)), mode="edge")
        smooth = np.stack([np.convolve(root[:, c], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for c in range(3)], 1)
        J = J + (smooth - J[:, 0])[:, None]
    return canonical(level(Motion(J, float(fps), source)))


def level(m: Motion) -> Motion:
    """Remove a sideways tilt of the ground (a camera roll the pose estimator did not know): the standing heights
    of the left and right foot keypoints should match."""
    J = m.joints.copy()
    pairs = [(IDX["LeftFoot"], IDX["RightFoot"]), (IDX["LeftToeBase"], IDX["RightToeBase"])]
    yaw = float(np.mean(heading(J[:3])))
    c, s = np.cos(-yaw), np.sin(-yaw)
    lateral = lambda p: s * p[..., 0] + c * p[..., 1]           # noqa: E731 - left of the initial heading
    slopes = []
    for a, b in pairs:
        za, zb = np.percentile(J[:, a, 2], 20), np.percentile(J[:, b, 2], 20)
        dy = float(np.median(lateral(J[:, a]) - lateral(J[:, b])))
        if abs(dy) > 0.08:
            slopes.append((za - zb) / dy)
    if not slopes:
        return m
    roll = float(np.arctan(np.mean(slopes)))                    # rotate about the heading axis by -roll
    ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) - np.sin(roll) * K + (1 - np.cos(roll)) * K @ K
    centre = J[0, 0]
    return Motion((J - centre) @ R.T + centre, m.fps, m.source, dict(m.meta, level_roll_deg=round(np.degrees(roll), 1)))


# ------------------------------------------------------------------------------------------------- geometry
def heading(J: np.ndarray) -> np.ndarray:
    """Yaw (rad) per frame of the hips' forward direction (left hip - right hip, turned -90 deg about z)."""
    lat = J[:, IDX["LeftLeg"], :2] - J[:, IDX["RightLeg"], :2]
    return np.arctan2(-lat[:, 0], lat[:, 1])


def _rotz(J: np.ndarray, yaw: float, about: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (J - about) @ R.T + about


def canonical(m: Motion) -> Motion:
    """Ground at z = 0 (low percentile of the feet), first root at x = y = 0, facing +x at the first frame."""
    J = m.joints.copy()
    J[..., 2] -= float(np.percentile(J[:, FEET, 2].min(1), 5))
    J[..., :2] -= J[0, 0, :2]
    J = _rotz(J, -float(np.mean(heading(J[:3]))), np.zeros(3))
    return Motion(J, m.fps, m.source, dict(m.meta))


def _align(prev_end: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Move and turn `nxt` (T, J, 3) so its first frame has the root position and heading of `prev_end` (J, 3)."""
    yaw = float(heading(prev_end[None])[0] - heading(nxt[:1])[0])
    J = _rotz(nxt, yaw, nxt[0, 0] * [1, 1, 0])
    J[..., :2] += prev_end[0, :2] - J[0, 0, :2]
    return J


def _fps(motions: Sequence[Motion]) -> float:
    return float(motions[0].fps)


def _resample(J: np.ndarray, n: int) -> np.ndarray:
    if len(J) == n:
        return J.copy()
    src = np.linspace(0.0, 1.0, len(J))
    dst = np.linspace(0.0, 1.0, n)
    flat = J.reshape(len(J), -1)
    return np.stack([np.interp(dst, src, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(n, *J.shape[1:])


def _at_fps(m: Motion, fps: float) -> np.ndarray:
    return m.joints if abs(m.fps - fps) < 1e-6 else _resample(m.joints, max(1, round(m.frames * fps / m.fps)))


def _smoothstep(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------------------------------------- edits
def concat(*motions: Motion, blend: float = 0.3) -> Motion:
    """Play clips one after another; each starts where (and facing where) the previous ended, cross-faded over
    `blend` seconds (the overlap shortens the result by that much per join)."""
    if not motions:
        raise ValueError("concat: give at least one motion")
    fps = _fps(motions)
    J = _at_fps(motions[0], fps).copy()
    for m in motions[1:]:
        nxt = _at_fps(m, fps)
        k = int(min(round(blend * fps), len(J) - 1, len(nxt) - 1))
        if k <= 0:
            J = np.concatenate([J, _align(J[-1], nxt)])
            continue
        nxt = _align(J[-k], nxt)
        w = _smoothstep(k)[:, None, None]
        J = np.concatenate([J[:-k], (1 - w) * J[-k:] + w * nxt[:k], nxt[k:]])
    return Motion(J, fps, " + ".join(m.source or "motion" for m in motions))


def blend(a: Motion, b: Motion, seconds: float = 0.5) -> Motion:
    """A smooth transition from `a` into `b` over `seconds` (same as concat with a longer cross-fade)."""
    return concat(a, b, blend=seconds)


def loop(m: Motion, times: int = 2, blend: float = 0.25) -> Motion:
    """Repeat a clip `times` times; each repeat continues from where the previous one ended."""
    return concat(*[m] * max(1, int(times)), blend=blend)


def mirror(m: Motion) -> Motion:
    """Swap left and right (mirror across the vertical plane along the initial heading)."""
    J = m.joints.copy()
    yaw = float(np.mean(heading(J[:3])))
    origin = J[0, 0] * [1, 1, 0]
    J = _rotz(J, -yaw, origin)
    J[..., 1] = 2 * origin[1] - J[..., 1]
    swap = [IDX[n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")] for n in NAMES]
    J = _rotz(J[:, swap], yaw, origin)
    return Motion(J, m.fps, f"mirror({m.source})")


def retime(m: Motion, speed: float | None = None, seconds: float | None = None) -> Motion:
    """Play faster (speed > 1) or slower, or stretch to a duration in seconds."""
    if (speed is None) == (seconds is None):
        raise ValueError("retime: give speed or seconds")
    n = max(2, round((seconds * m.fps) if seconds is not None else m.frames / float(speed)))
    return Motion(_resample(m.joints, n), m.fps, f"retime({m.source})")


def trim(m: Motion, start: float = 0.0, end: float | None = None) -> Motion:
    """The part between start and end (seconds; end None = to the end)."""
    a = int(np.clip(round(start * m.fps), 0, m.frames - 1))
    b = m.frames if end is None else int(np.clip(round(end * m.fps), a + 1, m.frames))
    return Motion(m.joints[a:b].copy(), m.fps, f"trim({m.source})")


def hold(pose: Motion, seconds: float = 1.0) -> Motion:
    """Stand still in a pose (the last frame of `pose`) for `seconds`."""
    n = max(1, round(seconds * pose.fps))
    return Motion(np.repeat(pose.joints[-1:], n, axis=0), pose.fps, f"hold({pose.source})")


def keyframe(m: Motion, pose: Motion, at: float, width: float = 0.8, root: bool = False) -> Motion:
    """Constrain `m` to pass through `pose` at time `at` (seconds): the body offset (relative to the root and
    heading, so the pose is matched where the motion is at that time) is added with a smooth bump of `width`
    seconds on each side, full at `at`; from `at` on it is held when `at` is the last frame. `root`: also match the
    pelvis height."""
    J = m.joints.copy()
    i = int(np.clip(round(at * m.fps), 0, m.frames - 1))
    p = pose.joints[-1]
    yaw = float(heading(J[i:i + 1])[0] - heading(p[None])[0])
    tgt = _rotz(p - p[0] * [1, 1, 0], yaw, np.zeros(3)) + J[i, 0] * [1, 1, 0]
    if not root:
        tgt[:, 2] += J[i, 0, 2] - tgt[0, 2]
    delta = tgt - J[i]
    k = max(1, round(width * m.fps))
    t = np.arange(m.frames)
    w = np.clip(1 - np.abs(t - i) / k, 0, 1)
    w = w * w * (3 - 2 * w)
    J += w[:, None, None] * delta
    return Motion(J, m.fps, f"keyframe({m.source}, {at:g}s)")


EDITS = {"concat": concat, "blend": blend, "loop": loop, "mirror": mirror, "retime": retime, "trim": trim,
         "hold": hold, "keyframe": keyframe}


__all__ = ["EDITS", "FEET", "IDX", "Motion", "NAMES", "PARENTS", "blend", "canonical", "concat", "from_mediapipe",
           "from_soma", "heading", "hold", "keyframe", "level", "loop", "mirror", "retime", "trim"]
