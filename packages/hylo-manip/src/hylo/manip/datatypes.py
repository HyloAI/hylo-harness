"""Data types of the toolkit (re-exported by `hylo.manip.tools`). Frames: world, metres, radians, +z up; poses are
4x4 homogeneous matrices; the TCP follows hylo.core.robot (+x approach, +y closing axis, origin between the pads)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

Verb = Literal["grasp", "place", "reach", "lift", "push", "press", "retreat", "home"]


def _r(x: Any, nd: int = 3) -> Any:
    """Short text for an array in a repr: small arrays rounded, large ones by shape (printing a View or a Mask must not
    flood a model's context with pixels)."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.size <= 16:
        return np.round(a.astype(float), nd).tolist() if a.dtype.kind in "fiub" else a.tolist()
    return f"<array {a.shape} {a.dtype}>"


@dataclass
class CameraView:
    """One camera: rgb (H, W, 3) uint8; depth (H, W) metres or None; K (3, 3); T_world_cam (4, 4) in the OpenCV
    optical frame; seg (H, W) int32 instance ids when the environment gives segmentation."""
    rgb: np.ndarray
    depth: np.ndarray | None
    K: np.ndarray | None
    T_world_cam: np.ndarray | None
    seg: np.ndarray | None = None

    def __repr__(self) -> str:
        return (f"CameraView(rgb={_r(self.rgb)}, depth={_r(self.depth)}, seg={_r(self.seg)}, "
                f"position={_r(None if self.T_world_cam is None else self.T_world_cam[:3, 3])})")


@dataclass
class View:
    """One observation: cams {name: CameraView}; points (N, 3) world points fused from every depth camera
    (voxel-downsampled, robot included); segments {instance id: label} when given; step at which it was taken."""
    cams: dict[str, CameraView]
    points: np.ndarray
    segments: dict[int, str] = field(default_factory=dict)
    step: int = 0

    def __repr__(self) -> str:
        return f"View(cams={list(self.cams)}, points={_r(self.points)}, segments={self.segments}, step={self.step})"


@dataclass
class Mask:
    """An instance mask in one camera: label (the text label it answers), camera, mask (H, W) bool, score in [0, 1]
    (1: exact label match), instance (the environment's instance id, when known; label: view.segments[instance])."""
    label: str
    camera: str
    mask: np.ndarray
    score: float = 1.0
    instance: int | None = None

    def __repr__(self) -> str:
        return (f"Mask(label={self.label!r}, camera={self.camera!r}, pixels={int(np.count_nonzero(self.mask))}, "
                f"score={round(float(self.score), 3)}, instance={self.instance})")


@dataclass
class Table:
    """The support plane: height (z), normal (3,) (+z for a level table), extent [[xmin, xmax], [ymin, ymax]]."""
    height: float
    normal: np.ndarray
    extent: np.ndarray | None = None


@dataclass
class Detection:
    """One object: label, score, points (N, 3) world points fused across cameras (table and outliers removed)."""
    label: str
    score: float
    points: np.ndarray

    def __repr__(self) -> str:
        n = len(self.points)
        c = _r(np.asarray(self.points).mean(axis=0)) if n else None
        return f"Detection(label={self.label!r}, score={round(float(self.score), 3)}, points=<{n} points, mean {c}>)"


@dataclass
class Box:
    """Oriented box: center (3,), size (3,) = length along yaw >= width, height; yaw in (-pi/2, pi/2] about +z;
    bottom z. `box.pose` is the 4x4 object frame (origin at center, x along length, z up)."""
    center: np.ndarray
    size: np.ndarray
    yaw: float
    bottom: float

    def __repr__(self) -> str:
        return (f"Box(center={_r(self.center)}, size={_r(self.size)}, yaw={round(float(self.yaw), 3)}, "
                f"bottom={round(float(self.bottom), 4)})")

    @property
    def pose(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        T[:3, 3] = self.center
        return T


@dataclass
class Fit:
    """Rigid registration: T (4, 4) maps model points onto observed ones; rmse (m) over inliers; inliers: fraction of
    observed points within the tolerance of the moved model."""
    T: np.ndarray
    rmse: float
    inliers: float


@dataclass
class Goal:
    """One motion the planner may choose. verb: grasp, place, reach, lift, push, press, retreat, home. arm: arm name
    (None: the robot's only arm).
    tcp: (4, 4) TCP pose to end at. pre: (4, 4) approach pose: the planner reaches `pre` around obstacles, then moves
    in a straight line to `tcp` (that line is checked against the table and the robot only, not observed points, so it
    may touch the object). Without `pre`, verbs lift, retreat, push and press move in a straight line from the current
    TCP; other verbs plan a collision-free joint path to `tcp`. joints: a joint-space goal instead of `tcp` (e.g.
    home). hand_before: hand command while moving (None: keep the current one). hand: hand command at the end, held
    until the hand settles (None: no change). rank: lower is tried first among goals passed together. free: "x" or "z"
    lets the planner rotate freely about that TCP axis (e.g. "x" for a top-down approach to a round object).
    attached: (N, 3) points of a held object in the TCP frame: they move with the hand and are checked against the
    table and observed points like the hand itself (place_at fills it); observed points of the object where it is
    now are ignored."""
    verb: Verb
    arm: str | None = None
    tcp: np.ndarray | None = None
    pre: np.ndarray | None = None
    hand_before: Sequence[float] | None = None
    hand: Sequence[float] | None = None
    joints: np.ndarray | None = None
    rank: float = 0.0
    free: Literal["x", "z"] | None = None
    attached: np.ndarray | None = None

    def __repr__(self) -> str:
        pos = None if self.tcp is None else _r(np.asarray(self.tcp)[:3, 3])
        return (f"Goal(verb={self.verb!r}, arm={self.arm!r}, tcp_position={pos}, pre={'set' if self.pre is not None else None}, "
                f"hand_before={self.hand_before}, hand={self.hand}, joints={_r(self.joints)}, rank={self.rank}, "
                f"free={self.free}, attached={_r(self.attached)})")


@dataclass
class Plan:
    """A timed motion: goal (the one chosen; several arms: the first, all in info["goals"]), actions (T,
    action_dim) joint targets at control rate, start_q {arm: q} it was planned from (run refuses if the robot moved
    away), info (ik residual, path length, duration, method), path {arm: joint waypoints} before timing."""
    goal: Goal
    actions: np.ndarray
    start_q: dict[str, np.ndarray]
    info: dict[str, Any] = field(default_factory=dict)
    path: dict[str, list] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Plan(goal={self.goal!r}, actions={_r(self.actions)}, info={self.info})"


__all__ = ["Box", "CameraView", "Detection", "Fit", "Goal", "Mask", "Plan", "Table", "Verb", "View"]
