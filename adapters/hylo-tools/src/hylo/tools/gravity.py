"""`gravity`: one image -> camera gravity direction (roll, pitch) and focal length (GeoCalib, on the CPU).

    hylo tool gravity IMAGE [--weights pinhole|distorted] [--focal-prior PX] [--out DIR]

Python:
    g = gravity(rgb)     # Gravity: roll, pitch (rad), gravity_cam (3,), R_level_cam (3, 3), K (3, 3), vfov_deg

Conventions (GeoCalib's): with g the unit gravity (down) in the camera frame (OpenCV), roll = atan2(g_x, g_y)
(positive when the image content is rotated counter-clockwise) and pitch = -asin(g_z) (positive looking up).
`R_level_cam` turns camera-frame (OpenCV) vectors into a level frame (z up); with a known camera height h it gives a
camera pose for a single image: T_world_cam = [[R_level_cam, (0, 0, h)], [0, 1]] (yaw fixed: camera looks along +x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .runner import ToolResult, ToolServer, run_worker

MODEL = "GeoCalib (geocalib-pinhole)"
LICENCE = "Apache-2.0 code; CC-BY-4.0 weights"


@dataclass
class Gravity:
    roll: float                  # rad, GeoCalib convention: atan2(g_x, g_y)
    pitch: float                 # rad, positive when the camera looks up: -asin(g_z)
    gravity_cam: np.ndarray      # (3,) unit, camera frame (OpenCV), pointing down
    R_level_cam: np.ndarray      # (3, 3)
    K: np.ndarray                # (3, 3)
    vfov_deg: float
    result: ToolResult

    def T_world_cam(self, height: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[2, 3] = self.R_level_cam, height
        return T


def gravity(image: Any, *, weights: str = "pinhole", camera_model: str = "pinhole", focal_prior: float | None = None,
            preview: bool = True, server: ToolServer | None = None, **run_kw: Any) -> Gravity:
    """Gravity direction and intrinsics for one image (path, (H, W, 3) uint8 array or PIL image)."""
    req = {"weights": weights, "camera_model": camera_model, "focal_prior": focal_prior, "preview": preview}
    res = server.call(req, {"image": image}) if server else run_worker("gravity", req, {"image": image}, **run_kw)
    a = res.arrays("gravity")
    return Gravity(float(a["roll"]), float(a["pitch"]), a["gravity_cam"], a["R_level_cam"], a["K"],
                   float(res.outputs["vfov_deg"]), res)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("gravity", __doc__)
    ap.add_argument("image")
    ap.add_argument("--weights", default="pinhole", choices=["pinhole", "distorted"])
    ap.add_argument("--camera-model", default="pinhole", choices=["pinhole", "simple_radial", "simple_divisional"])
    ap.add_argument("--focal-prior", type=float, help="known focal length in pixels")
    a = ap.parse_args(argv)
    return _cli.guard(lambda: _cli.finish(gravity(a.image, weights=a.weights, camera_model=a.camera_model,
                                                  focal_prior=a.focal_prior, **_cli.run_kw(a)).result, a.out))


def roll_pitch_from_R(R_world_cam: np.ndarray) -> tuple[float, float]:
    """Ground-truth roll and pitch (GeoCalib's convention) of a camera with known world rotation (world z up): the
    gravity direction in the camera frame g = R^T (0, 0, -1)."""
    g = np.asarray(R_world_cam).T @ np.array([0.0, 0.0, -1.0])
    roll = math.atan2(g[0], g[1])
    pitch = -math.asin(np.clip(g[2], -1, 1))
    return roll, pitch
