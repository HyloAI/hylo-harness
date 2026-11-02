"""`depth`: one image -> metric depth, camera-frame points, normals and intrinsics (MoGe-2, on the CPU).

    hylo tool depth IMAGE [--model Ruicheng/moge-2-vitl-normal] [--fov-x DEG] [--out DIR]

Python:
    d = depth(rgb)            # Depth: depth (H, W) m, points (H, W, 3), normal, mask, K (3, 3)
    cam = d.camera_view(rgb)  # a hylo.manip CameraView (camera at the world origin unless T_world_cam is given)

Models: Ruicheng/moge-2-vitl-normal (default, 326M), moge-2-vitb-normal (104M), moge-2-vits-normal (35M); MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .runner import ToolResult, ToolServer, run_worker

MODEL = "Ruicheng/moge-2-vitl-normal"
LICENCE = "MIT (code and weights)"


@dataclass
class Depth:
    depth: np.ndarray            # (H, W) metres, inf where invalid
    points: np.ndarray           # (H, W, 3) camera frame (OpenCV)
    normal: np.ndarray | None    # (H, W, 3)
    mask: np.ndarray             # (H, W) bool
    K: np.ndarray                # (3, 3) pixels
    result: ToolResult

    def camera_view(self, rgb: np.ndarray, T_world_cam: np.ndarray | None = None) -> Any:
        from hylo.manip.datatypes import CameraView
        return CameraView(rgb=np.asarray(rgb), depth=np.where(self.mask, self.depth, 0.0), K=self.K,
                          T_world_cam=np.eye(4) if T_world_cam is None else np.asarray(T_world_cam))


def depth(image: Any, *, model: str = MODEL, fov_x: float | None = None, resolution_level: int = 9,
          preview: bool = True, server: ToolServer | None = None, **run_kw: Any) -> Depth:
    """Metric depth and intrinsics for one image (path, (H, W, 3) uint8 array or PIL image)."""
    req = {"model": model, "fov_x": fov_x, "resolution_level": resolution_level, "preview": preview}
    res = server.call(req, {"image": image}) if server else run_worker("depth", req, {"image": image}, **run_kw)
    a = res.arrays("depth")
    return Depth(a["depth"], a["points"], a.get("normal"), a["mask"].astype(bool), a["K"], res)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("depth", __doc__)
    ap.add_argument("image")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fov-x", type=float, help="horizontal field of view hint in degrees")
    ap.add_argument("--resolution-level", type=int, default=9)
    a = ap.parse_args(argv)
    return _cli.guard(lambda: _cli.finish(depth(a.image, model=a.model, fov_x=a.fov_x,
                                                resolution_level=a.resolution_level, **_cli.run_kw(a)).result, a.out))
