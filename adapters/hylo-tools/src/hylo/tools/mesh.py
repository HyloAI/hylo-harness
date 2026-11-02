"""`mesh`: image + mask -> object mesh (.glb and .obj).

    hylo tool mesh IMAGE MASK [--depth DEPTH.npz] [--backend auto|sam3d|depth] [--method carve|hull|relief]
                   [--scale S] [--out DIR]

Backends behind one interface:
- sam3d: SAM 3D Objects (facebook/sam-3d-objects, gated). Used when the weights are accessible; while access is
  pending (403) `auto` says so and falls back.
- depth (CPU fallback): metric depth (the `depth` tool, MoGe-2, unless a depth map is given) -> masked back-projection
  -> cleaned point cloud, back completed by reflection across the silhouette plane -> closed surface (frustum carving + marching cubes, or
  convex hull) with vertex colours, in metres in the camera frame (or world, with T_world_cam). `scale` corrects a
  known metric scale error of the depth (e.g. from a known object size or table distance).

MASK: a (H, W) bool .npy, a PNG (white = object) or a masks.npz from `segment` (--instance picks one; default union).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .runner import ToolError, ToolResult, run_worker

MODEL = "SAM 3D Objects (facebook/sam-3d-objects) | fallback: MoGe-2 depth + frustum carving"
LICENCE = "SAM License (gated) | fallback: MIT (MoGe-2), BSD-3 (scikit-image), Apache-2.0"


@dataclass
class Mesh:
    glb: Path
    obj: Path
    backend: str                    # sam3d | depth
    info: dict[str, Any]            # vertices, faces, watertight, volume_m3, extents_m, centre, frame, ...
    result: ToolResult
    depth_result: ToolResult | None = None
    fallback_reason: str | None = None
    timing: dict[str, float] = field(default_factory=dict)


def _mask_input(mask: Any, instance: int | None) -> Any:
    if isinstance(mask, (str, Path)) and str(mask).endswith(".npz"):
        with np.load(mask) as z:
            m = z["masks"] if "masks" in z.files else z[z.files[0]]
        m = np.asarray(m, bool)
        return m[instance] if (instance is not None and m.ndim == 3) else m
    if isinstance(mask, np.ndarray):
        return mask.astype(bool)
    return mask


def mesh(image: Any, mask: Any, *, depth: Any = None, K: np.ndarray | None = None, backend: str = "auto",
         method: str = "carve", scale: float = 1.0, T_world_cam: np.ndarray | None = None, instance: int | None = None,
         depth_model: str = "Ruicheng/moge-2-vitl-normal", **run_kw: Any) -> Mesh:
    """An object mesh from one image and the object's mask. `depth`: None (run the depth tool), a depth.Depth, a
    path to an .npz with depth and K, or an (H, W) array (then pass K)."""
    if backend not in ("auto", "sam3d", "depth"):
        raise ValueError(f"backend {backend!r}: auto, sam3d or depth")
    mask_in = _mask_input(mask, instance)
    reason = None
    if backend in ("auto", "sam3d"):
        try:
            res = run_worker("mesh", {"backend": "sam3d"}, {"image": image, "mask": mask_in}, **run_kw)
            return Mesh(res.files["glb"], res.files["obj"], "sam3d", res.outputs, res, timing=res.timing)
        except ToolError as exc:
            if backend == "sam3d" or not exc.blocked:
                raise
            reason = f"{exc} -> {exc.blocked}"
    depth_res = None
    if depth is None:
        from .depth import depth as run_depth
        d = run_depth(image, model=depth_model, preview=False, **run_kw)
        depth_res, depth_in = d.result, d.result.files["depth"]
    elif hasattr(depth, "result"):
        depth_res, depth_in = depth.result, depth.result.files["depth"]
    elif isinstance(depth, np.ndarray):
        if K is None:
            raise ValueError("pass K with a depth array")
        depth_in = {"depth": depth, "K": np.asarray(K)}
    else:
        depth_in = depth
    req = {"backend": "depth", "method": method, "scale": scale,
           "T_world_cam": None if T_world_cam is None else np.asarray(T_world_cam).tolist(),
           "K": None if (K is None or isinstance(depth, np.ndarray)) else np.asarray(K).tolist()}
    res = run_worker("mesh", req, {"image": image, "mask": mask_in, "depth": depth_in}, **run_kw)
    timing = {"mesh_wall_s": res.timing["wall_s"]}
    if depth_res is not None and depth is None:
        timing["depth_wall_s"] = depth_res.timing["wall_s"]
    timing["wall_s"] = round(sum(timing.values()), 3)
    return Mesh(res.files["glb"], res.files["obj"], "depth", res.outputs, res, depth_res, reason, timing)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("mesh", __doc__)
    ap.add_argument("image")
    ap.add_argument("mask")
    ap.add_argument("--depth", help=".npz with depth (H, W) and K (3, 3), e.g. the depth tool's depth.npz")
    ap.add_argument("--instance", type=int, help="instance index in a segment masks.npz (default: union)")
    ap.add_argument("--backend", default="auto", choices=["auto", "sam3d", "depth"])
    ap.add_argument("--method", default="carve", choices=["carve", "hull", "relief"])
    ap.add_argument("--scale", type=float, default=1.0, help="metric scale correction of the depth")
    a = ap.parse_args(argv)

    def go() -> int:
        m = mesh(a.image, a.mask, depth=a.depth, backend=a.backend, method=a.method, scale=a.scale,
                 instance=a.instance, **_cli.run_kw(a))
        if m.fallback_reason:
            import sys
            print(f"note: fallback to the depth backend: {m.fallback_reason}", file=sys.stderr)
        return _cli.finish(m.result, a.out)
    return _cli.guard(go)
