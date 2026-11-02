"""`physicalize`: a visual mesh -> collision geometry for a stage entity (one box, the hull, convex parts or voxel boxes).

    hylo tool physicalize MESH [--strategy auto|box|hull|parts|voxels] [--density 500] [--mass KG] [--out DIR]

Python:
    p = physicalize("chair.obj")         # Physical: path (collision.json), strategy, info
    # in a stage: {"id": "chair", "mesh": {"path": "chair.obj"}, "body": {...},
    #              "collision": {"strategy": p.stage_strategy, "parts": "chair_collision.json"}}

`auto` measures the mesh: one box when it fills at least 90% of its oriented bounding box, its convex hull when it is
nearly convex (concavity = 1 - volume / hull volume below 0.25), convex parts (CoACD) when it is concave, and voxel
boxes (filled voxels merged greedily into boxes) when a closed mesh would need more than 16 convex parts. Mass comes
from the mesh volume and the density unless given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import ToolResult, run_worker

MODEL = "CoACD + trimesh"
LICENCE = "MIT"


@dataclass
class Physical:
    path: Path
    strategy: str
    info: dict[str, Any]
    result: ToolResult

    @property
    def stage_strategy(self) -> str:
        """The `collision.strategy` of a stage entity using this file."""
        return "voxels" if self.strategy in ("box", "voxels") else "parts"


def physicalize(mesh: Any, *, strategy: str = "auto", scale: float = 1.0, density: float = 500.0,
                mass: float | None = None, voxel: float | None = None, max_boxes: int = 256, **run_kw: Any) -> Physical:
    req = {"strategy": strategy, "scale": scale, "density": density, "mass": mass, "voxel": voxel,
           "max_boxes": max_boxes}
    res = run_worker("physicalize", req, {"mesh": str(mesh)}, **run_kw)
    return Physical(res.files["collision"], res.outputs["strategy"], res.outputs, res)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("physicalize", __doc__)
    ap.add_argument("mesh")
    ap.add_argument("--strategy", default="auto", choices=["auto", "box", "hull", "parts", "voxels"])
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--mass", type=float)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, help="voxel size in metres (voxels strategy)")
    a = ap.parse_args(argv)

    def go() -> int:
        p = physicalize(a.mesh, strategy=a.strategy, density=a.density, mass=a.mass, scale=a.scale, voxel=a.voxel,
                        **_cli.run_kw(a))
        if a.out:
            import shutil
            Path(a.out).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p.path, Path(a.out) / "collision.json")
        return _cli.finish(p.result, None)
    return _cli.guard(go)
