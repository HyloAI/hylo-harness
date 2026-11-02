"""`convex`: mesh -> convex collision parts (CoACD) and an MJCF snippet for MuJoCo.

    hylo tool convex MESH [--name NAME] [--threshold 0.05] [--max-parts N] [--mass KG] [--out DIR]

Python:
    c = convex("mug.obj", name="mug")    # Convex: mjcf (path), scene (path), parts [paths], info
    # in a MuJoCo scene:  <include file=".../mug.xml"/>   (meshes/ next to it)

The snippet has one body (free joint by default) with the visual mesh (group 2, no collisions) and one mesh geom per
convex part (group 3, collisions). It is checked by compiling it with a floor and dropping the object for 3 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import ToolResult, run_worker

MODEL = "CoACD"
LICENCE = "MIT"


@dataclass
class Convex:
    mjcf: Path
    scene: Path
    parts: list[Path]
    info: dict[str, Any]
    result: ToolResult


def convex(mesh: Any, *, name: str = "object", threshold: float = 0.05, max_parts: int = -1,
           resolution: int = 2000, mass: float | None = None, density: float = 500.0, scale: float = 1.0,
           center: bool = True, freejoint: bool = True, simulate: bool = True, preview: bool = True,
           **run_kw: Any) -> Convex:
    """Convex decomposition of `mesh` (path to .obj/.glb/.stl/.ply, or a mesh.Mesh) and its MJCF snippet."""
    path = getattr(mesh, "obj", mesh)
    req = {"name": name, "threshold": threshold, "max_parts": max_parts, "resolution": resolution, "mass": mass,
           "density": density, "scale": scale, "center": center, "freejoint": freejoint, "simulate": simulate,
           "preview": preview}
    res = run_worker("convex", req, {"mesh": path}, **run_kw)
    parts = [p for k, p in sorted(res.files.items()) if k.startswith("part_")]
    return Convex(res.files["mjcf"], res.files["scene"], parts, res.outputs, res)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("convex", __doc__)
    ap.add_argument("mesh")
    ap.add_argument("--name", default="object")
    ap.add_argument("--threshold", type=float, default=0.05, help="concavity threshold (lower: more parts)")
    ap.add_argument("--max-parts", type=int, default=-1)
    ap.add_argument("--resolution", type=int, default=2000)
    ap.add_argument("--mass", type=float, help="total mass in kg (default: from density)")
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-simulate", action="store_true", help="skip the drop test")
    a = ap.parse_args(argv)

    def go() -> int:
        c = convex(a.mesh, name=a.name, threshold=a.threshold, max_parts=a.max_parts, resolution=a.resolution,
                   mass=a.mass, density=a.density, scale=a.scale, simulate=not a.no_simulate, **_cli.run_kw(a))
        if a.out:     # keep the snippet's relative mesh paths valid
            import shutil
            dst = Path(a.out)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(c.mjcf.parent / "meshes", dst / "meshes", dirs_exist_ok=True)
            for p in (c.mjcf, c.scene, c.result.files.get("preview")):
                if p:
                    shutil.copy2(p, dst / p.name)
        return _cli.finish(c.result, None)
    return _cli.guard(go)
