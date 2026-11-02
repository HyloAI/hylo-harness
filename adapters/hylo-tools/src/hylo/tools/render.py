"""`render`: a MuJoCo scene or trajectory -> MP4 and a still, rendered offline on the CPU with a better look than
MuJoCo's default (Blender Cycles; MuJoCo offscreen fallback).

    hylo tool render SCENE.xml [--trajectory traj.npz] [--simulate SECONDS] [--camera orbit|fixed|track|NAME]
                     [--backend auto|blender|mujoco] [--width 640 --height 360 --fps 30 --samples 32] [--out DIR]

Python:
    r = render("scene.xml", simulate=2.0)                  # Rendering: video, still, backend, frames, result
    r = render("scene.xml", trajectory={"qpos": qpos, "time": t}, camera="track")

Trajectory: .npz (or a dict) with qpos (T, nq) and optionally time (T,) seconds or fps; resampled to --fps. Without a
trajectory the scene is simulated for --simulate seconds, or shown static for --duration seconds with the camera
path. Camera paths: orbit (eased arc of --orbit-degrees around the scene), fixed, track (follows the moving bodies),
or a camera of the MJCF.

Backends: `blender` runs the external Blender program ($HYLO_BLENDER, default ~/.local/opt/blender/blender; extra
shared libraries from $HYLO_BLENDER_LIBS, os.pathsep-separated) with Cycles on the CPU: principled materials, soft
area lights, a sky gradient, ground plane, AgX colour, OpenImageDenoise. `mujoco` uses MuJoCo's offscreen renderer
(OSMesa or EGL; the user-space OSMesa under ~/.local/mesa is used when MUJOCO_GL is unset) with a three-light rig,
4096 shadow map, 8x MSAA, 2x supersampling, sky and floor. `auto` (default): blender, falling back to mujoco.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runner import ToolResult, run_worker

MODEL = "Blender Cycles (CPU) / MuJoCo offscreen"
LICENCE = "Blender GPL-3.0-or-later (external program); MuJoCo Apache-2.0"
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


@dataclass
class Rendering:
    video: Path | None       # MP4 (None for a single frame)
    still: Path              # PNG of the middle frame
    backend: str             # "blender" or "mujoco"
    frames: int
    fallback_reason: str | None
    result: ToolResult


def gl_env() -> dict[str, str]:
    """Worker environment for MuJoCo offscreen rendering: the user-space OSMesa when MUJOCO_GL is not set."""
    if os.environ.get("MUJOCO_GL") or not (USER_MESA / "libOSMesa.so.8").exists():
        return {}
    return {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
            "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}


def render(scene: str | Path, trajectory: str | Path | Mapping[str, Any] | None = None, *, backend: str = "auto",
           width: int = 640, height: int = 360, fps: float = 30, samples: int = 32, camera: str = "orbit",
           azimuth: float = 135.0, elevation: float = -25.0, orbit_degrees: float = 90.0, fovy: float = 40.0,
           distance: float | None = None, lookat: list[float] | None = None, duration: float = 3.0,
           simulate: float = 0.0, keyframe: int | None = None, max_frames: int = 300, supersample: int = 2,
           still_only: bool = False, env: Mapping[str, str] | None = None, **run_kw: Any) -> Rendering:
    """Render an MJCF scene (path) and optional trajectory to MP4 + still PNG. See the module doc for options."""
    req = {"backend": backend, "width": width, "height": height, "fps": fps, "samples": samples, "camera": camera,
           "azimuth": azimuth, "elevation": elevation, "orbit_degrees": orbit_degrees, "fovy": fovy,
           "distance": distance, "lookat": lookat, "duration": duration, "simulate": simulate, "keyframe": keyframe,
           "max_frames": max_frames, "supersample": supersample, "still_only": still_only}
    inputs: dict[str, Any] = {"scene": Path(scene)}
    if trajectory is not None:
        inputs["trajectory"] = dict(trajectory) if isinstance(trajectory, Mapping) else Path(trajectory)
    res = run_worker("render", req, inputs, env={**gl_env(), **(env or {})}, **run_kw)
    o = res.outputs
    return Rendering(res.files.get("video"), res.files["still"], o["backend"], o["frames"], o.get("fallback_reason"),
                     res)


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("render", __doc__)
    ap.add_argument("scene", help="MJCF file")
    ap.add_argument("--trajectory", help=".npz with qpos (T, nq) [, time (T,) | fps]")
    ap.add_argument("--simulate", type=float, default=0.0, help="simulate this many seconds (no trajectory)")
    ap.add_argument("--duration", type=float, default=3.0, help="static scene: seconds of camera path")
    ap.add_argument("--backend", choices=["auto", "blender", "mujoco"], default="auto")
    ap.add_argument("--camera", default="orbit", help="orbit, fixed, track or an MJCF camera name")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-25.0)
    ap.add_argument("--orbit-degrees", type=float, default=90.0)
    ap.add_argument("--fovy", type=float, default=40.0)
    ap.add_argument("--distance", type=float, help="camera distance (default: frames the scene)")
    ap.add_argument("--lookat", type=float, nargs=3, help="camera target x y z (default: scene centre)")
    ap.add_argument("--keyframe", type=int)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--still", action="store_true", help="render one frame only")
    a = ap.parse_args(argv)
    return _cli.guard(lambda: _cli.finish(render(
        a.scene, a.trajectory, backend=a.backend, width=a.width, height=a.height, fps=a.fps, samples=a.samples,
        camera=a.camera, azimuth=a.azimuth, elevation=a.elevation, orbit_degrees=a.orbit_degrees, fovy=a.fovy,
        distance=a.distance, lookat=a.lookat,
        duration=a.duration, simulate=a.simulate, keyframe=a.keyframe, max_frames=a.max_frames, still_only=a.still,
        **_cli.run_kw(a)).result, a.out))
