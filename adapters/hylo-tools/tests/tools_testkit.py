"""Shared helpers for the tool tests (imported by conftest and the tests; pytest runs with importlib mode).

The model tests run each tool on the CPU in its own environment (minutes on first use: environment build and weight
download). They are skipped unless HYLO_TOOLS_RUN=1. Each records whether it ran, its CPU wall time and its
qualitative output under examples/ (examples/runs/<tool>.json, PNG/MP4 files of at most 1 MB).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
STRIP = HERE.parents[3] / "docs" / "docs" / "assets" / "figures" / "libero-episode-strip.png"
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
MAX_EXAMPLE_BYTES = 1_000_000

RUN = os.environ.get("HYLO_TOOLS_RUN") == "1"
needs_run = pytest.mark.skipif(not RUN, reason="model test: set HYLO_TOOLS_RUN=1 (runs the tool on the CPU)")


def gl_env() -> dict[str, str]:
    """Environment for MuJoCo offscreen rendering: the user-space OSMesa when present, else the caller's."""
    env = dict(os.environ)
    if not env.get("MUJOCO_GL") and (USER_MESA / "libOSMesa.so.8").exists():
        env.update({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                    "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), env.get("LD_LIBRARY_PATH")]))})
    return env


def save_example_png(image, name: str, max_side: int = 768) -> Path:
    """Save a qualitative output under examples/ (downscaled until it is at most 1 MB)."""
    from PIL import Image
    img = image if hasattr(image, "save") else Image.fromarray(np.asarray(image))
    img = img.convert("RGB")
    EXAMPLES.mkdir(exist_ok=True)
    p = EXAMPLES / f"{name}.png"
    while True:
        if max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        img.save(p, optimize=True)
        if p.stat().st_size <= MAX_EXAMPLE_BYTES:
            return p
        max_side = int(max(img.size) * 0.8)


def copy_example(src: Path, name: str) -> Path:
    """Copy a file (e.g. an MP4) into examples/; fails if it is larger than 1 MB."""
    import shutil
    EXAMPLES.mkdir(exist_ok=True)
    dst = EXAMPLES / name
    shutil.copy2(src, dst)
    assert dst.stat().st_size <= MAX_EXAMPLE_BYTES, f"{dst} is {dst.stat().st_size} bytes (> 1 MB)"
    return dst


def record(tool: str, *, status: str, model: str, licence: str, wall_s: float, timing: dict | None = None,
           examples: list[Path] | None = None, notes: str = "", **extra) -> None:
    """Record one tool run in examples/runs/<tool>.json (replaced on each run)."""
    d = EXAMPLES / "runs"
    d.mkdir(parents=True, exist_ok=True)
    body = {"tool": tool, "status": status, "model": model, "licence": licence, "cpu_wall_s": round(float(wall_s), 2),
            "timing": timing or {}, "examples": [str(Path(e).relative_to(EXAMPLES.parent)) for e in examples or []],
            "notes": notes, "date": time.strftime("%Y-%m-%d"),
            "threads": int(os.environ.get("HYLO_TOOLS_THREADS", "32")), **extra}
    (d / f"{tool}.json").write_text(json.dumps(body, indent=2) + "\n")


def write_cup_obj(path: Path, radius: float = 0.04, height: float = 0.09, wall: float = 0.006, n: int = 48) -> Path:
    """A watertight open-top cup (a thick-walled lathe surface, metres) as an OBJ: a non-convex test mesh."""
    prof = [(radius, 0.0), (radius, height), (radius - wall, height), (radius - wall, wall)]   # bottom-out, r > 0
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = [(0.0, 0.0, 0.0), (0.0, 0.0, wall)]                       # bottom pole, inner floor pole
    for r, z in prof:
        V += [(r * np.cos(t), r * np.sin(t), z) for t in a]
    F = []
    ring = lambda k, i: 2 + k * n + (i % n)                       # noqa: E731
    for i in range(n):
        F.append((0, ring(0, i + 1), ring(0, i)))                  # bottom disc (normal -z)
        for k in range(len(prof) - 1):
            p, q, r_, s = ring(k, i), ring(k, i + 1), ring(k + 1, i + 1), ring(k + 1, i)
            F += [(p, q, r_), (p, r_, s)]
        F.append((1, ring(3, i), ring(3, i + 1)))                  # inner floor (normal +z)
    with open(path, "w") as f:
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V)
        f.writelines(f"f {i + 1} {j + 1} {k + 1}\n" for i, j, k in F)
    return path
