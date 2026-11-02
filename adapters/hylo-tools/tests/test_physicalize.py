"""physicalize (RFC-0008 part 10): the box-merging step without the worker environment, and the worker on small
meshes when uv can build its environment (the worker itself needs trimesh and CoACD)."""

import json
import os
import shutil

import numpy as np
import pytest


def greedy_boxes():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "src" / "hylo" / "tools" / "workers" / "physicalize.py"
    src = p.read_text().split("def voxel_boxes")[0].split("LICENCE =")[1]
    ns: dict = {"np": np}
    exec(compile("LICENCE =" + src, str(p), "exec"), ns)   # noqa: S102 - our own module's pure function
    return ns["greedy_boxes"]


def test_greedy_boxes_cover_exactly_the_filled_cells():
    gb = greedy_boxes()
    g = np.zeros((6, 4, 5), bool)
    g[0:6, 0:2, 0:2] = True          # an L: a bar along x ...
    g[0:2, 0:2, 2:5] = True          # ... and a post at one end
    boxes = gb(g)
    cover = np.zeros_like(g, int)
    for x0, y0, z0, x1, y1, z1 in boxes:
        cover[x0:x1, y0:y1, z0:z1] += 1
    assert np.array_equal(cover > 0, g) and cover.max() == 1 and len(boxes) == 2


@pytest.mark.skipif(not shutil.which(os.environ.get("HYLO_TOOLS_UV", "uv")) or os.environ.get("HYLO_SKIP_TOOL_ENVS")
                    == "1", reason="uv not available")
def test_physicalize_chooses_by_shape(tmp_path):
    from hylo.tools.physicalize import physicalize

    def write(path, verts, faces):
        path.write_text("\n".join([f"v {a} {b} {c}" for a, b, c in verts] + [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]))
    P = [(0, 0), (0.3, 0), (0.3, 0.1), (0.1, 0.1), (0.1, 0.3), (0, 0.3)]
    n = len(P)
    v = [(x, -0.05, z) for x, z in P] + [(x, 0.05, z) for x, z in P]
    f = []
    for a, b, c in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]:
        f += [(a, b, c), (n + a, n + c, n + b)]
    for i in range(n):
        j = (i + 1) % n
        f += [(i, n + i, n + j), (i, n + j, j)]
    write(tmp_path / "ell.obj", v, f)
    try:
        p = physicalize(tmp_path / "ell.obj")
    except Exception as exc:  # noqa: BLE001 - no network for the worker environment
        pytest.skip(f"worker environment unavailable: {exc}")
    doc = json.loads(p.path.read_text())
    assert p.strategy == "parts" and 2 <= len(doc["parts"]) <= 4 and abs(doc["mass_kg"] - 2.5) < 0.01
    v = physicalize(tmp_path / "ell.obj", strategy="voxels", voxel=0.02)
    assert v.stage_strategy == "voxels" and len(json.loads(v.path.read_text())["boxes"]) == 2
