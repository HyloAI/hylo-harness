"""mesh: the depth fallback with ground-truth depth (geometry check, no model) and the full auto pipeline
(segment -> SAM 3D access check -> MoGe-2 depth -> carved mesh) on a LIBERO frame."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tools_testkit import needs_run, record, save_example_png

from hylo.tools.mesh import LICENCE, MODEL, mesh


@pytest.mark.skipif(__import__("shutil").which("uv") is None, reason="needs uv")
def test_mesh_from_true_depth(scene_frame):
    z = np.load(scene_frame / "scene.npz")
    names = list(z["names"])
    m = mesh(scene_frame / "scene.png", z["seg"] == names.index("blue_can"), depth=z["depth"], K=z["K"],
             backend="depth", T_world_cam=z["T_world_cam"])
    assert m.backend == "depth" and m.info["watertight"] and m.glb.exists() and m.obj.exists()
    ext, centre = np.array(m.info["extents_m"]), np.array(m.info["centre"])
    assert np.allclose(ext, [0.07, 0.07, 0.12], atol=0.015), ext          # cylinder r 3.5 cm, h 12 cm
    assert np.allclose(centre, [0.10, 0.10, 0.06], atol=0.01), centre


@needs_run
def test_mesh_auto_libero(libero_frame):
    from hylo.tools.segment import segment
    seg = segment(libero_frame, ["basket"], overlay=False)
    assert len(seg) >= 1
    t0 = time.perf_counter()
    m = mesh(libero_frame, seg.masks[0])
    wall = time.perf_counter() - t0
    if m.backend == "depth":
        assert m.fallback_reason and "403" in m.fallback_reason
    assert m.info["watertight"] and 0.05 < max(m.info["extents_m"]) < 1.0
    from PIL import Image
    ex = [save_example_png(Image.open(m.result.files["preview"]), "mesh_libero_basket")]
    status = "works" if m.backend == "sam3d" else "works with fallback"
    record("mesh", status=status, model=MODEL, licence=LICENCE, wall_s=wall, timing=m.timing, examples=ex,
           notes=(f"backend {m.backend}: {m.fallback_reason or ''}; basket mask from segment, MoGe-2 depth "
                  f"{m.timing.get('depth_wall_s')} s + carve {m.timing.get('mesh_wall_s')} s; watertight, extents "
                  f"{m.info['extents_m']} m; with true depth on the rendered can: extents within 1.5 cm"))
