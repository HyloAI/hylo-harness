"""convex: CoACD on a non-convex cup, MJCF snippet compiled and dropped in MuJoCo; then on a carved mesh."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from tools_testkit import needs_run, record, save_example_png, write_cup_obj

from hylo.tools.convex import LICENCE, MODEL, convex

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")


@needs_run
def test_convex_cup(tmp_path):
    cup = write_cup_obj(tmp_path / "cup.obj")
    t0 = time.perf_counter()
    c = convex(cup, name="cup", mass=0.2)
    wall = time.perf_counter() - t0
    info = c.info
    assert info["parts"] >= 4                      # an open cup is not convex
    assert 0.95 < info["volume_ratio"] < 1.25      # the parts cover the cup without much excess
    assert info["drop"]["settled"] and info["drop"]["above_floor"] and info["drop"]["ncon"] > 0
    assert c.mjcf.exists() and len(c.parts) == info["parts"] and all(p.exists() for p in c.parts)
    assert "<freejoint" in c.mjcf.read_text()
    from PIL import Image
    ex = [save_example_png(Image.open(c.result.files["preview"]), "convex_cup")]
    record("convex", status="works", model=MODEL, licence=LICENCE, wall_s=wall, timing=c.result.timing, examples=ex,
           notes=(f"open cup (384 faces): {info['parts']} parts, hull/mesh volume {info['volume_ratio']}, MJCF "
                  f"compiled and dropped: rests upright after {info['drop']['steps']} steps"))


@needs_run
def test_convex_carved_mesh(scene_frame):
    from hylo.tools.mesh import mesh
    z = np.load(scene_frame / "scene.npz")
    names = list(z["names"])
    sel = (z["seg"] == names.index("mug_body")) | (z["seg"] == names.index("mug_handle"))
    m = mesh(scene_frame / "scene.png", sel, depth=z["depth"], K=z["K"], backend="depth", T_world_cam=z["T_world_cam"])
    c = convex(m, name="mug", threshold=0.08)
    assert c.info["parts"] >= 1 and c.info["drop"]["above_floor"]
    assert Path(c.mjcf).read_text().count('type="mesh"') == c.info["parts"] + 1
