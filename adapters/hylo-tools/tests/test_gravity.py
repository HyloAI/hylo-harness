"""gravity: GeoCalib on the CPU; roll / pitch / field of view checked on a rendered scene with a known camera."""

from __future__ import annotations

import math
import time

import numpy as np
from tools_testkit import needs_run, record, save_example_png

from hylo.tools.gravity import LICENCE, MODEL, gravity, roll_pitch_from_R
from hylo.tools.runner import ToolServer


@needs_run
def test_gravity_scene_and_libero(scene_frame, libero_frame):
    gt = np.load(scene_frame / "scene.npz")
    roll_gt, pitch_gt = roll_pitch_from_R(gt["T_world_cam"][:3, :3])
    t0 = time.perf_counter()
    with ToolServer("gravity") as srv:
        g = gravity(scene_frame / "scene.png", server=srv)
        t_scene = time.perf_counter() - t0
        lib = gravity(libero_frame, server=srv)
    droll, dpitch = math.degrees(g.roll - roll_gt), math.degrees(g.pitch - pitch_gt)
    assert abs(droll) < 5 and abs(dpitch) < 10, (droll, dpitch)
    assert abs(g.vfov_deg - 55.0) < 15
    # R_level_cam maps the camera's gravity to straight down
    assert np.allclose(g.R_level_cam @ g.gravity_cam, [0, 0, -1], atol=1e-6)
    assert lib.pitch < 0                                    # the LIBERO camera looks down at the table
    from PIL import Image
    ex = [save_example_png(Image.open(lib.result.files["preview"]), "gravity_libero"),
          save_example_png(Image.open(g.result.files["preview"]), "gravity_scene")]
    record("gravity", status="works", model=MODEL, licence=LICENCE, wall_s=t_scene, timing=g.result.timing,
           examples=ex, notes=(f"480x480 rendered scene: roll error {droll:+.1f} deg, pitch error {dpitch:+.1f} deg "
                               f"(true pitch {math.degrees(pitch_gt):.1f}), vfov {g.vfov_deg:.1f} deg (true 55); "
                               f"LIBERO frame: roll {math.degrees(lib.roll):.1f}, pitch {math.degrees(lib.pitch):.1f} "
                               f"deg"),
           roll_err_deg=round(droll, 2), pitch_err_deg=round(dpitch, 2))
