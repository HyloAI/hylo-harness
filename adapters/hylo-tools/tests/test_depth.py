"""depth: MoGe-2 on the CPU; checked against MuJoCo's metric depth on a rendered scene, example on a LIBERO frame."""

from __future__ import annotations

import math
import time

import numpy as np
from tools_testkit import needs_run, record, save_example_png

from hylo.tools.depth import LICENCE, MODEL, depth
from hylo.tools.runner import ToolServer


@needs_run
def test_depth_scene_and_libero(scene_frame, libero_frame):
    gt = np.load(scene_frame / "scene.npz")
    t0 = time.perf_counter()
    with ToolServer("depth") as srv:          # load the model once for both images
        d = depth(scene_frame / "scene.png", server=srv)
        t_scene = time.perf_counter() - t0
        lib = depth(libero_frame, server=srv)
    wall = time.perf_counter() - t0
    m = d.mask & (gt["depth"] < 5.0) & np.isfinite(d.depth)
    s = np.median(gt["depth"][m] / d.depth[m])
    absrel = float(np.mean(np.abs(s * d.depth[m] - gt["depth"][m]) / gt["depth"][m]))
    assert absrel < 0.10, absrel                         # shape: relative depth error after scale alignment
    fov_gt = 55.0
    fov = math.degrees(2 * math.atan(d.depth.shape[0] / 2 / d.K[1, 1]))
    assert abs(fov - fov_gt) < 15, fov
    assert lib.mask.mean() > 0.5 and lib.depth.shape == (387, 387)
    from PIL import Image
    ex = [save_example_png(Image.open(lib.result.files["preview"]), "depth_libero"),
          save_example_png(Image.open(d.result.files["preview"]), "depth_scene")]
    record("depth", status="works", model=MODEL, licence=LICENCE, wall_s=t_scene, timing=d.result.timing,
           examples=ex, notes=(f"480x480 rendered scene: abs-rel depth error {absrel:.3f} after median scaling; "
                               f"metric scale off by x{1 / s:.2f} on this toy scene (8 cm objects without real-world "
                               f"size cues); vfov {fov:.1f} deg (true 55); LIBERO frame inference "
                               f"{lib.result.timing['infer_s']} s; both images {wall:.1f} s with one model load"),
           absrel=round(absrel, 4), scale_ratio=round(float(1 / s), 3), vfov_deg=round(fov, 2))
