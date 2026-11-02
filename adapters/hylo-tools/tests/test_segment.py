"""segment: SAM 3 on the CPU on a LIBERO frame; the hylo.manip segmenter interface on a rendered scene."""

from __future__ import annotations

import time

import numpy as np
from tools_testkit import needs_run, record, save_example_png

from hylo.tools.segment import LICENCE, MODEL, Sam3Segmenter, segment


@needs_run
def test_segment_libero(libero_frame):
    t0 = time.perf_counter()
    seg = segment(libero_frame, ["basket", "soup can", "milk carton"])
    wall = time.perf_counter() - t0
    assert "basket" in seg.labels and "soup can" in seg.labels
    assert seg.masks.shape[1:] == (387, 387) and (seg.scores >= 0.5).all()
    from PIL import Image
    ex = save_example_png(Image.open(seg.result.files["overlay"]), "segment_libero")
    record("segment", status="works", model=MODEL, licence=LICENCE, wall_s=wall, timing=seg.result.timing,
           examples=[ex], notes=f"3 prompts on a 387x387 LIBERO frame: {len(seg)} instances; "
                                f"infer {seg.result.timing['infer_s']} s (image encoder once, ~2 s per prompt)")


@needs_run
def test_manip_segmenter(scene_frame):
    from hylo.manip.datatypes import CameraView, View
    from PIL import Image
    rgb = np.asarray(Image.open(scene_frame / "scene.png").convert("RGB"))
    view = View(cams={"front": CameraView(rgb=rgb, depth=None, K=None, T_world_cam=None)}, points=np.zeros((0, 3)))
    seg = Sam3Segmenter()
    try:
        masks = seg.segment(view, ["red box", "blue can"])
        again = seg(rgb, ["green ball"])            # the persistent worker answers a second call
    finally:
        seg.close()
    assert {m.label for m in masks} == {"red box", "blue can"}
    assert all(m.camera == "front" and m.mask.shape == rgb.shape[:2] for m in masks)
    assert again and again[0][0] == "green ball"
    # the masks land on the right objects: compare with MuJoCo's own segmentation
    z = np.load(scene_frame / "scene.npz")
    names = list(z["names"])
    for m in masks:
        gt = z["seg"] == names.index("red_box" if m.label == "red box" else "blue_can")
        iou = (gt & m.mask).sum() / max(1, (gt | m.mask).sum())
        assert iou > 0.7, (m.label, iou)
