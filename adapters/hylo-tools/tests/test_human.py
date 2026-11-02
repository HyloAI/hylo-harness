"""human: MediaPipe Pose fallback with approximate world grounding, end to end; the GVHMR blocked message. Input: a
short camera pan / zoom synthesised from MediaPipe's Apache-2.0 pose test image (a static person in a lunge)."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from tools_testkit import copy_example, needs_run, record, save_example_png

from hylo.tools import human as HU
from hylo.tools.runner import ToolError, cache_dir


def media(name: str) -> Path:
    p = cache_dir("media", "mediapipe-assets") / name
    if not p.exists():
        try:
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-assets/" + name, p)
        except OSError as exc:
            pytest.skip(f"cannot download {name}: {exc}")
    return p


def pan_video(n: int = 24, out: tuple[int, int] = (384, 256)) -> np.ndarray:
    from PIL import Image
    im = Image.open(media("pose.jpg")).convert("RGB")
    W, H = im.size
    frames = []
    for k in range(n):
        a = k / (n - 1)
        cw = int(W * (0.95 - 0.25 * a))
        ch = int(cw * out[1] / out[0])
        cx = int(W * (0.48 + 0.04 * np.sin(2 * np.pi * a)))
        x0, y0 = max(0, cx - cw // 2), max(0, H - ch - 5)
        frames.append(np.asarray(im.crop((x0, y0, x0 + cw, y0 + ch)).resize(out, Image.LANCZOS)))
    return np.stack(frames)


def test_gvhmr_blocked():
    st = HU.status()
    assert st["backend"] == "mediapipe"
    with pytest.raises(ToolError) as e:
        HU.human(np.zeros((1, 8, 8, 3), np.uint8), backend="gvhmr")
    if st["gvhmr_files_missing"]:
        assert "smpl-x.is.tue.mpg.de" in e.value.blocked and "SMPLX_NEUTRAL.npz" in e.value.blocked


@needs_run
def test_human_video():
    video = pan_video()
    t0 = time.perf_counter()
    m = HU.human(video)
    wall = time.perf_counter() - t0
    T = len(video)
    assert m.valid.sum() == T and m.joints_world.shape == (T, 33, 3)
    J = HU.JOINTS
    w = m.joints_world[m.valid]
    feet = w[:, [J.index("left_heel"), J.index("right_heel")], 2]
    assert np.median(np.abs(feet)) < 0.15                     # feet on the ground plane
    head = w[:, J.index("nose"), 2]
    assert 1.0 < np.median(head) < 2.2, np.median(head)       # an adult's head height (a lunge lowers it)
    shoulder = np.linalg.norm(w[:, J.index("left_shoulder")] - w[:, J.index("right_shoulder")], axis=1)
    assert 0.25 < np.median(shoulder) < 0.5
    assert 0.2 < m.camera_height < 3.0
    from PIL import Image
    ex = [save_example_png(Image.open(m.result.files["overlay"]), "human_mediapipe"),
          copy_example(m.result.files["overlay_mp4"], "human_mediapipe.mp4")]
    st = HU.status()
    record("human", status="works with fallback", model=m.result.provenance.get("model", ""),
           licence=m.result.provenance.get("licence", ""), wall_s=wall, timing=m.result.timing, examples=ex,
           notes=(f"{T}-frame 384x256 pan/zoom video: person in {int(m.valid.sum())}/{T} frames; median head height "
                  f"{np.median(head):.2f} m, shoulder width {np.median(shoulder):.2f} m, camera height "
                  f"{m.camera_height:.2f} m (approximate grounding; the synthetic camera motion shows as root motion)"),
           blocked=st["blocked"], gvhmr_cpu=st["gvhmr_cpu"])
