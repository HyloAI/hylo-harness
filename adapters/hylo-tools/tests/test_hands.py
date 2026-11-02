"""hands: MediaPipe keypoint fallback end to end, WiLoR's hand detector (the MANO-free part of the primary path),
and the blocked message while MANO is missing. Input: a short video synthesised from MediaPipe's Apache-2.0 test
images (four gestures, slow zoom and rotation)."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from tools_testkit import copy_example, needs_run, record, save_example_png

from hylo.tools import hands as H
from hylo.tools.runner import ToolError, cache_dir

ASSETS = "https://storage.googleapis.com/mediapipe-assets/"
GESTURES = ["thumb_up.jpg", "pointing_up.jpg", "victory.jpg", "fist.jpg"]


def media(name: str) -> Path:
    """A MediaPipe test image (Apache-2.0), downloaded once into the tools cache."""
    p = cache_dir("media", "mediapipe-assets") / name
    if not p.exists():
        try:
            urllib.request.urlretrieve(ASSETS + name, p)
        except OSError as exc:
            pytest.skip(f"cannot download {name}: {exc}")
    return p


def gesture_video(size: int = 320, per: int = 8) -> np.ndarray:
    from PIL import Image
    frames = []
    for name in GESTURES:
        im = Image.open(media(name)).convert("RGB")
        for k in range(per):
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            w = int(size * 0.8 * (1.0 + 0.12 * k / per))
            r = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
            r = r.rotate(-6 + 12 * k / per, expand=True, fillcolor=(255, 255, 255))
            canvas.paste(r, ((size - r.width) // 2, (size - r.height) // 2))
            frames.append(np.asarray(canvas))
    return np.stack(frames)


def test_wilor_blocked_without_mano():
    if H.mano_path():
        pytest.skip("MANO_RIGHT.pkl is present")
    with pytest.raises(ToolError) as e:
        H.hands(np.zeros((1, 8, 8, 3), np.uint8), backend="wilor")
    assert "mano.is.tue.mpg.de" in e.value.blocked and "MANO_RIGHT.pkl" in e.value.blocked
    assert H.status()["backend"] == "mediapipe" and not H.status()["mano_present"]


@needs_run
def test_hands_video():
    video = gesture_video()
    t0 = time.perf_counter()
    tr = H.hands(video)                                    # auto: WiLoR with MANO, else MediaPipe
    wall = time.perf_counter() - t0
    T = len(video)
    assert tr.valid.shape == (T, 2) and tr.kp3d_cam.shape == (T, 2, 21, 3)
    seen = tr.valid.any(1)
    assert seen.sum() >= T - 4
    assert (tr.handedness[tr.valid] == 1).mean() > 0.8      # MediaPipe's gesture images show right hands
    first = tr.valid.argmax(1)
    kc = tr.kp3d_cam[np.arange(T), first][seen]
    assert ((kc[..., 2] > 0.1) & (kc[..., 2] < 3.0)).all()   # in front of the camera
    kh = tr.kp3d_hand[np.arange(T), first][seen]
    palm = np.linalg.norm(kh[:, 9] - kh[:, 0], axis=1)      # wrist -> middle finger MCP
    assert 0.05 < np.median(palm) < 0.13, np.median(palm)

    # the primary path without MANO: WiLoR's own hand detector
    t1 = time.perf_counter()
    det = H.hands(video, backend="detect")
    wall_det = time.perf_counter() - t1
    assert det.valid.any(1).sum() >= T - 4 and (det.handedness[det.valid] == 1).mean() > 0.8

    from PIL import Image
    ex = [save_example_png(Image.open(tr.result.files["overlay"]), f"hands_{tr.backend}"),
          copy_example(tr.result.files["overlay_mp4"], f"hands_{tr.backend}.mp4"),
          save_example_png(Image.open(det.result.files["overlay"]), "hands_wilor_detect")]
    fallback = tr.backend != "wilor"
    record("hands", status="works with fallback" if fallback else "works",
           model=tr.result.provenance.get("model", ""), licence=tr.result.provenance.get("licence", ""),
           wall_s=wall, timing=tr.result.timing, examples=ex,
           notes=(f"{T}-frame 320x320 gesture video: hand in {int(seen.sum())}/{T} frames, median palm length "
                  f"{np.median(palm) * 100:.1f} cm; WiLoR detector (no MANO) found a hand in "
                  f"{int(det.valid.any(1).sum())}/{T} frames in {wall_det:.1f} s"),
           blocked=H.MANO_ACTION if fallback else None,
           primary={"model": det.result.provenance.get("model"), "licence": det.result.provenance.get("licence"),
                    "stage": "detector only (MANO missing)", "wall_s": round(wall_det, 2), "timing": det.result.timing})
