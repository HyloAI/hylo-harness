"""`hands`: video -> per-frame 3D hand poses (WiLoR with MANO; MediaPipe Hands keypoints while MANO is missing).

    hylo tool hands VIDEO [--backend auto|wilor|mediapipe|detect] [--max-hands 2] [--fov 60] [--out DIR]
    hylo tool hands --status                     which backend would run, and what is missing for WiLoR

Backends (VIDEO: a video file, a folder of images or a (T, H, W, 3) .npy):
- wilor: WiLoR (CVPR 2025, CC BY-NC-ND 4.0): YOLO hand detector -> ViT -> MANO pose, shape, camera-frame keypoints
  and vertices. Needs MANO_RIGHT.pkl from https://mano.is.tue.mpg.de (registration; see MANO_ACTION) at
  $HYLO_MANO_DIR/MANO_RIGHT.pkl (default ~/.cache/hylo/tools/body_models/mano/MANO_RIGHT.pkl).
- mediapipe: MediaPipe Hand Landmarker (Apache-2.0): 21 keypoints per hand, image and metric hand-centred 3D,
  placed in the camera frame by fitting the 2D projection (approximate; depends on the intrinsics).
- detect: WiLoR's hand detector only (boxes and handedness), the part of the WiLoR path that runs without MANO.
- auto (default): wilor when the MANO file is present, else mediapipe.

Python:
    track = hands("clip.mp4")           # HandTrack: kp2d (T, Hn, 21, 2), kp3d_cam (T, Hn, 21, 3), handedness, valid
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .runner import ToolError, ToolResult, run_worker

MODEL = "WiLoR (rolpotamias/WiLoR) + MANO; fallback MediaPipe Hand Landmarker"
LICENCE = "WiLoR CC BY-NC-ND 4.0 + MANO licence (non-commercial); fallback Apache-2.0"
MANO_ACTION = ("register at https://mano.is.tue.mpg.de, download mano_v1_2.zip (Downloads > Models & Code), unzip "
               "and copy mano_v1_2/models/MANO_RIGHT.pkl to $HYLO_MANO_DIR/MANO_RIGHT.pkl "
               "(default ~/.cache/hylo/tools/body_models/mano/MANO_RIGHT.pkl)")
WORKERS = {"wilor": "hands_wilor", "detect": "hands_wilor", "mediapipe": "hands"}


def mano_dir() -> Path:
    """$HYLO_MANO_DIR, else $HYLO_BODY_MODELS/mano (default ~/.cache/hylo/tools/body_models/mano)."""
    if os.environ.get("HYLO_MANO_DIR"):
        return Path(os.environ["HYLO_MANO_DIR"]).expanduser()
    root = os.environ.get("HYLO_BODY_MODELS") or Path.home() / ".cache" / "hylo" / "tools" / "body_models"
    return Path(root).expanduser() / "mano"


def mano_path() -> Path | None:
    """The user's MANO right-hand model file, when present."""
    p = mano_dir() / "MANO_RIGHT.pkl"
    return p if p.exists() else None


def status() -> dict[str, Any]:
    p = mano_path()
    return {"backend": "wilor" if p else "mediapipe", "mano_right": str(p or mano_dir() / "MANO_RIGHT.pkl"),
            "mano_present": p is not None, "blocked": None if p else MANO_ACTION}


@dataclass
class HandTrack:
    """Per-frame hands (Hn slots; with two distinct hands slot 0 is the right hand, slot 1 the left).
    kp2d px; kp3d_cam metres in the camera frame (x right, y down, z forward); kp3d_hand metres, hand-centred (or
    root-relative for WiLoR); handedness 1 right / 0 left / -1 none. `mano` holds WiLoR's MANO parameters
    (global_orient, hand_pose, betas, cam_t) when the wilor backend ran."""
    backend: str
    fps: float
    kp2d: np.ndarray | None
    kp3d_cam: np.ndarray | None
    kp3d_hand: np.ndarray | None
    handedness: np.ndarray
    valid: np.ndarray
    score: np.ndarray
    K: np.ndarray | None
    boxes: np.ndarray | None
    result: ToolResult
    mano: dict[str, np.ndarray] = field(default_factory=dict)
    fallback_reason: str | None = None


def hands(video: Any, *, backend: str = "auto", max_hands: int = 2, stride: int = 1, max_frames: int | None = None,
          fps: float | None = None, K: Any = None, fov_deg: float = 60.0, overlay: bool = True,
          save_vertices: bool = False, **run_kw: Any) -> HandTrack:
    """3D hand poses per frame. `video`: path (file, image folder, .npy) or (T, H, W, 3) uint8 frames."""
    if isinstance(video, (list, tuple)):
        video = np.stack([np.asarray(f) for f in video])
    reason = None
    if backend == "auto":
        backend = "wilor" if mano_path() else "mediapipe"
        reason = None if backend == "wilor" else f"MANO_RIGHT.pkl missing: {MANO_ACTION}"
    if backend not in WORKERS:
        raise ValueError(f"hands: unknown backend {backend!r} (auto, wilor, mediapipe, detect)")
    if backend == "wilor" and not mano_path():
        raise ToolError("hands: WiLoR needs the MANO right-hand model", blocked=MANO_ACTION)
    req: dict[str, Any] = {"max_hands": max_hands, "stride": stride, "max_frames": max_frames, "fps": fps,
                           "fov_deg": fov_deg, "overlay": overlay}
    if K is not None:
        req["K"] = np.asarray(K, float).tolist()
    if backend in ("wilor", "detect"):
        req.update(mode="full" if backend == "wilor" else "detect", save_vertices=save_vertices,
                   mano_path=str(mano_path()) if mano_path() else None)
    res = run_worker(WORKERS[backend], req, {"video": video}, **run_kw)
    a = res.arrays("hands")
    mano = {k: a[k] for k in ("global_orient", "hand_pose", "betas", "cam_t", "vertices") if k in a}
    if reason:
        res.provenance["fallback_reason"] = reason
    return HandTrack(backend, float(a["fps"]), a.get("kp2d"), a.get("kp3d_cam"), a.get("kp3d_hand"), a["handedness"],
                     a["valid"], a["score"], a.get("K"), a.get("boxes"), res, mano, reason)


def cli(argv: list[str]) -> int:
    import json

    from . import _cli
    ap = _cli.parser("hands", __doc__)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--status", action="store_true", help="print which backend would run and what is missing")
    ap.add_argument("--backend", default="auto", choices=["auto", "wilor", "mediapipe", "detect"])
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--fps", type=float, help="frame rate of an image folder / array input")
    ap.add_argument("--fov", type=float, default=60.0, help="horizontal field of view when no intrinsics are known")
    ap.add_argument("--vertices", action="store_true", help="also save MANO vertices (wilor)")
    a = ap.parse_args(argv)
    if a.status or not a.video:
        print(json.dumps(status(), indent=2))
        return 0
    return _cli.guard(lambda: _cli.finish(hands(a.video, backend=a.backend, max_hands=a.max_hands, stride=a.stride,
                                                max_frames=a.max_frames, fps=a.fps, fov_deg=a.fov,
                                                save_vertices=a.vertices, **_cli.run_kw(a)).result, a.out))
