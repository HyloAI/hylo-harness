"""`human`: video -> world-grounded human motion (GVHMR with SMPL-X; MediaPipe Pose 3D keypoints meanwhile).

    hylo tool human VIDEO [--backend auto|mediapipe|gvhmr] [--fov 60 | --K fx,fy,cx,cy] [--pitch DEG --roll DEG]
    hylo tool human --status                     what GVHMR still needs (files and where to put them)

Backends (VIDEO: a video file, a folder of images or a (T, H, W, 3) .npy):
- gvhmr: GVHMR (SIGGRAPH Asia 2024; non-commercial licence), gravity-view world-grounded SMPL-X motion. Blocked:
  needs the SMPL and SMPL-X body models (registration) and GVHMR's checkpoints (GVHMR_FILES), and GVHMR's own
  environment is CUDA-only today (a pytorch3d CUDA wheel, `.cuda()` calls); `status()` lists what is missing.
- mediapipe: MediaPipe Pose Landmarker heavy (Apache-2.0), 33 keypoints per frame: image, metric hip-centred 3D,
  camera frame (projection fit) and an approximate world frame (static camera, pitch/roll given or level, ground
  plane z = 0 under the feet). Pass pitch/roll from the `gravity` tool for a better ground.
- auto (default): mediapipe until the GVHMR path is available.

Python:
    motion = human("clip.mp4")          # HumanMotion: joints_world (T, 33, 3), root_world (T, 3), kp2d, valid
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .runner import ToolError, ToolResult, run_worker

MODEL = "GVHMR + SMPL-X (blocked); fallback MediaPipe Pose Landmarker (heavy)"
LICENCE = "GVHMR non-commercial + SMPL/SMPL-X licences; fallback Apache-2.0"

# MediaPipe Pose landmark names (index order)
JOINTS = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
          "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
          "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
          "left_heel", "right_heel", "left_foot_index", "right_foot_index"]


def body_models_dir() -> Path:
    return Path(os.environ.get("HYLO_BODY_MODELS") or Path.home() / ".cache" / "hylo" / "tools" / "body_models")


def gvhmr_dir() -> Path:
    return Path(os.environ.get("HYLO_GVHMR_DIR") or Path.home() / ".cache" / "hylo" / "tools" / "gvhmr" / "checkpoints")


# file -> where it comes from (GVHMR's docs/INSTALL.md layout, rooted at $HYLO_BODY_MODELS and $HYLO_GVHMR_DIR)
def gvhmr_files() -> dict[Path, str]:
    b, g = body_models_dir(), gvhmr_dir()
    drive = "GVHMR's Google Drive folder https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
    return {
        b / "smplx" / "SMPLX_NEUTRAL.npz": "register at https://smpl-x.is.tue.mpg.de, Download > 'SMPL-X v1.1 (NPZ+PKL)' "
                                           "(models_smplx_v1_1.zip), copy models/smplx/SMPLX_NEUTRAL.npz",
        b / "smpl" / "SMPL_NEUTRAL.pkl": "register at https://smpl.is.tue.mpg.de, Download > 'SMPL for Python' "
                                         "(SMPL_python_v.1.1.0.zip), copy smpl/models/basicmodel_neutral_lbs_10_207_0_"
                                         "v1.1.0.pkl and rename it SMPL_NEUTRAL.pkl",
        g / "gvhmr" / "gvhmr_siga24_release.ckpt": drive,
        g / "hmr2" / "epoch=10-step=25000.ckpt": drive,
        g / "vitpose" / "vitpose-h-multi-coco.pth": drive,
        g / "yolo" / "yolov8x.pt": drive,
    }


def status() -> dict[str, Any]:
    missing = {str(p): src for p, src in gvhmr_files().items() if not p.exists()}
    return {"backend": "mediapipe", "gvhmr_files_missing": missing,
            "gvhmr_cpu": "not wired: GVHMR's environment is CUDA-only (pytorch3d CUDA wheel, .cuda() calls)",
            "blocked": ("; ".join(f"{Path(p).name}: {s} -> {p}" for p, s in missing.items()) if missing else None)}


@dataclass
class HumanMotion:
    """One person per frame, 33 MediaPipe keypoints (JOINTS). Metres. joints_world: z up, ground z = 0, x/y centred on
    the first frame's root (approximate grounding, see the module docstring); joints_cam: camera frame (x right,
    y down, z forward); joints_local: hip-centred."""
    backend: str
    fps: float
    kp2d: np.ndarray
    visibility: np.ndarray
    joints_local: np.ndarray
    joints_cam: np.ndarray
    joints_world: np.ndarray
    root_world: np.ndarray
    valid: np.ndarray
    R_world_cam: np.ndarray
    camera_height: float
    K: np.ndarray
    result: ToolResult
    joint_names: tuple[str, ...] = tuple(JOINTS)
    fallback_reason: str | None = None


def human(video: Any, *, backend: str = "auto", stride: int = 1, max_frames: int | None = None,
          fps: float | None = None, K: Any = None, fov_deg: float = 60.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
          R_world_cam: Any = None, smooth: int = 5, model: str = "heavy", overlay: bool = True,
          **run_kw: Any) -> HumanMotion:
    """Human motion from a video. `video`: path (file, image folder, .npy) or (T, H, W, 3) uint8 frames."""
    if isinstance(video, (list, tuple)):
        video = np.stack([np.asarray(f) for f in video])
    st = status()
    reason = None
    if backend == "gvhmr":
        raise ToolError("human: GVHMR is not available on this machine", blocked=st["blocked"] or st["gvhmr_cpu"])
    if backend == "auto":
        backend = "mediapipe"
        reason = "GVHMR blocked: " + (st["blocked"] or st["gvhmr_cpu"])
    if backend != "mediapipe":
        raise ValueError(f"human: unknown backend {backend!r} (auto, mediapipe, gvhmr)")
    req: dict[str, Any] = {"stride": stride, "max_frames": max_frames, "fps": fps, "fov_deg": fov_deg,
                           "pitch_deg": pitch_deg, "roll_deg": roll_deg, "smooth": smooth, "model": model,
                           "overlay": overlay}
    if K is not None:
        req["K"] = np.asarray(K, float).tolist()
    if R_world_cam is not None:
        req["R_world_cam"] = np.asarray(R_world_cam, float).tolist()
    res = run_worker("human", req, {"video": video}, **run_kw)
    if reason:
        res.provenance["fallback_reason"] = reason
    a = res.arrays("human")
    return HumanMotion(backend, float(a["fps"]), a["kp2d"], a["visibility"], a["joints_local"], a["joints_cam"],
                       a["joints_world"], a["root_world"], a["valid"], a["R_world_cam"], float(a["camera_height"]),
                       a["K"], res, fallback_reason=reason)


def cli(argv: list[str]) -> int:
    import json

    from . import _cli
    ap = _cli.parser("human", __doc__)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--status", action="store_true", help="print what GVHMR still needs")
    ap.add_argument("--backend", default="auto", choices=["auto", "mediapipe", "gvhmr"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--fps", type=float, help="frame rate of an image folder / array input")
    ap.add_argument("--fov", type=float, default=60.0, help="horizontal field of view when no intrinsics are known")
    ap.add_argument("--K", help="intrinsics fx,fy,cx,cy (pixels)")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera pitch in degrees (positive: looking down)")
    ap.add_argument("--roll", type=float, default=0.0, help="camera roll in degrees")
    ap.add_argument("--model", default="heavy", choices=["heavy", "full", "lite"])
    a = ap.parse_args(argv)
    if a.status or not a.video:
        print(json.dumps(status(), indent=2))
        return 0
    K = None
    if a.K:
        fx, fy, cx, cy = (float(x) for x in a.K.split(","))
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return _cli.guard(lambda: _cli.finish(human(a.video, backend=a.backend, stride=a.stride, max_frames=a.max_frames,
                                                fps=a.fps, K=K, fov_deg=a.fov, pitch_deg=a.pitch, roll_deg=a.roll,
                                                model=a.model, **_cli.run_kw(a)).result, a.out))
