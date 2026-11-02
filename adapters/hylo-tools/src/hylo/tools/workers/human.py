# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.21", "opencv-contrib-python-headless<4.11", "numpy>=1.26,<2", "pillow>=10",
#                 "imageio>=2.30", "imageio-ffmpeg>=0.5"]
# [tool.uv]
# override-dependencies = ["opencv-contrib-python; sys_platform == 'never'", "opencv-python; sys_platform == 'never'"]
# ///
"""Human 3D pose per frame with MediaPipe Pose Landmarker (heavy, CPU), grounded in an approximate world frame: the
`human` tool's fallback while GVHMR waits for the SMPL / SMPL-X body models.

World grounding (approximate, labelled so in the outputs): the metric hip-centred landmarks are placed in the camera
frame by fitting their projection to the 2D landmarks under K; the camera is assumed static, with the given pitch and
roll (or R_world_cam, e.g. from the `gravity` tool), else level; the ground is the plane z = 0 under the feet (low
percentile of the foot landmarks); x, y are centred on the first frame's root. A moving camera shows up as root motion.

params: stride (1), max_frames, fps, K (3x3) or fov_deg (60), pitch_deg (0, positive: looking down), roll_deg (0),
        R_world_cam (3x3, overrides pitch/roll), smooth (5 frames, moving average of the camera-frame root),
        min_detection (0.5), min_presence (0.5), min_tracking (0.5), model ("heavy" | "full" | "lite"), overlay (true)
inputs: video
outputs: human.npz: kp2d (T, 33, 2) px, visibility (T, 33), joints_local (T, 33, 3) m (MediaPipe world landmarks,
         hip-centred, camera-aligned axes), joints_cam (T, 33, 3) m, joints_world (T, 33, 3) m (z up, ground z = 0),
         root_world (T, 3), valid (T,), R_world_cam (3, 3), camera_height, K (3, 3), fps; overlay.mp4 and overlay.png
"""

from __future__ import annotations

import numpy as np

from _common import main, versions
from _video import POSE_EDGES, default_K, draw_skeleton, fetch, fit_translation, panel_3d, read_video, write_mp4

URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{m}/float16/1/pose_landmarker_{m}.task"
LICENCE = "Apache-2.0 (MediaPipe and the Pose Landmarker model)"
FEET = [27, 28, 29, 30, 31, 32]
HIPS = [23, 24]


def load(params):
    from mediapipe.tasks.python import BaseOptions, vision
    m = params.get("model", "heavy")
    path, sha = fetch(URL.format(m=m), "mediapipe")
    return {"vision": vision, "BaseOptions": BaseOptions, "path": str(path), "sha": sha, "model": m}


def rot_world_cam(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera (x right, y down, z forward) -> world (z up, y forward): level camera, then pitch about x, roll about z."""
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    R0 = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    Rx = np.array([[1, 0, 0], [0, np.cos(-p), -np.sin(-p)], [0, np.sin(-p), np.cos(-p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return R0 @ Rx @ Rz


def _smooth(x: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or valid.sum() < 2:
        return x
    out = x.copy()
    idx = np.flatnonzero(valid)
    for j, t in enumerate(idx):
        win = idx[max(0, j - k // 2): j + k // 2 + 1]
        out[t] = x[win].mean(0)
    return out


def handle(st, req):
    import mediapipe as mp
    from PIL import Image
    vision = st["vision"]
    frames, fps = read_video(req.input("video"), int(req.get("stride", 1)), req.get("max_frames"))
    if req.get("fps"):
        fps = float(req.get("fps")) / max(1, int(req.get("stride", 1)))
    T, H, W = frames.shape[:3]
    K = np.asarray(req.get("K"), float) if req.get("K") is not None else default_K(W, H, float(req.get("fov_deg", 60)))
    opts = vision.PoseLandmarkerOptions(
        base_options=st["BaseOptions"](model_asset_path=st["path"]), running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=float(req.get("min_detection", 0.5)),
        min_pose_presence_confidence=float(req.get("min_presence", 0.5)),
        min_tracking_confidence=float(req.get("min_tracking", 0.5)))
    kp2d = np.zeros((T, 33, 2), np.float32)
    vis = np.zeros((T, 33), np.float32)
    local = np.zeros((T, 33, 3), np.float32)
    valid = np.zeros(T, bool)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for t in range(T):
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[t])),
                                      int(round(t * 1000.0 / fps)))
            if not res.pose_landmarks:
                continue
            kp2d[t] = [[p.x * W, p.y * H] for p in res.pose_landmarks[0]]
            vis[t] = [p.visibility for p in res.pose_landmarks[0]]
            local[t] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]]
            valid[t] = True
    trans = np.zeros((T, 3))
    for t in np.flatnonzero(valid):
        trans[t] = fit_translation(local[t].astype(float), kp2d[t].astype(float), K, weights=np.clip(vis[t], 0.05, 1))
    trans = _smooth(trans, valid, int(req.get("smooth", 5)))
    cam = (local + trans[:, None, :]).astype(np.float32)
    R = (np.asarray(req.get("R_world_cam"), float) if req.get("R_world_cam") is not None
         else rot_world_cam(float(req.get("pitch_deg", 0)), float(req.get("roll_deg", 0))))
    world = cam @ R.T
    cam_h = 0.0
    if valid.any():
        feet = world[valid][:, FEET, 2]
        fv = vis[valid][:, FEET] > 0.5
        ground = float(np.percentile(feet[fv] if fv.any() else feet, 5))
        cam_h = -ground
        world[..., 2] -= ground
        root0 = world[np.flatnonzero(valid)[0]][HIPS].mean(0)
        world[..., :2] -= root0[:2]
    world[~valid] = 0
    cam[~valid] = 0
    root = world[:, HIPS].mean(1)
    req.save_npz("human", kp2d=kp2d, visibility=vis, joints_local=local, joints_cam=cam,
                 joints_world=world.astype(np.float32), root_world=root.astype(np.float32), valid=valid,
                 R_world_cam=R, camera_height=np.float32(cam_h), K=K, fps=np.float32(fps))
    if req.get("overlay", True):
        out = []
        g = 1.2   # ground square half size (m)
        ground_sq = np.array([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]], float)
        to_panel = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)   # world (z up) -> panel axes (y down)
        view = {"size": H, "yaw_deg": 30.0, "pitch_deg": -15.0, "extent": 3.2, "center": np.array([0.0, -0.8, 0.0])}
        for t in range(T):
            im = Image.fromarray(frames[t])
            if valid[t]:
                draw_skeleton(im, kp2d[t], POSE_EDGES, (255, 90, 60), valid=vis[t] > 0.3)
            panel = panel_3d([(to_panel(ground_sq), (90, 100, 110))], [(0, 1), (1, 2), (2, 3), (3, 0)],
                             label="world (approx.), z up, ground z = 0", **view)
            if valid[t]:
                tr = to_panel(root[: t + 1][valid[: t + 1]])
                panel_3d([(tr, (80, 200, 255))], [(i, i + 1) for i in range(len(tr) - 1)], base=panel, radius=1,
                         **view)
                panel_3d([(to_panel(world[t]), (255, 90, 60))], POSE_EDGES, base=panel, **view)
            canvas = Image.new("RGB", (W + H, H))
            canvas.paste(im, (0, 0))
            canvas.paste(panel, (W, 0))
            out.append(np.asarray(canvas))
        write_mp4(req.path("overlay.mp4", "overlay_mp4"), out, fps=max(1.0, fps))
        req.save_png("overlay", out[len(out) // 2])
    req.provenance.update({"model": f"mediapipe/pose_landmarker_{st['model']} (float16, v1)",
                           "revision": f"sha256:{st['sha'][:16]}", "licence": LICENCE,
                           "backend": "mediapipe.tasks PoseLandmarker (fallback: 33 keypoints, no SMPL)",
                           "model_url": URL.format(m=st["model"]), "packages": versions("mediapipe")})
    return {"frames": T, "image_size": [H, W], "fps": fps, "frames_with_person": int(valid.sum()),
            "camera_height_m": round(cam_h, 3),
            "root_path_m": round(float(np.linalg.norm(np.diff(root[valid], axis=0), axis=1).sum()) if valid.sum() > 1 else 0.0, 3),
            "grounding": "approximate: static camera, given or level pitch/roll, ground under the feet"}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
