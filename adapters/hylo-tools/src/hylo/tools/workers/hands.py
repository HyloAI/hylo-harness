# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.21", "opencv-contrib-python-headless<4.11", "numpy>=1.26,<2", "pillow>=10",
#                 "imageio>=2.30", "imageio-ffmpeg>=0.5"]
# [tool.uv]
# override-dependencies = ["opencv-contrib-python; sys_platform == 'never'", "opencv-python; sys_platform == 'never'"]
# ///
"""Per-frame hand keypoints from a video with MediaPipe Hand Landmarker (CPU): the keypoint-level fallback of the
`hands` tool while the MANO-based model (WiLoR, workers/hands_wilor.py) waits for the MANO files. mediapipe is pinned
to 0.10.21: the 1.x wheels link libEGL / libGLESv2, which a headless workstation may not have.

params: max_hands (2), stride (1), max_frames (none), fps (for array input, 30), K (3x3, optional), fov_deg (60, when
        no K), mirrored (false: the video is not a selfie view), min_detection (0.5), min_presence (0.5),
        min_tracking (0.5), overlay (true)
inputs: video (video file, image folder or (T, H, W, 3) .npy)
outputs: hands.npz: kp2d (T, Hn, 21, 2) px; kp3d_hand (T, Hn, 21, 3) m, MediaPipe world landmarks (origin at the
         hand's centre, camera-aligned axes); kp3d_cam (T, Hn, 21, 3) m, the same placed in the camera frame by fitting
         its projection to kp2d (approximate: depends on K); handedness (T, Hn) 1 right / 0 left / -1 none;
         score (T, Hn); valid (T, Hn); K (3, 3); fps; overlay.mp4 and overlay.png
"""

from __future__ import annotations

import numpy as np

from _common import main, versions
from _video import HAND_EDGES, default_K, draw_skeleton, fetch, fit_translation, panel_3d, read_video, write_mp4

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LICENCE = "Apache-2.0 (MediaPipe and the Hand Landmarker model)"
COLORS = {1: (255, 120, 40), 0: (60, 170, 255), -1: (200, 200, 200)}


def load(params):
    from mediapipe.tasks.python import BaseOptions, vision
    path, sha = fetch(MODEL_URL, "mediapipe")
    return {"vision": vision, "BaseOptions": BaseOptions, "path": str(path), "sha": sha}


def handle(st, req):
    import mediapipe as mp
    from PIL import Image
    vision = st["vision"]
    frames, fps = read_video(req.input("video"), int(req.get("stride", 1)), req.get("max_frames"))
    if req.get("fps"):
        fps = float(req.get("fps")) / max(1, int(req.get("stride", 1)))
    T, H, W = frames.shape[:3]
    K = np.asarray(req.get("K"), float) if req.get("K") is not None else default_K(W, H, float(req.get("fov_deg", 60)))
    Hn = int(req.get("max_hands", 2))
    opts = vision.HandLandmarkerOptions(
        base_options=st["BaseOptions"](model_asset_path=st["path"]), running_mode=vision.RunningMode.VIDEO,
        num_hands=Hn, min_hand_detection_confidence=float(req.get("min_detection", 0.5)),
        min_hand_presence_confidence=float(req.get("min_presence", 0.5)),
        min_tracking_confidence=float(req.get("min_tracking", 0.5)))
    kp2d = np.zeros((T, Hn, 21, 2), np.float32)
    kp3d = np.zeros((T, Hn, 21, 3), np.float32)
    kpc = np.zeros((T, Hn, 21, 3), np.float32)
    handed = np.full((T, Hn), -1, np.int8)
    score = np.zeros((T, Hn), np.float32)
    valid = np.zeros((T, Hn), bool)
    mirrored = bool(req.get("mirrored", False))
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for t in range(T):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[t]))
            res = lm.detect_for_video(img, int(round(t * 1000.0 / fps)))
            found = []
            for i, lms in enumerate(res.hand_landmarks):
                cat = res.handedness[i][0]
                # the Tasks API labels handedness for an unmirrored image (checked on MediaPipe's right_hands.jpg)
                right = (cat.category_name == "Right") != mirrored
                uv = np.array([[p.x * W, p.y * H] for p in lms], np.float32)
                w3 = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], np.float32)
                found.append((int(right), float(cat.score), uv, w3))
            labels = [f[0] for f in found]
            by_label = len(set(labels)) == len(labels) and Hn >= 2
            for k, (r, s, uv, w3) in enumerate(found[:Hn]):
                slot = (0 if r == 1 else 1) if by_label else k
                kp2d[t, slot], kp3d[t, slot], handed[t, slot], score[t, slot], valid[t, slot] = uv, w3, r, s, True
                kpc[t, slot] = w3 + fit_translation(w3.astype(float), uv.astype(float), K)
    req.save_npz("hands", kp2d=kp2d, kp3d_hand=kp3d, kp3d_cam=kpc, handedness=handed, score=score, valid=valid,
                 K=K, fps=np.float32(fps))
    if req.get("overlay", True):
        out = []
        for t in range(T):
            im = Image.fromarray(frames[t])
            for h in range(Hn):
                if valid[t, h]:
                    draw_skeleton(im, kp2d[t, h], HAND_EDGES, COLORS[int(handed[t, h])])
            sk = [(kpc[t, h], COLORS[int(handed[t, h])]) for h in range(Hn) if valid[t, h]]
            p3 = panel_3d(sk, HAND_EDGES, H, label="3D (camera frame)")
            canvas = Image.new("RGB", (W + p3.width, H))
            canvas.paste(im, (0, 0))
            canvas.paste(p3, (W, 0))
            out.append(np.asarray(canvas))
        write_mp4(req.path("overlay.mp4", "overlay_mp4"), out, fps=max(1.0, fps))
        mid = int(np.argmax(valid.sum(1))) if valid.any() else 0
        req.save_png("overlay", out[mid])
    req.provenance.update({"model": "mediapipe/hand_landmarker (float16, v1)", "revision": f"sha256:{st['sha'][:16]}",
                           "licence": LICENCE, "backend": "mediapipe.tasks HandLandmarker (fallback: keypoints, no MANO)",
                           "model_url": MODEL_URL, "packages": versions("mediapipe")})
    per = valid.sum(0).tolist()
    return {"frames": T, "image_size": [H, W], "fps": fps, "hands_per_slot": per,
            "frames_with_hand": int(valid.any(1).sum()),
            "right_frames": int((handed == 1).sum()), "left_frames": int((handed == 0).sum()),
            "kp3d_cam_note": "approximate: MediaPipe world landmarks translated to fit kp2d under K"}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
