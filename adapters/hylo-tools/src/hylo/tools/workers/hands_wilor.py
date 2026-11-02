# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "wilor-mini @ git+https://github.com/warmshao/WiLoR-mini@ebec42f94c389070cdd7dda6fd1bf0b4a659c960",
#   "torch>=2.2,<=2.5.1", "torchvision", "numpy>=1.26,<2", "opencv-python-headless<4.11", "pillow>=10",
#   "huggingface_hub", "imageio>=2.30", "imageio-ffmpeg>=0.5", "ultralytics==8.1.34", "dill", "smplx==0.1.28",
#   "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da",
# ]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["opencv-python; sys_platform == 'never'", "opencv-contrib-python; sys_platform == 'never'"]
# extra-build-dependencies = { chumpy = ["pip", "numpy<2"] }
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""3D hand poses with WiLoR (CVPR 2025) on the CPU: YOLO hand detector -> ViT -> MANO parameters, camera-frame
keypoints and vertices. The `hands` tool's primary path.

The MANO layer needs MANO_RIGHT.pkl, which only the MANO website distributes (registration at
https://mano.is.tue.mpg.de, file mano_v1_2.zip -> mano_v1_2/models/MANO_RIGHT.pkl). The request passes its path as
`mano_path`. Without it this worker runs only the detector (`mode: detect`) and refuses `mode: full` (Blocked). The
WiLoR weights come from the official repo (rolpotamias/WiLoR on Hugging Face) and the MANO mean parameters from the
official GitHub repo. WiLoR-mini (the pip package WiLoR's README points to) would otherwise download a MANO file from
a third-party repo; that download is disabled here.

params: mode ("full" | "detect"), mano_path, max_hands (2), stride (1), max_frames, fps, K (3x3, optional),
        hand_conf (0.3), rescale_factor (2.5), save_vertices (false), overlay (true)
inputs: video
outputs: hands.npz: boxes (T, Hn, 4), handedness (T, Hn) 1 right / 0 left / -1 none, score, valid; full mode also kp2d
         (T, Hn, 21, 2) px, kp3d_cam (T, Hn, 21, 3) m, kp3d_hand (T, Hn, 21, 3) m (root-relative), global_orient
         (T, Hn, 3) and hand_pose (T, Hn, 15, 3) axis-angle, betas (T, Hn, 10), cam_t (T, Hn, 3), focal; overlay.mp4/png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from _common import Blocked, main, setup_torch, versions
from _video import HAND_EDGES, draw_skeleton, panel_3d, read_video, write_mp4

REPO = "rolpotamias/WiLoR"
WILOR_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"
MEAN_URL = f"https://raw.githubusercontent.com/rolpotamias/WiLoR/{WILOR_COMMIT}/mano_data/mano_mean_params.npz"
LICENCE = ("WiLoR code and weights CC BY-NC-ND 4.0 (non-commercial); MANO licence (registration, non-commercial); "
           "ultralytics AGPL-3.0 (detector runtime)")
MANO_ACTION = ("register at https://mano.is.tue.mpg.de, download mano_v1_2.zip (Models & Code), and copy "
               "mano_v1_2/models/MANO_RIGHT.pkl to $HYLO_MANO_DIR/MANO_RIGHT.pkl "
               "(default ~/.cache/hylo/tools/body_models/mano/MANO_RIGHT.pkl)")
COLORS = {1: (255, 120, 40), 0: (60, 170, 255), -1: (200, 200, 200)}


def _cache() -> Path:
    return Path(os.environ.get("HYLO_TOOLS_CACHE") or Path.home() / ".cache" / "hylo" / "tools")


def load(params):
    torch = setup_torch()
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    det_path = hf_hub_download(REPO, "detector.pt")
    st = {"torch": torch, "detector": YOLO(det_path), "det_path": det_path, "pipe": None}
    st["detector"].to("cpu")
    return st


def _pipeline(st, mano_path: str):
    """WiLoR-mini's pipeline on a directory holding the official weights and the user's MANO file."""
    if st["pipe"] is not None:
        return st["pipe"]
    import urllib.request

    from huggingface_hub import hf_hub_download
    import wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline as wp

    root = _cache() / "wilor"
    pm = root / "pretrained_models"
    pm.mkdir(parents=True, exist_ok=True)
    links = {"MANO_RIGHT.pkl": Path(mano_path).expanduser().resolve(), "detector.pt": Path(st["det_path"]),
             "wilor_final.ckpt": Path(hf_hub_download(REPO, "wilor_final.ckpt"))}
    for name, src in links.items():
        dst = pm / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    mean = pm / "mano_mean_params.npz"
    if not mean.exists():
        urllib.request.urlretrieve(MEAN_URL, mean)

    def refuse(*a, **k):   # never fetch model files (in particular MANO) from WiLoR-mini's third-party repo
        raise RuntimeError(f"hands_wilor: refusing a download from {k.get('repo_id', a[:1])}: {k.get('filename')}")

    wp.hf_hub_download = refuse
    st["pipe"] = wp.WiLorHandPose3dEstimationPipeline(device=st["torch"].device("cpu"), dtype=st["torch"].float32,
                                                      wilor_pretrained_dir=str(root), verbose=False,
                                                      focal_length=st.get("focal", 5000))
    return st["pipe"]


def handle(st, req):
    from PIL import Image
    torch = st["torch"]
    mode = req.get("mode", "full")
    mano = req.get("mano_path")
    if mode == "full" and not (mano and Path(mano).expanduser().exists()):
        raise Blocked(f"WiLoR needs the MANO right-hand model (not found: {mano})", MANO_ACTION)
    frames, fps = read_video(req.input("video"), int(req.get("stride", 1)), req.get("max_frames"))
    if req.get("fps"):
        fps = float(req.get("fps")) / max(1, int(req.get("stride", 1)))
    T, H, W = frames.shape[:3]
    Hn = int(req.get("max_hands", 2))
    conf = float(req.get("hand_conf", 0.3))
    boxes = np.zeros((T, Hn, 4), np.float32)
    handed = np.full((T, Hn), -1, np.int8)
    score = np.zeros((T, Hn), np.float32)
    valid = np.zeros((T, Hn), bool)
    full = mode == "full"
    if full:
        if req.get("K") is not None:   # WiLoR's focal convention: the focal of a 256 px crop of the longest side
            st["focal"] = float(np.asarray(req.get("K"))[0][0]) * 256.0 / max(H, W)
        pipe = _pipeline(st, mano)
        kp2d = np.zeros((T, Hn, 21, 2), np.float32)
        kpc = np.zeros((T, Hn, 21, 3), np.float32)
        kph = np.zeros((T, Hn, 21, 3), np.float32)
        go = np.zeros((T, Hn, 3), np.float32)
        hp = np.zeros((T, Hn, 15, 3), np.float32)
        betas = np.zeros((T, Hn, 10), np.float32)
        camt = np.zeros((T, Hn, 3), np.float32)
        verts = np.zeros((T, Hn, 778, 3), np.float16) if req.get("save_vertices") else None
        focal = 0.0
    with torch.inference_mode():
        for t in range(T):
            img = np.ascontiguousarray(frames[t])
            if full:
                dets = pipe.predict(img, hand_conf=conf, rescale_factor=float(req.get("rescale_factor", 2.5)))
                found = [(int(d["is_right"]), 1.0, np.array(d["hand_bbox"], np.float32), d.get("wilor_preds"))
                         for d in dets]
            else:
                r = st["detector"](img, conf=conf, verbose=False)[0]
                found = [(int(c), float(s), b.astype(np.float32), None) for b, c, s in
                         zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            found.sort(key=lambda f: -f[1])
            labels = [f[0] for f in found[:Hn]]
            by_label = len(set(labels)) == len(labels) and Hn >= 2
            for k, (right, s, box, pred) in enumerate(found[:Hn]):
                slot = (0 if right == 1 else 1) if by_label else k
                boxes[t, slot], handed[t, slot], score[t, slot], valid[t, slot] = box, right, s, True
                if full and pred is not None:
                    k3 = pred["pred_keypoints_3d"][0]
                    ct = pred["pred_cam_t_full"][0]
                    kp2d[t, slot], kpc[t, slot], kph[t, slot] = pred["pred_keypoints_2d"][0], k3 + ct, k3 - k3[0]
                    go[t, slot] = np.asarray(pred["global_orient"]).reshape(-1)[:3]
                    hp[t, slot] = np.asarray(pred["hand_pose"]).reshape(15, 3)
                    betas[t, slot], camt[t, slot] = np.asarray(pred["betas"]).reshape(10), ct
                    focal = float(np.asarray(pred["scaled_focal_length"]).reshape(-1)[0])
                    if verts is not None:
                        verts[t, slot] = (pred["pred_vertices"][0] + ct).astype(np.float16)
    arrays = {"boxes": boxes, "handedness": handed, "score": score, "valid": valid, "fps": np.float32(fps)}
    if full:
        arrays.update(kp2d=kp2d, kp3d_cam=kpc, kp3d_hand=kph, global_orient=go, hand_pose=hp, betas=betas, cam_t=camt,
                      K=np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], np.float32))
        if verts is not None:
            arrays["vertices"] = verts
    req.save_npz("hands", **arrays)
    if req.get("overlay", True):
        from PIL import ImageDraw
        out = []
        for t in range(T):
            im = Image.fromarray(frames[t])
            d = ImageDraw.Draw(im)
            for h in range(Hn):
                if valid[t, h]:
                    c = COLORS[int(handed[t, h])]
                    d.rectangle(boxes[t, h].tolist(), outline=c, width=2)
                    d.text((boxes[t, h, 0] + 3, boxes[t, h, 1] + 2), ("R" if handed[t, h] == 1 else "L") +
                           f" {score[t, h]:.2f}", fill=c)
                    if full:
                        draw_skeleton(im, kp2d[t, h], HAND_EDGES, c)
            if full:
                p3 = panel_3d([(kpc[t, h], COLORS[int(handed[t, h])]) for h in range(Hn) if valid[t, h]], HAND_EDGES,
                              H, label="WiLoR 3D (camera frame)")
                canvas = Image.new("RGB", (W + H, H))
                canvas.paste(im, (0, 0))
                canvas.paste(p3, (W, 0))
                im = canvas
            out.append(np.asarray(im))
        write_mp4(req.path("overlay.mp4", "overlay_mp4"), out, fps=max(1.0, fps))
        req.save_png("overlay", out[int(np.argmax(valid.sum(1))) if valid.any() else 0])
    req.provenance.update({"model": f"{REPO} (detector.pt" + (", wilor_final.ckpt)" if full else " only)"),
                           "revision": _hf_rev(), "code": f"WiLoR-mini ebec42f (WiLoR {WILOR_COMMIT[:7]})",
                           "licence": LICENCE, "backend": "WiLoR" if full else "WiLoR hand detector (no MANO: detect mode)",
                           "dtype": "float32", "packages": versions("torch", "ultralytics", "wilor-mini")})
    return {"mode": mode, "frames": T, "image_size": [H, W], "fps": fps, "frames_with_hand": int(valid.any(1).sum()),
            "right_frames": int((handed == 1).sum()), "left_frames": int((handed == 0).sum())}


def _hf_rev() -> str | None:
    from _common import hf_revision
    return hf_revision(REPO)


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
