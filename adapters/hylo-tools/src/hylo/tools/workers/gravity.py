# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.5", "torchvision",
#   "geocalib @ git+https://github.com/cvg/GeoCalib.git@1dedb29f94fc11b896c5f572005978aeebccd67f",
#   "opencv-python-headless", "numpy>=1.26", "pillow>=10",
# ]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["opencv-python; sys_platform == 'never'"]
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Single-image camera calibration with GeoCalib on the CPU: gravity direction (roll, pitch), focal length / field of
view and their uncertainties.

params: weights (pinhole | distorted; default pinhole), camera_model (pinhole | simple_radial | simple_divisional),
        focal_prior (pixels, optional), preview (bool, default true)
inputs: image
outputs: gravity.npz: gravity_cam (3,) unit vector of gravity (down) in the camera frame (OpenCV: x right, y down,
         z forward); R_level_cam (3, 3) rotation from the camera frame to a level frame (z up, yaw fixed so the camera
         looks along +x); K (3, 3); roll, pitch (rad); preview.png (horizon line and up arrows)
         JSON: roll_deg, pitch_deg, vfov_deg, focal_px, uncertainties (deg / px) when the model gives them
"""

from __future__ import annotations

import math

import numpy as np

from _common import load_rgb, main, setup_torch, versions

LICENCE = "Apache-2.0 (code); CC-BY-4.0 (geocalib-pinhole / geocalib-distorted weights)"


def load(params):
    torch = setup_torch()
    from geocalib import GeoCalib
    weights = params.get("weights", "pinhole")
    return {"torch": torch, "model": GeoCalib(weights=weights).to("cpu").eval(), "weights": weights}


def _f(x):
    return float(x.reshape(-1)[0]) if hasattr(x, "reshape") else float(x)


def level_rotation(g_cam: np.ndarray) -> np.ndarray:
    """R_level_cam: maps camera-frame vectors to a frame with z up (against gravity) and the camera's optical axis
    projected onto the horizontal plane as +x."""
    up = -g_cam / np.linalg.norm(g_cam)
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    if np.linalg.norm(fwd) < 1e-6:        # looking straight up or down: use the image x axis
        fwd = np.array([1.0, 0.0, 0.0]) - up * up[0]
    x = fwd / np.linalg.norm(fwd)
    y = np.cross(up, x)
    return np.stack([x, y, up])           # rows: level axes expressed in the camera frame


def preview(rgb: np.ndarray, K: np.ndarray, g_cam: np.ndarray, text: str):
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb).convert("RGB")
    H, W = rgb.shape[:2]
    d = ImageDraw.Draw(img)
    # horizon: pixels whose ray is perpendicular to gravity: (g^T K^-1) [u, v, 1] = 0
    a, b, c = g_cam @ np.linalg.inv(K)
    pts = []
    if abs(b) > 1e-9:
        pts = [(0, -c / b), (W, -(a * W + c) / b)]
    elif abs(a) > 1e-9:
        pts = [(-c / a, 0), (-c / a, H)]
    if pts:
        d.line(pts, fill=(255, 230, 0), width=3)
    # up arrows: project a short step against gravity from points at 2 m depth on a grid
    Kinv = np.linalg.inv(K)
    for v in np.linspace(0.2, 0.8, 3) * H:
        for u in np.linspace(0.2, 0.8, 4) * W:
            p = Kinv @ np.array([u, v, 1.0]) * 2.0
            q = p - 0.25 * g_cam
            if q[2] <= 0.05:
                continue
            uq, vq = (K @ (q / q[2]))[:2]
            d.line([(u, v), (uq, vq)], fill=(0, 255, 120), width=3)
            d.ellipse([uq - 4, vq - 4, uq + 4, vq + 4], fill=(0, 255, 120))
    d.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return img


def handle(st, req):
    torch, model = st["torch"], st["model"]
    rgb = load_rgb(req.input("image"))
    H, W = rgb.shape[:2]
    image = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    kw = {"camera_model": req.get("camera_model", "pinhole")}
    if req.get("focal_prior"):
        kw["priors"] = {"focal": torch.tensor([float(req.get("focal_prior"))])}
    with torch.inference_mode():
        res = model.calibrate(image, **kw)
    cam, grav = res["camera"], res["gravity"]
    roll, pitch = (float(v) for v in grav.rp.reshape(-1)[:2])
    g_cam = -grav.vec3d.reshape(-1).cpu().numpy().astype(np.float64)   # GeoCalib: up; here: down
    g_cam /= np.linalg.norm(g_cam)
    fx, fy = (float(v) for v in cam.f.reshape(-1)[:2])
    cx, cy = (float(v) for v in cam.c.reshape(-1)[:2])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    unc = {}
    for k in ("roll_uncertainty", "pitch_uncertainty", "vfov_uncertainty", "focal_uncertainty"):
        if k in res:
            v = _f(res[k].cpu().numpy())
            unc[k.replace("_uncertainty", "")] = round(math.degrees(v) if k != "focal_uncertainty" else v, 3)
    R = level_rotation(g_cam)
    req.save_npz("gravity", gravity_cam=g_cam, R_level_cam=R, K=K, roll=roll, pitch=pitch)
    vfov = math.degrees(2 * math.atan(H / (2 * fy)))
    if req.get("preview", True):
        req.save_png("preview", preview(rgb, K, g_cam, f"roll {math.degrees(roll):.1f}  pitch "
                                                       f"{math.degrees(pitch):.1f}  vfov {vfov:.1f} deg"))
    req.provenance.update({"model": f"GeoCalib ({st['weights']})", "revision": "cvg/GeoCalib@1dedb29",
                           "licence": LICENCE, "backend": "geocalib", "dtype": "float32",
                           "packages": versions("torch", "geocalib")})
    return {"roll_deg": round(math.degrees(roll), 3), "pitch_deg": round(math.degrees(pitch), 3),
            "vfov_deg": round(vfov, 3), "focal_px": round(fy, 2), "K": K.round(3).tolist(),
            "gravity_cam": g_cam.round(5).tolist(), "uncertainty": unc, "image_size": [H, W]}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
