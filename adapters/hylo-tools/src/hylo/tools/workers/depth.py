# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.5", "torchvision",
#   "moge @ git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec",
#   "opencv-python-headless", "numpy>=1.26", "pillow>=10", "matplotlib",
# ]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["opencv-python; sys_platform == 'never'", "gradio; sys_platform == 'never'"]
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Monocular metric geometry from one image with MoGe-2 on the CPU: depth, camera-frame points, normals, intrinsics.

MoGe is pinned to the last MoGe-2 commit (before MoGe-3 added the CUDA-only FlexGEMM extension).

params: model (Ruicheng/moge-2-vitl-normal), fov_x (degrees; optional hint, else estimated), resolution_level (0-9,
        default 9), num_tokens (optional), preview (bool, default true)
inputs: image
outputs: depth.npz: depth (H, W) metres (inf where invalid), points (H, W, 3) camera frame (OpenCV: x right, y down,
         z forward), normal (H, W, 3) (when the model has a normal head), mask (H, W) bool, K (3, 3) pixels,
         K_normalized (3, 3); preview.png (image | depth | normals)
"""

from __future__ import annotations

import numpy as np

from _common import hf_revision, load_rgb, main, setup_torch, versions

LICENCE = "MIT (code and MoGe-2 weights; DINOv2 backbone Apache-2.0)"


def load(params):
    torch = setup_torch()
    from moge.model.v2 import MoGeModel
    repo = params.get("model", "Ruicheng/moge-2-vitl-normal")
    model = MoGeModel.from_pretrained(repo).to("cpu").eval()
    return {"torch": torch, "model": model, "repo": repo}


def colourise(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import matplotlib
    d = np.where(mask, depth, np.nan)
    lo, hi = np.nanpercentile(d, 1), np.nanpercentile(d, 99)
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (matplotlib.colormaps["turbo"](1 - np.nan_to_num(x, nan=0.0))[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def handle(st, req):
    torch, model = st["torch"], st["model"]
    rgb = load_rgb(req.input("image"))
    H, W = rgb.shape[:2]
    x = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    kw = {"resolution_level": int(req.get("resolution_level", 9)), "use_fp16": False}
    if req.get("fov_x") is not None:
        kw["fov_x"] = float(req.get("fov_x"))
    if req.get("num_tokens") is not None:
        kw["num_tokens"] = int(req.get("num_tokens"))
    with torch.inference_mode():
        out = model.infer(x, **kw)
    o = {k: v.cpu().numpy() for k, v in out.items() if hasattr(v, "cpu")}
    mask = o["mask"].astype(bool)
    Kn = o["intrinsics"].astype(np.float64)
    K = Kn.copy()
    K[0] *= W
    K[1] *= H
    arrays = {"depth": np.where(mask, o["depth"], np.inf).astype(np.float32), "points": o["points"].astype(np.float32),
              "mask": mask, "K": K, "K_normalized": Kn}
    if "normal" in o:
        arrays["normal"] = o["normal"].astype(np.float32)
    req.save_npz("depth", **arrays)
    if req.get("preview", True):
        panels = [rgb, colourise(o["depth"], mask)]
        if "normal" in o:
            n = ((o["normal"] * [0.5, -0.5, -0.5] + 0.5).clip(0, 1) * 255).astype(np.uint8)
            n[~mask] = 0
            panels.append(n)
        req.save_png("preview", np.concatenate(panels, axis=1))
    d = o["depth"][mask]
    req.provenance.update({"model": st["repo"], "revision": hf_revision(st["repo"]), "licence": LICENCE,
                           "backend": "moge.model.v2 (microsoft/MoGe@b942f00)", "dtype": "float32",
                           "packages": versions("torch", "moge")})
    return {"image_size": [H, W], "K": K.round(3).tolist(),
            "fov_x_deg": round(float(np.degrees(2 * np.arctan(0.5 / Kn[0, 0]))), 2),
            "fov_y_deg": round(float(np.degrees(2 * np.arctan(0.5 / Kn[1, 1]))), 2),
            "depth_m": {"min": round(float(d.min()), 4), "median": round(float(np.median(d)), 4),
                        "max": round(float(d.max()), 4)} if d.size else None,
            "valid_fraction": round(float(mask.mean()), 4)}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
