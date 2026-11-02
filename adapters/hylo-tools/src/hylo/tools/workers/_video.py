# hylo: worker-side (runs in a tool worker's own environment; never imported by Hylo packages)
"""Video helpers for the hands and human workers: read frames, write an MP4, draw keypoint skeletons, fit a metric
3D keypoint set into the camera frame from its 2D projection, fetch a model file once into the tools cache."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
# MediaPipe Pose (33 landmarks)
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
              (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), (15, 17), (15, 19),
              (15, 21), (16, 18), (16, 20), (16, 22), (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
              (9, 10)]


def read_video(path: str, stride: int = 1, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Frames (T, H, W, 3) uint8 and fps from a video file, a folder of images or a .npy/.npz (T, H, W, 3) array
    (fps: the request's or 30)."""
    p = Path(path)
    if p.suffix in (".npy", ".npz"):
        if p.suffix == ".npy":
            frames = np.load(p)
        else:
            with np.load(p) as z:
                frames = z[z.files[0]]
        fps = 30.0
    elif p.is_dir():
        from PIL import Image
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
        fps = 30.0
    else:
        import imageio.v3 as iio
        meta = iio.immeta(p, plugin="pyav") if _has_av() else iio.immeta(p)
        fps = float(meta.get("fps", 30.0) or 30.0)
        frames = iio.imread(p, plugin="pyav") if _has_av() else iio.imread(p)
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    frames = frames[..., :3].astype(np.uint8)[::max(1, stride)]
    if max_frames:
        frames = frames[:max_frames]
    return frames, fps / max(1, stride)


def _has_av() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def write_mp4(path: Path, frames: list[np.ndarray] | np.ndarray, fps: float, crf: int = 28) -> None:
    """H.264 MP4 (yuv420p, even dimensions) through imageio-ffmpeg's bundled ffmpeg."""
    import imageio.v2 as imageio
    frames = [np.asarray(f) for f in frames]
    H, W = frames[0].shape[:2]
    H2, W2 = H - H % 2, W - W % 2
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                            output_params=["-crf", str(crf), "-preset", "slow"], macro_block_size=1) as w:
        for f in frames:
            w.append_data(f[:H2, :W2])


def draw_skeleton(img: Any, kp2d: np.ndarray, edges: list[tuple[int, int]], color: tuple[int, int, int],
                  valid: np.ndarray | None = None, radius: int = 3, width: int = 2) -> Any:
    """Draw keypoints (N, 2) pixels and their edges on a PIL image (in place)."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    ok = np.ones(len(kp2d), bool) if valid is None else np.asarray(valid, bool)
    for a, b in edges:
        if a < len(kp2d) and b < len(kp2d) and ok[a] and ok[b]:
            d.line([tuple(kp2d[a]), tuple(kp2d[b])], fill=color, width=width)
    for i, (x, y) in enumerate(kp2d):
        if ok[i]:
            d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 255, 255), outline=color)
    return img


def default_K(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics with the given horizontal field of view, principal point at the centre."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def fit_translation(points: np.ndarray, uv: np.ndarray, K: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The translation t such that points + t (metric, camera-aligned axes) project onto uv with intrinsics K:
    linear least squares on (u - cx)(Z + tz) = fx (X + tx) and (v - cy)(Z + tz) = fy (Y + ty)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    n = len(points)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], b[:n] = fx, -du, du * Z - fx * X
    A[n:, 1], A[n:, 2], b[n:] = fy, -dv, dv * Z - fy * Y
    W = np.concatenate([w, w])
    t, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
    return t


def fetch(url: str, sub: str) -> tuple[Path, str]:
    """Download a model file once into $HYLO_TOOLS_CACHE/models/<sub>/; returns its path and sha256."""
    root = Path(os.environ.get("HYLO_TOOLS_CACHE") or Path.home() / ".cache" / "hylo" / "tools")
    dst = root / "models" / sub / url.rsplit("/", 1)[-1]
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst, hashlib.sha256(dst.read_bytes()).hexdigest()


def panel_3d(skeletons: list[tuple[np.ndarray, tuple[int, int, int]]], edges: list[tuple[int, int]], size: int,
             yaw_deg: float = 45.0, pitch_deg: float = 20.0, extent: float | None = None, label: str = "",
             center: np.ndarray | None = None, base: Any = None, radius: int = 2) -> Any:
    """A small orthographic view of 3D skeletons (camera axes: x right, y down, z forward), rotated by yaw about
    the vertical axis and pitch about x, centred on `center` (default: the joint mean) and scaled to fit (or to
    `extent` metres across). `base`: an earlier panel to draw onto (same view parameters)."""
    from PIL import Image, ImageDraw
    img = base if base is not None else Image.new("RGB", (size, size), (24, 26, 32))
    pts = [p for p, _ in skeletons if p is not None and len(p)]
    if label:
        ImageDraw.Draw(img).text((6, 4), label, fill=(200, 200, 210))
    if not pts:
        return img
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R = Rx @ Ry
    allp = np.concatenate(pts)
    c = allp.mean(0) if center is None else np.asarray(center, float)
    span = extent or max(1e-6, float(np.abs((allp - c) @ R.T)[:, :2].max()) * 2.2)
    s = size / span
    for p, color in skeletons:
        if p is None or not len(p):
            continue
        uv = ((p - c) @ R.T)[:, :2] * s + size / 2
        draw_skeleton(img, uv, edges, color, radius=radius, width=2)
    return img
