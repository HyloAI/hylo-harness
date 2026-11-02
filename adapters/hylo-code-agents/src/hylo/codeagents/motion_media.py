# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pillow>=10", "imageio>=2.30", "imageio-ffmpeg>=0.5"]
# ///
"""The motion studio's media worker: video windows, labelled video grids, and rigged characters in Blender.

Runs as an isolated script (`uv run --script motion_media.py request.json`, the same request / result.json protocol
as motion_sim); tests import it by path and call `handle`. No Hylo imports.

clip: a video window (or one image) -> frames.npy (T, H, W, 3) uint8 for the human tool.
compose: videos (MP4, or .npy frames) side by side or in a grid of `cols` columns, each with a label.
rig: a rigged character driven by keypoints (motion_blender.py inside Blender): solve only (metrics, posed joints)
  or render too (MP4 + still). The hylo-tools render tool is reused: its worker finds Blender ($HYLO_BLENDER,
  $HYLO_BLENDER_LIBS) and writes the video, its Blender script builds the look (render settings, sky, lights,
  ground); `render_tool` in params is the directory of those worker scripts.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BLENDER_SCRIPT = Path(__file__).with_name("motion_blender.py")


# ------------------------------------------------------------------------------------------------- clip
def clip(path: str, out: Path, start: float = 0.0, seconds: float | None = None, max_side: int = 640) -> dict:
    """A video (or one image, repeated 3 frames) -> frames.npy (T, H, W, 3) uint8."""
    import imageio.v2 as imageio
    from PIL import Image
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        frames, fps = [np.asarray(Image.open(p).convert("RGB"))] * 3, 30.0
    else:
        r = imageio.get_reader(str(p))
        fps = float(r.get_meta_data().get("fps", 30) or 30)
        a = int(round(start * fps))
        b = None if seconds is None else a + int(round(seconds * fps))
        frames = []
        for i, f in enumerate(r):
            if i >= a:
                frames.append(f[..., :3])
            if b is not None and i + 1 >= b:
                break
        r.close()
    if not frames:
        raise ValueError(f"no frames in {path} from {start} s")
    h, w = frames[0].shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1:
        size = (int(w * k) // 2 * 2, int(h * k) // 2 * 2)
        frames = [np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in frames]
    np.save(out / "frames.npy", np.stack(frames).astype(np.uint8))
    return {"outputs": {"frames": len(frames), "fps": fps, "size": list(frames[0].shape[:2])},
            "files": {"frames": str(out / "frames.npy")}}


# ------------------------------------------------------------------------------------------------- compose
def label(im, text: str, xy=(10, 8)) -> None:
    """Draw `text` on a white box at xy (PIL image, in place)."""
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(12, im.height // 22))
    except TypeError:                                 # Pillow < 10.1
        font = ImageFont.load_default()
    box = dr.textbbox(xy, text, font=font)
    dr.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255))
    dr.text(xy, text, fill=(20, 20, 20), font=font)


def _frames(v: str):
    from PIL import Image
    if str(v).endswith(".npy"):
        return [Image.fromarray(f) for f in np.load(v)], None
    import imageio.v2 as imageio
    r = imageio.get_reader(v)
    fps = float(r.get_meta_data().get("fps", 30))
    frames = [Image.fromarray(f) for f in r]
    r.close()
    return frames, fps


def compose(videos: list[str], labels: list[str], out: Path, fps: float | None = None, height: int = 360,
            cols: int | None = None) -> dict:
    """Videos side by side (or in a grid of `cols` columns) at one height, each labelled, as long as the longest
    (shorter ones hold their last frame)."""
    import imageio.v2 as imageio
    from PIL import Image
    clips, rate = [], fps
    for v in videos:
        fr, r = _frames(v)
        clips.append(fr)
        rate = rate or r
    rate = rate or 30.0
    n = max(len(c) for c in clips)
    cols = int(cols or len(clips))
    width = max(int(round(c[0].width * height / c[0].height)) // 2 * 2 for c in clips)
    with imageio.get_writer(str(out), fps=rate, codec="libx264", quality=7, pixelformat="yuv420p",
                            macro_block_size=8, ffmpeg_log_level="error") as wr:
        for i in range(n):
            tiles = []
            for c, lab in zip(clips, labels):
                im = c[min(i, len(c) - 1)].convert("RGB")
                im = im.resize((int(round(im.width * height / im.height)) // 2 * 2, height))
                canvas = Image.new("RGB", (width, height), (255, 255, 255))
                canvas.paste(im, ((width - im.width) // 2, 0))
                label(canvas, lab)
                tiles.append(np.asarray(canvas))
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            rows = [np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)]
            wr.append_data(np.concatenate(rows, 0))
    return {"outputs": {"frames": n, "fps": rate, "tiles": len(clips), "cols": cols}, "files": {"video": str(out)}}


# ------------------------------------------------------------------------------------------------- rig
def render_worker(directory: str):
    """The hylo-tools render worker module (its Blender locator and video writer)."""
    d = Path(directory)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("hylo_render_worker", d / "render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rig(params: dict, out: Path) -> dict:
    """Drive (and with render=True render) one rigged character in Blender; see motion_blender.py for the
    parameters. Returns the rig metrics, the posed joints (.npy) and, when rendered, video.mp4 and still.png."""
    rw = render_worker(params["render_tool"])
    exe, env = rw.blender_exe()
    if exe is None:
        raise RuntimeError("Blender not found (set HYLO_BLENDER to the blender executable)")
    frames = out / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    cfg = dict(params, out_dir=str(out), frames_dir=str(frames),
               render_blender=str(Path(params["render_tool"]) / "render_blender.py"))
    (out / "rig.json").write_text(json.dumps(cfg))
    threads = int(params.get("threads", 32))
    proc = subprocess.run([str(exe), "-b", "--factory-startup", "-noaudio", "-t", str(threads), "--python",
                           str(BLENDER_SCRIPT), "--", str(out / "rig.json")], env=env, capture_output=True, text=True,
                          timeout=float(params.get("timeout_s", 4 * 3600)))
    (out / "blender.log").write_text(proc.stdout + proc.stderr)
    res_path = out / "rig_result.json"
    if proc.returncode != 0 or not res_path.exists():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-20:])
        raise RuntimeError(f"Blender failed (exit {proc.returncode}):\n{tail}")
    res = json.loads(res_path.read_text())
    files = dict(res.get("files", {}))
    if params.get("render", True):
        pngs = sorted(frames.glob("*.png"))
        if not pngs:
            raise RuntimeError("Blender rendered no frames")
        n = rw.write_video(frames, out / "video.mp4", float(params.get("fps", 30)))
        shutil.copy(pngs[len(pngs) // 2], out / "still.png")
        files.update(video=str(out / "video.mp4"), still=str(out / "still.png"))
        res["outputs"]["video_frames"] = n
    return {"outputs": res["outputs"], "files": files}


# ------------------------------------------------------------------------------------------------- entry
def handle(op: str, params: dict, inputs: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if op == "clip":
        res = clip(inputs["media"], out, float(params.get("start", 0.0)), params.get("seconds"))
    elif op == "compose":
        res = compose(list(params["videos"]), list(params["labels"]), out / params.get("name", "split.mp4"),
                      fps=params.get("fps"), height=int(params.get("height", 360)), cols=params.get("cols"))
    elif op == "rig":
        res = rig(params, out)
    else:
        raise ValueError(f"unknown op {op!r} (clip, compose, rig)")
    res["timing"] = {"seconds": round(time.perf_counter() - t0, 3)}
    res["ok"] = True
    return res


def main(argv: list[str]) -> int:
    req = json.loads(Path(argv[0]).read_text())
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        res = handle(req["op"], req.get("params", {}), req.get("inputs", {}), out)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        import traceback
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
    (out / "result.json").write_text(json.dumps(res, default=float))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
