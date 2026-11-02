# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow>=15", "huggingface_hub>=0.24", "numpy>=1.26", "pillow>=10", "imageio-ffmpeg>=0.5"]
# ///
"""Fetch one episode of a LeRobot-format dataset from Hugging Face into a replay recording (run in its own uv
environment: `uv run --script replay_fetch.py ...`, so pyarrow and huggingface_hub never enter the Hylo venv).

    uv run --script replay_fetch.py REPO_ID EPISODE --camera KEY --out DIR [--state KEY] [--action KEY]
                                    [--fps F] [--max-frames N] [--rotate180] [--revision REV]

Reads meta/info.json (codebase v2.x or v3.0), finds the episode's rows in the data parquet files, decodes the camera
(images stored in the parquet, or the episode's time range of the mp4 with ffmpeg) and writes
DIR/recording.npz (rgb (T, H, W, 3) uint8, time (T,), state (T, S), action (T, A) when present) and
DIR/source.json (repo, revision, licence from the dataset card, task text, fps, keys). Only what the episode needs is
downloaded (one parquet file and one video file at most).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def _download(repo: str, path: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def _card_licence(repo: str, revision: str | None) -> str:
    try:
        text = Path(_download(repo, "README.md", revision)).read_text(errors="ignore")
    except Exception as exc:  # noqa: BLE001 - a card is optional
        return f"unknown (no dataset card: {exc})"
    m = re.search(r"^license:\s*(\S+)", text, re.M)
    return m.group(1) if m else "unknown (card has no license field)"


def _episode_rows_v3(repo: str, info: dict, ep: int, revision: str | None) -> tuple[dict, str]:
    import pyarrow.parquet as pq

    for chunk in range(0, 100):
        try:
            f = _download(repo, f"meta/episodes/chunk-{chunk:03d}/file-000.parquet", revision)
        except Exception:  # noqa: BLE001
            break
        # episodes may span several files per chunk
        files = [f]
        k = 1
        while True:
            try:
                files.append(_download(repo, f"meta/episodes/chunk-{chunk:03d}/file-{k:03d}.parquet", revision))
                k += 1
            except Exception:  # noqa: BLE001
                break
        for path in files:
            t = pq.read_table(path)
            cols = [c for c in t.schema.names if not c.startswith("stats/")]
            rows = t.select(cols).to_pylist()
            for r in rows:
                if int(r["episode_index"]) == ep:
                    task = (r.get("tasks") or [""])[0]
                    return r, str(task)
    raise SystemExit(f"episode {ep} not found in {repo}")


def _frames_from_video(video: str, start: float, end: float, fps: float) -> np.ndarray:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([exe, "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe)
    if not m:
        raise SystemExit(f"cannot read the video size of {video}")
    w, h = int(m.group(1)), int(m.group(2))
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}", "-i", video, "-t", f"{end - start:.6f}",
           "-an", "-vf", f"fps={fps:g}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def _decode_image(cell) -> np.ndarray:
    from PIL import Image

    if isinstance(cell, dict):
        cell = cell.get("bytes")
    return np.asarray(Image.open(io.BytesIO(cell)).convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("episode", type=int)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--state", default="observation.state")
    ap.add_argument("--action", default="action")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=None, help="output frame rate (default: the dataset's)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    info = json.loads(Path(_download(a.repo, "meta/info.json", a.revision)).read_text())
    version = str(info.get("codebase_version", "v2.0"))
    src_fps = float(info["fps"])
    feat = info["features"][a.camera]
    if version.startswith("v3"):
        row, task = _episode_rows_v3(a.repo, info, a.episode, a.revision)
        data_path = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
    else:
        chunk = a.episode // int(info.get("chunks_size", 1000))
        data_path = info["data_path"].format(episode_chunk=chunk, episode_index=a.episode)
        row, task = {}, ""
        try:
            tasks = [json.loads(line) for line in Path(_download(a.repo, "meta/episodes.jsonl", a.revision)).read_text().splitlines()
                     if line.strip()]
            task = next((str((t.get("tasks") or [""])[0]) for t in tasks if int(t["episode_index"]) == a.episode), "")
        except Exception:  # noqa: BLE001
            pass
    table = pq.read_table(_download(a.repo, data_path, a.revision))
    ep_col = np.asarray(table.column("episode_index").to_numpy()).reshape(-1)
    sel = np.flatnonzero(ep_col == a.episode)
    if not len(sel):
        raise SystemExit(f"no rows of episode {a.episode} in {data_path}")
    sub = table.take(sel)
    names = sub.schema.names
    time = np.asarray(sub.column("timestamp").to_numpy(), float).reshape(-1) if "timestamp" in names else \
        np.arange(len(sel)) / src_fps
    arrays: dict[str, np.ndarray] = {}
    for key, name in ((a.state, "state"), (a.action, "action")):
        if key in names:
            arrays[name] = np.asarray(sub.column(key).to_pylist(), np.float32)
    fps = a.fps or src_fps
    step = max(1, int(round(src_fps / fps)))
    idx = np.arange(0, len(sel), step)
    if feat.get("dtype") == "video":
        vkey = f"videos/{a.camera}"
        vpath = info["video_path"].format(video_key=a.camera, chunk_index=row.get(f"{vkey}/chunk_index", 0),
                                          file_index=row.get(f"{vkey}/file_index", 0),
                                          episode_chunk=a.episode // int(info.get("chunks_size", 1000)),
                                          episode_index=a.episode)
        start = float(row.get(f"{vkey}/from_timestamp", 0.0))
        end = float(row.get(f"{vkey}/to_timestamp", start + len(sel) / src_fps))
        rgb = _frames_from_video(_download(a.repo, vpath, a.revision), start, end, src_fps)
        n = min(len(rgb), len(sel))
        idx = idx[idx < n]
        rgb = rgb[idx]
    else:
        col = sub.column(a.camera).to_pylist()
        rgb = np.stack([_decode_image(col[i]) for i in idx])
    if a.max_frames:
        idx, rgb = idx[: a.max_frames], rgb[: a.max_frames]
    if a.rotate180:
        rgb = rgb[:, ::-1, ::-1].copy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "recording.npz", rgb=rgb, time=time[idx] - time[idx][0],
                        **{k: v[idx] for k, v in arrays.items()})
    meta = {"repo": a.repo, "revision": a.revision or "main", "episode": a.episode, "camera": a.camera,
            "licence": _card_licence(a.repo, a.revision), "task": task, "fps": src_fps / step, "source_fps": src_fps,
            "frames": int(len(idx)), "codebase_version": version, "state_key": a.state, "action_key": a.action,
            "rotate180": bool(a.rotate180), "image_shape": list(rgb.shape[1:])}
    (out / "source.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
