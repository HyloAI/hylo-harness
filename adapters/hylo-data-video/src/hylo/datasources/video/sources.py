"""Video data sources (RFC-0004 W1): human or robot videos without action labels.

FrameFolderSource: one directory per episode with numbered image files and an optional episode.json
    {"instruction": "...", "task_id": "...", "success": true, "camera": "agentview"}.
VideoFileSource: one video file per episode (mp4 needs the `ffmpeg` extra), with an optional sidecar <name>.json
    of the same shape.

Frames carry images only: no state, no actions. Such episodes can train what needs no actions (success judges,
progress models, video world models) and can be labelled by a calibrated judge (`core.instrument.label_run`);
action labels would need a labeler (latent actions, hand pose), which is not part of this package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from hylo.core.datasource import ExternalEpisode, ExternalFrame

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".gif")


def _sidecar(path: Path) -> dict:
    for p in (path / "episode.json", path.with_suffix(".json")):
        if p.is_file():
            return json.loads(p.read_text())
    return {}


def _resize(img: np.ndarray, size: int | None) -> np.ndarray:
    if not size:
        return img
    from PIL import Image

    return np.asarray(Image.fromarray(img).convert("RGB").resize((size, size)))


class _Base:
    def __init__(self, root: str, task_prefix: str = "video/", camera: str = "agentview", every: int = 1,
                 size: int | None = 256, max_episodes: int | None = None, licence: str = "unknown") -> None:
        self.root, self.task_prefix, self.camera = Path(root), task_prefix, camera
        self.every, self.size, self.max_episodes, self.licence = max(1, every), size, max_episodes, licence
        self.origin = f"{type(self).__name__}:{self.root}"
        self.name = f"{type(self).__name__}[{self.root.name}]"

    def _episode(self, eid: str, meta: dict, images: list[np.ndarray]) -> ExternalEpisode:
        instruction = str(meta.get("instruction", ""))
        task = meta.get("task_id") or "_".join(instruction.lower().split())[:120] or eid
        cam = meta.get("camera", self.camera)
        frames = [ExternalFrame(images={cam: _resize(im, self.size)}) for im in images]
        return ExternalEpisode(episode_id=eid, task_id=f"{self.task_prefix}{task}", instruction=instruction, frames=frames,
                               success=meta.get("success"), meta={k: v for k, v in meta.items() if k not in ("instruction", "success")})


class FrameFolderSource(_Base):
    def episodes(self) -> Iterator[ExternalEpisode]:
        from PIL import Image

        dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        for n, d in enumerate(dirs):
            if self.max_episodes is not None and n >= self.max_episodes:
                return
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)[:: self.every]
            if files:
                yield self._episode(d.name, _sidecar(d), [np.asarray(Image.open(f).convert("RGB")) for f in files])


class VideoFileSource(_Base):
    def episodes(self) -> Iterator[ExternalEpisode]:
        import imageio.v3 as iio

        files = sorted(f for f in self.root.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES)
        for n, f in enumerate(files):
            if self.max_episodes is not None and n >= self.max_episodes:
                return
            images = [np.asarray(fr)[..., :3] for i, fr in enumerate(iio.imiter(f)) if i % self.every == 0]
            if images:
                yield self._episode(f.stem, _sidecar(f), images)
