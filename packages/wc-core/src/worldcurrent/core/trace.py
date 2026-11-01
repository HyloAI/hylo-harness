"""Event-sourced, replayable episode trace.

Layout of a run directory:
    <run>/manifest.json                    RunManifest (config hash, versions, seeds)
    <run>/episodes/<episode_id>.jsonl      one Event per line, arrays inline when small
    <run>/blobs/<sha256>.npy               large arrays (images, point clouds), content-addressed
    <run>/evallog.json                     aggregated scores; points back at episode files

Scorers and shadow replay read only this; they never call the Agent or the Environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import codec
from .events import Event

BLOB_THRESHOLD_BYTES = 64 * 1024


class TraceStore:
    def __init__(self, run_dir: str | os.PathLike) -> None:
        self.root = Path(run_dir)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        self._files: dict[str, io.TextIOWrapper] = {}

    # ---- writing -------------------------------------------------------------------------
    def _offload(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray) and obj.nbytes > BLOB_THRESHOLD_BYTES:
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            data = buf.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            path = self.root / "blobs" / f"{digest}.npy"
            if not path.exists():
                path.write_bytes(data)
            return {"__blob__": f"{digest}.npy"}
        if isinstance(obj, dict):
            return {k: self._offload(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._offload(v) for v in obj]
        return obj

    def write(self, event: Event) -> None:
        f = self._files.get(event.episode_id)
        if f is None:
            f = open(self.root / "episodes" / f"{event.episode_id}.jsonl", "a", encoding="utf-8")
            self._files[event.episode_id] = f
        record = {"type": event.type, "seq": event.seq, "source": event.source,
                  "sim_time": event.sim_time, "wall_time": event.wall_time,
                  "payload": codec.encode(self._offload(dict(event.payload)))}
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close_episode(self, episode_id: str) -> None:
        f = self._files.pop(episode_id, None)
        if f:
            f.flush()
            os.fsync(f.fileno())
            f.close()

    def close(self) -> None:
        for eid in list(self._files):
            self.close_episode(eid)

    # ---- reading -------------------------------------------------------------------------
    def episode_ids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "episodes").glob("*.jsonl"))

    def read(self, episode_id: str) -> "EpisodeTrace":
        return EpisodeTrace(self.root, episode_id)


class EpisodeTrace:
    def __init__(self, root: Path, episode_id: str) -> None:
        self.root, self.episode_id = root, episode_id
        self._events: list[dict[str, Any]] | None = None

    def _resolve(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__blob__" in obj and len(obj) == 1:
                return np.load(self.root / "blobs" / obj["__blob__"], allow_pickle=False)
            return {k: self._resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve(v) for v in obj]
        return obj

    @property
    def events(self) -> list[dict[str, Any]]:
        if self._events is None:
            path = self.root / "episodes" / f"{self.episode_id}.jsonl"
            with open(path, encoding="utf-8") as f:
                self._events = [json.loads(line) for line in f if line.strip()]
            for e in self._events:
                e["payload"] = self._resolve(codec.decode(e["payload"]))
        return self._events

    def of_type(self, *types: str) -> Iterator[dict[str, Any]]:
        return (e for e in self.events if e["type"] in types)

    def first(self, type_: str) -> dict[str, Any] | None:
        return next(self.of_type(type_), None)

    def last(self, type_: str) -> dict[str, Any] | None:
        found = None
        for e in self.of_type(type_):
            found = e
        return found
