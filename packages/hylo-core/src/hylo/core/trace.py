"""Event-sourced, replayable episode trace.

Layout of a run directory:
    <run>/manifest.json                    RunManifest (config hash, versions, seeds)
    <run>/episodes/<episode_id>.jsonl      one Event per line, arrays inline when small
    <run>/blobs/<sha256>.npy               large arrays (images, point clouds), content-addressed
    <run>/evallog.json                     aggregated scores; points back at episode files
    <run>/seal.json                        optional: the run's file inventory, written by `hylo seal`

Hash chain (hylo.trace/v0.2): every event line carries `prev` (the previous line's hash, "" for the first) and `h`,
the sha256 of the line's canonical JSON without `h`. Blobs are named by their own digest, so the chain covers them
too. `hylo.core.seal.verify_chain` finds the first line that was changed, removed or reordered. Lines written before
the chain existed have neither key and are reported as unchained, not as broken.

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
        self._heads: dict[str, str] = {}

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
            path = self.root / "episodes" / f"{event.episode_id}.jsonl"
            self._heads[event.episode_id] = last_hash(path)
            f = open(path, "a", encoding="utf-8")
            self._files[event.episode_id] = f
        record = {"type": event.type, "seq": event.seq, "source": event.source,
                  "sim_time": event.sim_time, "wall_time": event.wall_time,
                  "payload": codec.encode(self._offload(dict(event.payload))),
                  "prev": self._heads.get(event.episode_id, "")}
        record["h"] = line_hash(record)
        self._heads[event.episode_id] = record["h"]
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


def line_hash(record: dict[str, Any]) -> str:
    """sha256 of an event line's canonical JSON, without its own `h`."""
    body = {k: v for k, v in record.items() if k != "h"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def last_hash(path: Path) -> str:
    """The `h` of the last line of an episode file ("" when there is none: a new file or an unchained one)."""
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - (1 << 20)))                  # the last line is within the last MiB
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    try:
        return str(json.loads(lines[-1]).get("h", "")) if lines else ""
    except json.JSONDecodeError:
        return ""


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
