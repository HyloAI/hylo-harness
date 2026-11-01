"""ActionBank: an in-memory, numpy-backed ActionMemory (RFC-0003).

Keys are L2-normalised vectors stored as float16 (cosine similarity = dot product); chunks keep their own length.
Capacity is enforced FIFO (Retrieve-then-Steer: a 5k FIFO bank matches an unlimited one). A saved bank is
`bank.npz` + `bank.json` (no pickle) and is frozen when loaded for evaluation: `add` raises, and `snapshot()` is the
content digest that goes into the agent spec.

    memory: {type: action_bank, params: {path: "artifact:l10-v1@<digest>"}}   # a store id (core.store) or a path
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from hylo.core.contract import current_schema
from hylo.core.experience import ActionSegment, EpisodeRef


class FrozenMemoryError(RuntimeError):
    pass


class ActionBank:
    def __init__(self, path: str | None = None, capacity: int = 50_000, frozen: bool | None = None) -> None:
        self.capacity = capacity
        self._keys: list[np.ndarray] = []
        self._segs: list[ActionSegment] = []
        self._matrix: np.ndarray | None = None
        if path is not None:
            self._load(Path(path))
        self.frozen = (path is not None) if frozen is None else frozen

    # -- write --------------------------------------------------------------------------------------
    def add(self, segments: Sequence[ActionSegment]) -> int:
        if self.frozen:
            raise FrozenMemoryError("this bank is a frozen snapshot; build a new one to change it")
        for s in segments:
            k = np.asarray(s.key, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(k))
            if not np.isfinite(n) or n == 0:
                continue
            if self._keys and k.shape != self._keys[0].shape:
                raise ValueError(f"key dim {k.shape} != bank key dim {self._keys[0].shape}")
            self._keys.append((k / n).astype(np.float16))
            self._segs.append(s.model_copy(update={"key": self._keys[-1]}))
        overflow = len(self._segs) - self.capacity
        if overflow > 0:                                  # FIFO eviction
            del self._keys[:overflow], self._segs[:overflow]
        self._matrix = None
        return len(segments)

    # -- read ---------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._segs)

    def query(self, key: np.ndarray, *, task_id: str | None = None, k: int = 10) -> list[tuple[ActionSegment, float]]:
        if not self._segs:
            return []
        if self._matrix is None:
            self._matrix = np.stack(self._keys).astype(np.float32)
            self._tasks = np.array([s.task_id for s in self._segs])
        q = np.asarray(key, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self._matrix @ q
        if task_id is not None:
            sims = np.where(self._tasks == task_id, sims, -np.inf)
        k = min(k, int(np.isfinite(sims).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._segs[i], float(sims[i])) for i in top]

    # -- persistence ----------------------------------------------------------------------------------
    def snapshot(self) -> str:
        h = hashlib.sha256()
        for s in self._segs:
            h.update(np.ascontiguousarray(s.key).tobytes())
            h.update(np.ascontiguousarray(np.asarray(s.actions, dtype=np.float32)).tobytes())
            h.update(f"{s.task_id}|{s.step}|{s.source.episode_id}".encode())
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self._segs:
            raise ValueError("refusing to save an empty bank")
        T = max(len(s.actions) for s in self._segs)
        A = np.asarray(self._segs[0].actions).shape[1]
        chunks = np.zeros((len(self._segs), T, A), dtype=np.float32)
        lengths = np.zeros(len(self._segs), dtype=np.int32)
        for i, s in enumerate(self._segs):
            a = np.asarray(s.actions, dtype=np.float32)
            chunks[i, : len(a)], lengths[i] = a, len(a)
        np.savez(path / "bank.npz", keys=np.stack(self._keys), chunks=chunks, lengths=lengths)
        digest = self.snapshot()
        meta = [{"task_id": s.task_id, "step": s.step, "progress": s.progress, "source": s.source.model_dump(mode="json")}
                for s in self._segs]
        (path / "bank.json").write_text(json.dumps({"schema": "hylo.action_bank/v0.1", "digest": digest, "n": len(self._segs),
                                                    "key_dim": int(self._keys[0].shape[0]), "segments": meta}))
        return digest

    def _load(self, path: Path) -> None:
        meta = json.loads((path / "bank.json").read_text())
        if current_schema(meta.get("schema")) != "hylo.action_bank/v0.1":
            raise ValueError(f"not a hylo.action_bank/v0.1 directory: {path}")
        with np.load(path / "bank.npz", allow_pickle=False) as z:
            keys, chunks, lengths = z["keys"], z["chunks"], z["lengths"]
        for k, c, n, m in zip(keys, chunks, lengths, meta["segments"]):
            self._keys.append(k)
            self._segs.append(ActionSegment(key=k, actions=c[:n], task_id=m["task_id"], step=m["step"],
                                            progress=m["progress"], source=EpisodeRef.model_validate(m["source"])))
        if self.snapshot() != meta["digest"]:
            raise ValueError(f"bank content does not match its recorded digest ({path})")
        self.digest = meta["digest"]
