"""SQLite episodic/spatial memory store (reference backend for the core MemoryStore protocol)."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from worldcurrent.core.memory import Hit, MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan, WriteResult, now



_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, key TEXT, content TEXT NOT NULL,
  payload TEXT, entities TEXT, scope TEXT,
  frame_id TEXT, px REAL, py REAL, pz REAL,
  t_start REAL, t_end REAL,
  valid_from REAL, valid_to REAL, recorded_at REAL NOT NULL, expired_at REAL,
  confidence REAL, source TEXT, content_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_kind_key ON items(kind, key);
CREATE INDEX IF NOT EXISTS idx_pos ON items(frame_id, px, py);
CREATE INDEX IF NOT EXISTS idx_hash ON items(content_hash);
"""


class SQLiteMemoryStore:
    """Episodic + spatial store. Spatial queries use a bounding-box index then an exact radius check;
    a production spatial backend (Spark-DSG) would replace this behind the same interface."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)

    def write(self, items: Sequence[MemoryItem]) -> WriteResult:
        events: list[tuple[str, str]] = []
        t = now()
        with self.db:
            for it in items:
                h = it.content_hash()
                if self.db.execute("SELECT 1 FROM items WHERE content_hash=? AND valid_to IS NULL AND expired_at IS NULL",
                                   (h,)).fetchone():
                    events.append(("SKIP_DUP", it.id))
                    continue
                valid_from = it.valid_from if it.valid_from is not None else t
                if it.key is not None:
                    prev = self.db.execute(
                        "SELECT id FROM items WHERE kind=? AND key=? AND valid_to IS NULL AND expired_at IS NULL",
                        (it.kind.value, it.key)).fetchall()
                    for (pid,) in prev:
                        self.db.execute("UPDATE items SET valid_to=? WHERE id=?", (valid_from, pid))
                        events.append(("INVALIDATE", pid))
                pos = it.pose.position if it.pose else (None, None, None)
                self.db.execute(
                    "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it.id, it.kind.value, it.key, it.content, json.dumps(it.model_dump(mode="json")["payload"]),
                     json.dumps(list(it.entities)), json.dumps(dict(it.scope)),
                     it.pose.frame_id if it.pose else None, *pos,
                     it.event_time.start if it.event_time else None,
                     it.event_time.end if it.event_time else None,
                     valid_from, None, t, None, it.confidence, it.source, h))
                events.append(("ADD", it.id))
        return WriteResult(events=tuple(events))

    def _where(self, q: MemoryQuery) -> tuple[str, list[Any]]:
        clauses, args = ["expired_at IS NULL"], []
        if q.kinds:
            clauses.append(f"kind IN ({','.join('?' * len(q.kinds))})")
            args += [k.value for k in q.kinds]
        if q.key is not None:
            clauses.append("key = ?")
            args.append(q.key)
        if q.text:
            clauses.append("content LIKE ?")
            args.append(f"%{q.text}%")
        if q.as_of is not None:
            clauses.append("valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)")
            args += [q.as_of, q.as_of]
        elif not q.include_invalidated:
            clauses.append("valid_to IS NULL")
        if q.time_range is not None:
            lo, hi = q.time_range
            clauses.append("t_start IS NOT NULL AND t_start <= ? AND COALESCE(t_end, t_start) >= ?")
            args += [hi, lo]
        if q.near is not None:
            pose, r = q.near
            clauses.append("frame_id = ? AND px BETWEEN ? AND ? AND py BETWEEN ? AND ?")
            args += [pose.frame_id, pose.position[0] - r, pose.position[0] + r,
                     pose.position[1] - r, pose.position[1] + r]
        return " AND ".join(clauses), args

    def query(self, q: MemoryQuery) -> list[Hit]:
        where, args = self._where(q)
        rows = self.db.execute(f"SELECT * FROM items WHERE {where} ORDER BY recorded_at DESC", args).fetchall()
        cols = [c[0] for c in self.db.execute("SELECT * FROM items LIMIT 0").description]
        hits: list[Hit] = []
        for row in rows:
            r = dict(zip(cols, row))
            item = self._to_item(r)
            if q.entities and not set(q.entities) & set(item.entities):
                continue
            if q.scope and any(item.scope.get(k) != v for k, v in q.scope.items()):
                continue
            details = {"confidence": item.confidence}
            if q.near is not None:
                d = math.dist(q.near[0].position, item.pose.position)  # type: ignore[union-attr]
                if d > q.near[1]:
                    continue
                details["spatial"] = 1.0 - d / q.near[1] if q.near[1] > 0 else 1.0
            score = sum(details.values()) / len(details)
            hits.append(Hit(item=item, score=score, details=details, valid_to=r["valid_to"]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: q.k]

    def forget(self, q: MemoryQuery) -> int:
        where, args = self._where(q)
        with self.db:
            cur = self.db.execute(f"UPDATE items SET expired_at=? WHERE {where}", [now(), *args])
        return cur.rowcount

    def snapshot(self) -> str:
        rows = self.db.execute("SELECT content_hash, valid_to IS NULL FROM items "
                               "WHERE expired_at IS NULL ORDER BY content_hash").fetchall()
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]

    @staticmethod
    def _to_item(r: dict[str, Any]) -> MemoryItem:
        pose = Pose(frame_id=r["frame_id"], position=(r["px"], r["py"], r["pz"])) if r["frame_id"] else None
        span = TimeSpan(start=r["t_start"], end=r["t_end"]) if r["t_start"] is not None else None
        return MemoryItem(kind=MemoryKind(r["kind"]), content=r["content"], key=r["key"],
                          payload=json.loads(r["payload"] or "{}"), entities=tuple(json.loads(r["entities"])),
                          scope=json.loads(r["scope"]), pose=pose, event_time=span,
                          valid_from=r["valid_from"], confidence=r["confidence"], source=r["source"], id=r["id"])
