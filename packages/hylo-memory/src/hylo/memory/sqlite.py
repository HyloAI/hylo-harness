"""SQLite episodic/spatial backend with scoped identities and lossless typed records."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Sequence

from hylo.core.memory import Hit, MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan, WriteResult, now

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


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SQLiteMemoryStore:
    """Single-owner store. Facts replace only the same (kind, key, scope, clock).

    Event time zero is valid. Backdated replacements are rejected atomically; callers should
    retain late evidence as unkeyed events instead of rewriting a fact's historical interval.
    ``forget`` expires ALL exact matches (``k`` only limits retrieval), never deletes evidence.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(_SCHEMA)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(items)")}
        # Additive legacy migration: unknown orientations cannot be recovered. Old event clocks
        # were reconstructed as sim, so preserve that interpretation and rehash once.
        with self.db:
            migrated = "orientation" not in columns or "t_clock" not in columns
            if "orientation" not in columns:
                self.db.execute("ALTER TABLE items ADD COLUMN orientation TEXT")
            if "t_clock" not in columns:
                self.db.execute("ALTER TABLE items ADD COLUMN t_clock TEXT")
                self.db.execute("UPDATE items SET t_clock=CASE WHEN t_start IS NULL THEN 'wall' ELSE 'sim' END")
            if migrated:
                names = [c[0] for c in self.db.execute("SELECT * FROM items LIMIT 0").description]
                for row in self.db.execute("SELECT * FROM items").fetchall():
                    it = self._to_item(dict(zip(names, row)))
                    self.db.execute("UPDATE items SET scope=?, content_hash=? WHERE id=?",
                                    (_json(it.scope), it.content_hash(), it.id))
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_scope_key ON items(kind,key,scope,t_clock)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_event_time ON items(t_clock,t_start)")

    def close(self) -> None:
        self.db.close()

    def write(self, items: Sequence[MemoryItem]) -> WriteResult:
        events: list[tuple[str, str]] = []
        recorded = now()
        with self.db:
            for it in items:
                h, scope = it.content_hash(), _json(it.scope)
                clock = it.event_time.clock if it.event_time else "wall"
                if self.db.execute("SELECT 1 FROM items WHERE content_hash=? AND valid_to IS NULL AND expired_at IS NULL",
                                   (h,)).fetchone():
                    events.append(("SKIP_DUP", it.id))
                    continue
                valid_from = it.valid_from if it.valid_from is not None else (
                    it.event_time.start if it.event_time else recorded)
                if not math.isfinite(valid_from):
                    raise ValueError("valid_from must be finite")
                if it.key is not None:
                    prev = self.db.execute(
                        "SELECT id,valid_from FROM items WHERE kind=? AND key=? AND scope=? AND t_clock=? "
                        "AND valid_to IS NULL AND expired_at IS NULL", (it.kind.value, it.key, scope, clock)).fetchall()
                    if any(valid_from < start for _, start in prev):
                        raise ValueError("backdated fact replacement; append late evidence without a key")
                    for pid, _ in prev:
                        self.db.execute("UPDATE items SET valid_to=? WHERE id=?", (valid_from, pid))
                        events.append(("INVALIDATE", pid))
                pos = it.pose.position if it.pose else (None, None, None)
                self.db.execute(
                    "INSERT INTO items (id,kind,key,content,payload,entities,scope,frame_id,px,py,pz,t_start,t_end,"
                    "valid_from,valid_to,recorded_at,expired_at,confidence,source,content_hash,orientation,t_clock) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it.id, it.kind.value, it.key, it.content, _json(it.model_dump(mode="json")["payload"]),
                     _json(it.entities), scope, it.pose.frame_id if it.pose else None, *pos,
                     it.event_time.start if it.event_time else None, it.event_time.end if it.event_time else None,
                     valid_from, None, recorded, None, it.confidence, it.source, h,
                     _json(it.pose.orientation) if it.pose and it.pose.orientation is not None else None, clock))
                events.append(("ADD", it.id))
        return WriteResult(events=tuple(events))

    def _where(self, q: MemoryQuery) -> tuple[str, list[Any]]:
        clauses, args = ["expired_at IS NULL"], []
        if q.kinds is not None:
            clauses.append(f"kind IN ({','.join('?' * len(q.kinds))})" if q.kinds else "0")
            args += [k.value for k in q.kinds]
        if q.clock is not None:
            clauses.append("t_clock = ?")
            args.append(q.clock)
        if q.key is not None:
            clauses.append("key = ?")
            args.append(q.key)
        if q.text:
            clauses.append("instr(content, ?) > 0")
            args.append(q.text)
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
            pose, radius = q.near
            clauses.append("frame_id = ? AND px BETWEEN ? AND ? AND py BETWEEN ? AND ?")
            args += [pose.frame_id, pose.position[0] - radius, pose.position[0] + radius,
                     pose.position[1] - radius, pose.position[1] + radius]
        return " AND ".join(clauses), args

    def _matching(self, q: MemoryQuery) -> Iterator[tuple[MemoryItem, dict[str, Any]]]:
        where, args = self._where(q)
        cur = self.db.execute(f"SELECT * FROM items WHERE {where} ORDER BY recorded_at DESC,rowid DESC", args)
        cols = [c[0] for c in cur.description]
        for row in cur:
            record = dict(zip(cols, row))
            item = self._to_item(record)
            if q.entities and not set(q.entities) & set(item.entities):
                continue
            if q.scope and any(item.scope.get(k) != v for k, v in q.scope.items()):
                continue
            if q.near is not None and math.dist(q.near[0].position, item.pose.position) > q.near[1]:
                continue
            yield item, record

    def query(self, q: MemoryQuery) -> list[Hit]:
        hits = []
        for item, record in self._matching(q):
            details = {"confidence": item.confidence}
            if q.near is not None:
                distance = math.dist(q.near[0].position, item.pose.position)
                details["spatial"] = 1.0 - distance / q.near[1] if q.near[1] else 1.0
            hits.append(Hit(item=item, score=sum(details.values()) / len(details),
                            details=details, valid_to=record["valid_to"]))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:q.k]

    def forget(self, q: MemoryQuery) -> int:
        with self.db:
            ids = [(now(), it.id) for it, _ in self._matching(q)]
            self.db.executemany("UPDATE items SET expired_at=? WHERE id=?", ids)
        return len(ids)

    def snapshot(self) -> str:
        # Include provenance, full intervals, expired records and pose/clock semantics; omitting
        # these made behaviourally different stores share a digest. This is a v2 digest change.
        cols = [c[0] for c in self.db.execute("SELECT * FROM items LIMIT 0").description]
        rows = [dict(zip(cols, row)) for row in self.db.execute("SELECT * FROM items ORDER BY id")]
        return hashlib.sha256(_json({"schema": "hylo.sqlite/v2", "rows": rows}).encode()).hexdigest()[:16]

    @staticmethod
    def _to_item(r: dict[str, Any]) -> MemoryItem:
        pose = Pose(frame_id=r["frame_id"], position=(r["px"], r["py"], r["pz"]),
                    orientation=json.loads(r["orientation"]) if r.get("orientation") else None) if r["frame_id"] else None
        span = TimeSpan(start=r["t_start"], end=r["t_end"], clock=r.get("t_clock") or "sim") if r["t_start"] is not None else None
        return MemoryItem(kind=MemoryKind(r["kind"]), content=r["content"], key=r["key"],
                          payload=json.loads(r["payload"] or "{}"), entities=tuple(json.loads(r["entities"])),
                          scope=json.loads(r["scope"]), pose=pose, event_time=span,
                          valid_from=r["valid_from"], confidence=r["confidence"], source=r["source"], id=r["id"])
