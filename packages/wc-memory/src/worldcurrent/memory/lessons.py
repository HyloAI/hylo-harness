"""LessonBook: language lessons from the agent's own experience, curated by delta operations (RFC-0003 M4).

After ExpeL (ADD / UPVOTE / DOWNVOTE / EDIT insight lists) and ACE (itemised playbook with helpful / harmful counters,
incremental deltas instead of rewrites, which avoids "context collapse"). An LLM reflector proposes operations; this
class applies them deterministically:
- ADD whose title is near-identical (ratio >= dedup) to an active lesson in the same scope becomes an UPVOTE;
- UPVOTE / DOWNVOTE move the counters (outcome feedback: candidates that cited a lesson and were promoted or refuted);
- EDIT replaces the content and keeps the counters; RETIRE removes it from retrieval;
- a lesson with harmful >= helpful + retire_margin retires itself.
Every applied operation is appended to ops.jsonl (audit trail); lessons.json is the current state.

    lessons: {type: lesson_book, params: {path: runs/lessons/libero10, frozen: true}}
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from worldcurrent.core.experience import EpisodeRef, Lesson

from .action_bank import FrozenMemoryError

OPS = ("ADD", "UPVOTE", "DOWNVOTE", "EDIT", "RETIRE")


class LessonBook:
    def __init__(self, path: str, frozen: bool = False, dedup: float = 0.9, retire_margin: int = 2) -> None:
        self.root, self.frozen, self.dedup, self.retire_margin = Path(path), frozen, dedup, retire_margin
        self._lessons: dict[str, Lesson] = {}
        state = self.root / "lessons.json"
        if state.exists():
            for d in json.loads(state.read_text()):
                lesson = Lesson.model_validate(d)
                self._lessons[lesson.id] = lesson
        elif not frozen:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- curation ---------------------------------------------------------------------------------------
    def _similar(self, scope: str, title: str) -> Lesson | None:
        t = title.strip().lower()
        best, score = None, 0.0
        for les in self._lessons.values():
            if les.status == "active" and les.scope == scope:
                r = difflib.SequenceMatcher(None, t, les.title.strip().lower()).ratio()
                if r > score:
                    best, score = les, r
        return best if score >= self.dedup else None

    def apply(self, ops: Sequence[dict[str, Any]], *, source: str = "") -> list[dict[str, Any]]:
        """Apply operations in order; returns what was actually done (invalid operations are reported, not raised)."""
        if self.frozen:
            raise FrozenMemoryError("this lesson book is a frozen snapshot")
        done = []
        for op in ops:
            kind = str(op.get("op", "")).upper()
            try:
                if kind == "ADD":
                    scope, title = str(op["scope"]), str(op["title"]).strip()
                    twin = self._similar(scope, title)
                    if twin is not None:
                        done.append(self._vote(twin.id, +1, op, note=f"duplicate of {twin.id}"))
                        continue
                    lid = "L" + hashlib.sha256(f"{scope}|{title}".encode()).hexdigest()[:8]
                    self._lessons[lid] = Lesson(id=lid, scope=scope, kind=op.get("kind", "strategy"), title=title,
                                                content=str(op["content"]).strip(),
                                                evidence=[EpisodeRef.model_validate(e) for e in op.get("evidence", [])])
                    done.append({"op": "ADD", "id": lid})
                elif kind in ("UPVOTE", "DOWNVOTE"):
                    done.append(self._vote(str(op["id"]), +1 if kind == "UPVOTE" else -1, op))
                elif kind == "EDIT":
                    les = self._lessons[str(op["id"])]
                    self._lessons[les.id] = les.model_copy(update={"content": str(op["content"]).strip()})
                    done.append({"op": "EDIT", "id": les.id})
                elif kind == "RETIRE":
                    les = self._lessons[str(op["id"])]
                    self._lessons[les.id] = les.model_copy(update={"status": "retired"})
                    done.append({"op": "RETIRE", "id": les.id})
                else:
                    raise ValueError(f"unknown op {kind!r}")
            except (KeyError, ValueError, TypeError) as exc:
                done.append({"op": kind or "?", "error": f"{type(exc).__name__}: {exc}"[:200]})
        self._save(done, source)
        return done

    def _vote(self, lid: str, sign: int, op: dict[str, Any], note: str = "") -> dict[str, Any]:
        les = self._lessons[lid]
        upd = {"helpful": les.helpful + (sign > 0), "harmful": les.harmful + (sign < 0)}
        ev = [EpisodeRef.model_validate(e) for e in op.get("evidence", [])]
        if ev:
            upd["evidence"] = (les.evidence + ev)[-20:]
        les = les.model_copy(update=upd)
        if les.harmful >= les.helpful + self.retire_margin:
            les = les.model_copy(update={"status": "retired"})
        self._lessons[lid] = les
        return {"op": "UPVOTE" if sign > 0 else "DOWNVOTE", "id": lid, **({"note": note} if note else {}),
                **({"retired": True} if les.status == "retired" else {})}

    def _save(self, done: list[dict[str, Any]], source: str) -> None:
        (self.root / "lessons.json").write_text(json.dumps([l.model_dump(mode="json") for l in self._lessons.values()], indent=2))
        with open(self.root / "ops.jsonl", "a") as f:
            for d in done:
                f.write(json.dumps({"t": time.time(), "source": source, **d}) + "\n")

    # -- retrieval --------------------------------------------------------------------------------------
    def query(self, scopes: Sequence[str] = ("global",), *, kind: str | None = None, top_k: int = 8) -> list[Lesson]:
        """Active lessons in any of `scopes`, best net votes first (ties: most evidence)."""
        hits = [l for l in self._lessons.values() if l.status == "active" and l.scope in scopes
                and (kind is None or l.kind == kind)]
        hits.sort(key=lambda l: (l.helpful - l.harmful, len(l.evidence)), reverse=True)
        return hits[:top_k]

    def sources(self) -> set[str]:
        """Sources of applied operations (lets writers stay idempotent across resumed campaigns)."""
        f = self.root / "ops.jsonl"
        return {json.loads(l).get("source", "") for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()

    def all(self, status: str | None = None) -> list[Lesson]:
        return [l for l in self._lessons.values() if status is None or l.status == status]

    def snapshot(self) -> str:
        h = hashlib.sha256()
        for lid in sorted(self._lessons):
            h.update(self._lessons[lid].model_dump_json().encode())
        return h.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self._lessons)
