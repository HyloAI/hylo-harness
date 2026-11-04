"""Exact event recall above a durable SQLite evidence journal (RFC-0009).

Independent implementation of public event/sequence memory ideas; no external project code.
The journal cannot turn missing/ambiguous perception into ground truth. No automatic summary
replaces evidence, and budget exhaustion fails explicitly rather than dropping an early cue.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from hylo.core.episode_memory import EpisodeScope, MemoryEvent, RecallQuery, RecallResult
from hylo.core.memory import MemoryItem, MemoryKind, MemoryQuery, TimeSpan

from .sqlite import SQLiteMemoryStore


def _position(event: MemoryEvent) -> tuple[int, int]:
    return event.step, event.order


def _extend_digest(digest: str, events: Sequence[MemoryEvent]) -> str:
    for event in events:
        payload = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(bytes.fromhex(digest) + payload.encode()).hexdigest()
    return digest


class EventMemory:
    def __init__(self, path: str = ":memory:", *, max_events: int = 10000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.store = SQLiteMemoryStore(path)
        self.max_events = max_events
        self.scope: EpisodeScope | None = None
        self._events: dict[str, MemoryEvent] = {}

    def _scope(self) -> dict[str, str]:
        if self.scope is None:
            raise RuntimeError("reset with an episode scope before using memory")
        return {**self.scope.model_dump(), "memory": "episode-events/v1"}

    def reset(self, scope: EpisodeScope, *, resume: bool = False) -> None:
        target = {**scope.model_dump(), "memory": "episode-events/v1"}
        hits = self.store.query(MemoryQuery(kinds=frozenset({MemoryKind.EPISODIC}), scope=target,
                                            k=self.max_events + 1))
        if hits and not resume:
            raise ValueError("episode already has evidence; use a new ID or explicit resume=True")
        if len(hits) > self.max_events:
            raise OverflowError("persisted evidence exceeds max_events")
        events = [MemoryEvent.model_validate(h.item.payload["event"]) for h in hits]
        self.scope = scope
        self._events = {e.id: e for e in sorted(events, key=_position)}
        initial = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
        self._digest = _extend_digest(initial, list(self._events.values()))

    def append(self, events: Sequence[MemoryEvent]) -> int:
        scope = self._scope()
        if not events:
            return 0
        additions: dict[str, MemoryEvent] = {}
        last = next(reversed(self._events.values()), None)
        tail = _position(last) if last else (-1, -1)
        tail_time = last.time if last else 0.0
        items = []
        for original in events:
            # Isolate nested payloads from a caller changing them after append.
            e = MemoryEvent.model_validate_json(original.model_dump_json())
            existing = additions.get(e.id, self._events.get(e.id))
            if existing is not None:
                if existing != e:
                    raise ValueError(f"event ID reused with different evidence: {e.id}")
                continue
            if _position(e) <= tail or e.time < tail_time:
                raise ValueError("events must have increasing (step, order) and nondecreasing time")
            if len(self._events) + len(additions) >= self.max_events:
                raise OverflowError("event budget exhausted; refusing silent evidence loss")
            additions[e.id] = e
            tail, tail_time = _position(e), e.time
            identity = json.dumps([scope, e.id], sort_keys=True)
            items.append(MemoryItem(
                id=hashlib.sha256(identity.encode()).hexdigest(), kind=MemoryKind.EPISODIC,
                content=f"{e.kind}:{e.subject}", scope=scope, event_time=TimeSpan(start=e.time),
                entities=(e.subject,), source=e.source, confidence=e.confidence,
                payload={"event": e.model_dump(mode="json")},
            ))
        self.store.write(items)  # One transaction; cache changes only after persistence succeeds.
        self._events.update(additions)
        self._digest = _extend_digest(self._digest, list(additions.values()))
        return len(items)

    def recall(self, query: RecallQuery) -> RecallResult:
        self._scope()
        q = query
        events = [e for e in self._events.values() if q.as_of_step is None or e.step <= q.as_of_step]
        available = {e.id: e for e in events}
        for anchor in (q.before, q.after):
            if anchor is not None and anchor not in available:
                return RecallResult(name=q.name, status="unknown", reason="missing or future event anchor")
        if q.before is not None and q.after is not None and _position(available[q.after]) >= _position(available[q.before]):
            return RecallResult(name=q.name, status="unknown", reason="reversed event bounds")
        if q.after is not None:
            events = [e for e in events if _position(e) > _position(available[q.after])]
        if q.before is not None:
            events = [e for e in events if _position(e) < _position(available[q.before])]
        kinds = {q.kind, q.kind + ".done"} if q.op == "pending" else {q.kind}
        selected = [e for e in events if e.kind in kinds and (q.subject is None or e.subject == q.subject)]
        gap = any(e.kind == "_gap" for e in events)

        def result(value=None, chosen=selected, status="known", reason=""):
            answer = RecallResult(name=q.name, status=status, value=value,
                                  evidence=tuple(dict.fromkeys(ref for e in chosen for ref in e.evidence)), reason=reason)
            return RecallResult.model_validate_json(answer.model_dump_json())

        if q.op == "elapsed":
            if not selected or q.at_time is None or q.at_time < selected[-1].time:
                return result(status="unknown", reason="elapsed recall needs an observed anchor and causal clock")
            event = selected[-1]
            if event.confidence < q.min_confidence or gap:
                return result(status="unknown", reason="uncertain clock anchor")
            return result(q.at_time - event.time, [event])

        if q.op in ("latest", "first", "nth"):
            if not selected:
                return result(status="unknown", reason="no observed evidence")
            index = -1 if q.op == "latest" else (q.index if q.op == "nth" else 0)
            if index >= len(selected):
                return result(status="unknown", reason="ordinal outside observed history")
            event = selected[index]
            if event.confidence < q.min_confidence:
                return result(chosen=[event], status="unknown", reason="ambiguous observation")
            # A gap after a last-known fact may conceal an update. Never call stale state current.
            relevant_gap = any(e.kind == "_gap" and (
                q.op != "latest" or _position(e) > _position(event)) for e in events)
            if relevant_gap:
                return result(chosen=[event], status="unknown", reason="observation gap")
            if q.op == "nth" and any(e.confidence < q.min_confidence for e in selected[:index + 1]):
                return result(status="unknown", reason="uncertain ordinal prefix")
            return result(event.value, [event])

        if gap or any(e.confidence < q.min_confidence for e in selected):
            return result(status="unknown", reason="incomplete or ambiguous event history")
        if q.op == "count":
            return result(len(selected), status="known" if q.complete else "partial",
                          reason="" if q.complete else "observed count is only a lower bound")
        if q.op == "unseen":
            if not q.complete or not q.universe:
                return result(status="unknown", reason="negative recall needs a complete observed set and universe")
            seen = {e.subject for e in selected}
            return result([subject for subject in dict.fromkeys(q.universe) if subject not in seen])
        if q.op == "pending":
            last = {e.subject: e for e in selected}
            selected = [e for e in last.values() if e.kind == q.kind]
        truncated = len(selected) > q.limit
        chosen = selected[:q.limit]
        return result([e.value for e in chosen], chosen,
                      status="known" if q.complete and not truncated else "partial",
                      reason="result budget exceeded" if truncated else ("" if q.complete else "observed history only"))

    def snapshot(self) -> str:
        self._scope()
        return self._digest

    def close(self) -> None:
        self.store.close()
