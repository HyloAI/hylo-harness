"""Improvement candidates and the candidate registry.

- A Candidate is declarative: a patch of dotted config paths (v0.1). Code candidates are not
  accepted until an isolated runtime exists (same stance as Zetta stages.py).
- ProtectedScope: candidates may not touch safety, benchmark, scoring or trace configuration.
- Registry: immutable versions with parent links, an `active` pointer, an append-only ledger,
  and rollback as a first-class operation (no upstream project implements rollback).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from worldcurrent.core.errors import ProtectedScopeViolation
from worldcurrent.core.improvement import Candidate, CandidateKind  # noqa: F401 - the contract, re-exported

PROTECTED_PREFIXES = ("safety", "benchmark", "trace", "scorers", "evolution", "schema_version", "env")


# Kinds accepted by the v0.1 gate. "code" (model-generated executable code) is deliberately not
# enabled: the decision is pending (RFC-0001 D5) and requires an isolated execution runtime first.
ENABLED_KINDS: frozenset[str] = frozenset({"config_patch"})


def check_protected_scope(candidate: Candidate, protected: tuple[str, ...] = PROTECTED_PREFIXES) -> None:
    if candidate.kind not in ENABLED_KINDS:
        raise ProtectedScopeViolation(f"candidate kind {candidate.kind!r} is not enabled (enabled: {sorted(ENABLED_KINDS)})")
    for path in candidate.patch:
        if path.split(".")[0] in protected:
            raise ProtectedScopeViolation(f"candidate {candidate.id} touches protected scope {path!r}")


class CandidateRegistry:
    """Directory-backed registry: versions/<id>.json, ledger.jsonl, ACTIVE."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)

    def _ledger(self, op: str, **fields: Any) -> None:
        with open(self.root / "ledger.jsonl", "a") as f:
            f.write(json.dumps({"op": op, "at": time.time(), **fields}, default=str) + "\n")

    def register(self, candidate: Candidate, base_config: Mapping[str, Any] | None = None) -> None:
        path = self.root / "versions" / f"{candidate.id}.json"
        if path.exists():
            raise FileExistsError(candidate.id)
        path.write_text(json.dumps({"candidate": candidate.model_dump(mode="json"), "digest": candidate.digest(),
                                    "base_config": base_config}, indent=2, default=str))
        self._ledger("register", id=candidate.id, parent=candidate.parent_id, digest=candidate.digest())

    def get(self, cid: str) -> dict[str, Any]:
        return json.loads((self.root / "versions" / f"{cid}.json").read_text())

    @property
    def active(self) -> str | None:
        p = self.root / "ACTIVE"
        return p.read_text().strip() or None if p.exists() else None

    def _set_active(self, cid: str | None) -> None:
        (self.root / "ACTIVE").write_text(cid or "")

    def promote(self, cid: str, *, gate: Mapping[str, Any], approved_by: str) -> None:
        rec = self.get(cid)
        if rec["candidate"]["parent_id"] != self.active:
            raise ValueError(f"stale candidate: parent {rec['candidate']['parent_id']} != active {self.active}")
        if not gate.get("passed"):
            raise ValueError("cannot promote a candidate whose gate did not pass")
        previous = self.active
        self._set_active(cid)
        self._ledger("promote", id=cid, previous=previous, approved_by=approved_by, gate=dict(gate))

    def reject(self, cid: str, *, gate: Mapping[str, Any]) -> None:
        self._ledger("reject", id=cid, gate=dict(gate))

    def rollback(self, *, reason: str, by: str) -> str | None:
        current = self.active
        if current is None:
            raise ValueError("nothing to roll back")
        parent = self.get(current)["candidate"]["parent_id"]
        self._set_active(parent)
        self._ledger("rollback", id=current, restored=parent, reason=reason, by=by)
        return parent

    def lineage(self, cid: str | None = None) -> list[str]:
        out, cur = [], cid or self.active
        while cur:
            out.append(cur)
            cur = self.get(cur)["candidate"]["parent_id"]
        return out

    def effective_patch(self) -> dict[str, Any]:
        """Patches compose along the lineage, oldest first."""
        patch: dict[str, Any] = {}
        for cid in reversed(self.lineage()):
            patch.update(self.get(cid)["candidate"]["patch"])
        return patch

    def ledger(self) -> list[dict[str, Any]]:
        p = self.root / "ledger.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
