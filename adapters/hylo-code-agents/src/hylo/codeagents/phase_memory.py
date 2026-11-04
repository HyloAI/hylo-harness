"""Phase-scoped repair notes, retrieved only from active, promoted program lineages.

Independent Hylo implementation of a design studied in HarnessPAI; no upstream code is
incorporated. Notes share the sealed program's digest. They describe repairs, not separately
validated executable skills. Rolling back a campaign removes the retired branch from recall.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from pydantic import Field

from hylo.core.contract import Contract
from hylo.core.improvement import Candidate
from hylo.core.store import Store, is_ref

from .skills import read_program


class PhasePlan(Contract):
    name: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=1, max_length=500)
    completion: str = Field(min_length=1, max_length=500)
    fallback: str = Field(min_length=1, max_length=500)
    step_budget: int = Field(ge=1, le=10000)


class PhasePlans(Contract):
    phases: list[PhasePlan] = Field(min_length=1, max_length=12)


class RepairNote(Contract):
    phase: str = Field(min_length=1, max_length=80)
    trigger: str = Field(min_length=1, max_length=500)
    repair: str = Field(min_length=1, max_length=1500)
    limitation: str = Field(min_length=1, max_length=500)
    failure_refs: list[str] = Field(min_length=1, max_length=12)
    success_refs: list[str] = Field(min_length=1, max_length=12)


class RepairNotes(Contract):
    repairs: list[RepairNote] = Field(default_factory=list, max_length=8)


def validate_notes(notes: RepairNotes, phases: PhasePlans, evidence: dict) -> None:
    """Every note cites an actual failed case and a successful case of the selected program."""
    names = {phase.name for phase in phases.phases}
    for note in notes.repairs:
        if note.phase not in names:
            raise ValueError(f"unknown repair phase: {note.phase}")
        for refs, expected in ((note.failure_refs, False), (note.success_refs, True)):
            for ref in refs:
                row = evidence.get(ref)
                if row is None or row["success"] is not expected or (expected and not row["selected"]):
                    raise ValueError(f"unsupported repair evidence: {ref}")
        failed_keys = {evidence[ref]["key"] for ref in note.failure_refs}
        successful_keys = {evidence[ref]["key"] for ref in note.success_refs}
        if not failed_keys.intersection(successful_keys):
            raise ValueError("a repair needs a failed and successful replay of the same development case")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class PhaseRepairMemory:
    """Read-only view over campaign registries; never imports an unpromoted draft note."""

    def __init__(self, campaigns=()):
        self.campaigns = tuple(Path(p) for p in campaigns)
        self.skipped: list[dict] = []

    def retrieve(self, query: str, *, phase: str | None = None, k: int = 4) -> list[dict]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        self.skipped = []
        terms = _tokens(query)
        found = {}
        for campaign in self.campaigns:
            root = campaign / "registry"
            campaign_found = {}
            if not (root / "ACTIVE").exists() or not (root / "ledger.jsonl").exists():
                continue
            try:
                ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines() if line]
                promoted = {r["id"] for r in ledger if r.get("op") == "promote" and r.get("gate", {}).get("passed")}
                current = (root / "ACTIVE").read_text().strip()
                visited = set()
                while current:
                    if current in visited or not re.fullmatch(r"[A-Za-z0-9_-]+", current):
                        raise ValueError("invalid candidate lineage")
                    visited.add(current)
                    record = json.loads((root / "versions" / f"{current}.json").read_text())
                    candidate = Candidate.model_validate(record["candidate"])
                    if candidate.id != current or candidate.digest() != record["digest"] or current not in promoted:
                        raise ValueError("lineage is not a digest-matched promoted version")
                    current = candidate.parent_id
                    if candidate.kind != "code":
                        continue
                    patch = candidate.patch
                    artifact = patch.get("agent.params.artifact") or (patch.get("agent.params") or {}).get("artifact")
                    if not artifact:
                        continue
                    try:
                        _, manifest, meta = read_program(Store().path(artifact) if is_ref(artifact) else artifact)
                        admitted_digest = (candidate.evidence.get("artifact") or {}).get("digest")
                        if manifest.digest != admitted_digest:
                            raise ValueError("program digest differs from the admitted candidate")
                        pack = meta.get("phase_repairs")
                        if not pack:
                            continue
                        phases = PhasePlans.model_validate({"phases": pack["phases"]})
                        notes = RepairNotes.model_validate({"repairs": pack["repairs"]})
                        validate_notes(notes, phases, pack["evidence"])
                        for index, note in enumerate(notes.repairs):
                            words = _tokens(f"{note.phase} {note.trigger} {note.repair}")
                            score = len(terms & words) / max(1, len(terms))
                            if phase and phase.casefold() == note.phase.casefold():
                                score += 1.0
                            if score <= 0:
                                continue
                            rid = hashlib.sha256(f"{manifest.digest}:{index}".encode()).hexdigest()[:20]
                            campaign_found[rid] = {"id": rid, **note.model_dump(), "score": score,
                                          "candidate": candidate.id, "program_digest": manifest.digest,
                                          "program_artifact": str(artifact), "campaign": str(campaign),
                                          "program_gate_passed": True, "transfer_verified": False}
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        self.skipped.append({"candidate": candidate.id, "reason": str(exc)})
                found.update(campaign_found)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.skipped.append({"campaign": str(campaign), "reason": str(exc)})
        return sorted(found.values(), key=lambda row: (-row["score"], row["id"]))[:k]
