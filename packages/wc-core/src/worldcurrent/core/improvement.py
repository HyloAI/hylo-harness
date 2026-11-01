"""The `generators` plugin contract (wc.generator) and the records of an improvement campaign.

A generator (Proposer) reads a parent's diagnosis and the campaign's archive and proposes candidates: declarative
patches of dotted config paths, each with a hypothesis. The cascade, the gate and the registry that decide what
happens to a candidate live in wc-evolve; these are the shapes a generator plugin sees and returns.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Literal, Mapping, Protocol

from pydantic import Field

from .config import ExperimentConfig
from .contract import Contract

CandidateKind = Literal["config_patch", "critic_rule", "recovery_rule", "prompt", "skill", "policy_ref", "code"]


class Candidate(Contract):
    patch: dict[str, Any]                    # {"agent.params.kp": 2.0}
    rationale: str
    parent_id: str | None
    generator: str
    kind: CandidateKind = "config_patch"
    evidence: dict[str, Any] = Field(default_factory=dict)   # diagnosis refs
    id: str = Field(default_factory=lambda: "cand-" + uuid.uuid4().hex[:10])

    def digest(self) -> str:
        return hashlib.sha256(json.dumps({"patch": self.patch, "parent": self.parent_id, "kind": self.kind},
                                         sort_keys=True, default=str).encode()).hexdigest()[:16]


class StageResult(Contract):
    stage: int
    success: float | None = None
    parent_success: float | None = None
    wins: int = 0
    losses: int = 0
    n_pairs: int = 0
    p_value: float | None = None
    notes: str = ""


class ArchiveEntry(Contract):
    candidate: Candidate
    hypothesis: str
    stage_reached: int
    stages: list[StageResult] = Field(default_factory=list)
    verdict: str = "pending"             # rejected | archived | gated_fail | promoted
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class Proposer(Protocol):
    name: str

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]: ...   # (candidate, hypothesis)


__all__ = ["ArchiveEntry", "Candidate", "CandidateKind", "Proposer", "StageResult"]
