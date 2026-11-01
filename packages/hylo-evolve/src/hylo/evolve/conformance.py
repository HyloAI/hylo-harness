"""Conformance for generator plugins (proposers)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.core.errors import ProtectedScopeViolation
from hylo.testing.conformance import ConformanceReport, _run
from hylo.testing.setup import check_experiment

from .candidate import Candidate, check_protected_scope
from .steering import diagnose_traces

def check_generator(proposer: Any, *, cfg: ExperimentConfig | None = None) -> ConformanceReport:
    rep = ConformanceReport(subject=f"generator {getattr(proposer, 'name', type(proposer).__name__)}")
    cfg = check_experiment(cfg)                     # MuJoCo planar reach when installed, else the test double
    out: list[Any] = []

    def proposes() -> str:
        diag = diagnose_traces(run_experiment(cfg, split="search", run_root=Path(tempfile.mkdtemp(prefix="hylo-gen-")), label="diag"))
        out.extend(proposer.propose(diag, cfg, None, []))
        for item in out:
            cand, hyp = item
            if not isinstance(cand, Candidate) or not isinstance(hyp, str) or not hyp:
                raise AssertionError("propose returns (Candidate, hypothesis) pairs with a non-empty hypothesis")
        return f"{len(out)} candidates"

    def scoped() -> None:
        for cand, _ in out:
            try:
                check_protected_scope(cand)
            except ProtectedScopeViolation as exc:
                raise AssertionError(f"{cand.id}: {exc}") from exc
            cfg.with_overrides(dict(cand.patch))                    # the patch applies to the config

    def named() -> None:
        if not getattr(proposer, "name", None) or any(c.generator != proposer.name for c, _ in out):
            raise AssertionError("candidates carry the proposer's name as generator")

    _run(rep, "proposes (Candidate, hypothesis) pairs", proposes)
    _run(rep, "candidates stay outside protected scopes and apply to the config", scoped, skip=not out)
    _run(rep, "candidates name their generator", named, skip=not out)
    return rep
