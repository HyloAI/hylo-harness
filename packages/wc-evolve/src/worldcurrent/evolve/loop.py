"""Offline improvement cycle: Execute(train) -> Diagnose -> Generate -> Check -> Validate -> Held-out -> Promote.

Split discipline: diagnosis reads the `train` split (privileged state allowed); candidate
selection uses the `validation` split (paired, same seeds); `heldout` is only reported.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from pydantic import Field

from worldcurrent.eval.runner import EvalResult
from worldcurrent.eval.scoring import paired_compare
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.contract import Contract
from worldcurrent.core.errors import ProtectedScopeViolation
from worldcurrent.core.trace import TraceStore
from .candidate import Candidate, CandidateRegistry, check_protected_scope


class FailureReport(Contract):
    n_episodes: int
    n_failures: int
    buckets: dict[str, int]
    examples: dict[str, list[str]] = Field(default_factory=dict)  # bucket -> episode ids


def diagnose(result: EvalResult, near_threshold: float = 0.2) -> FailureReport:
    """Deterministic failure bucketing from the trace (an LLM diagnoser would add hypotheses on top)."""
    store = TraceStore(result.run_dir)
    buckets: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    failures = 0
    for s in result.log["samples"]:
        if s["status"] == "success":
            continue
        failures += 1
        tr = store.read(s["episode_id"])
        end = tr.last("episode.end")["payload"]
        priv = tr.last("privileged.state")
        dist = priv["payload"]["distance"] if priv and "distance" in priv["payload"] else None
        interventions = end.get("safety_interventions", 0)
        if s["status"] == "safety_abort":
            b = "safety_abort"
        elif s["status"] in ("error", "infra_invalid"):
            b = f"infra:{s['status']}"
        elif interventions > 5:
            b = "oscillation_or_aggressive"
        elif dist is not None and dist > near_threshold:
            b = "timeout_far"
        else:
            b = "timeout_near"
        buckets[b] += 1
        examples.setdefault(b, []).append(s["episode_id"])
    return FailureReport(n_episodes=len(result.log["samples"]), n_failures=failures, buckets=dict(buckets),
                         examples=examples)


class CandidateGenerator(Protocol):
    name: str

    def propose(self, report: FailureReport, cfg: ExperimentConfig, parent_id: str | None) -> list[Candidate]: ...


class GainRuleGenerator:
    """Maps failure buckets to gain changes. Stands in for an LLM/DSPy generator."""

    name = "rule.gain@0.1"

    def __init__(self, path: str = "agent.params.kp", factors: tuple[float, ...] = (2.0, 4.0, 8.0)) -> None:
        self.path, self.factors = path, factors

    def propose(self, report: FailureReport, cfg: ExperimentConfig, parent_id: str | None) -> list[Candidate]:
        kp = float(cfg.agent.params.get("kp", 1.0))
        too_slow = report.buckets.get("timeout_far", 0) + report.buckets.get("timeout_near", 0)
        too_fast = report.buckets.get("oscillation_or_aggressive", 0) + report.buckets.get("safety_abort", 0)
        factors = self.factors if too_slow >= too_fast else tuple(1 / f for f in self.factors)
        return [Candidate(patch={self.path: round(kp * f, 4)}, parent_id=parent_id, generator=self.name,
                          rationale=f"{report.buckets} -> scale kp by {f:g}", evidence={"buckets": report.buckets})
                for f in factors]


class GateResult(Contract):
    passed: bool
    reasons: list[str]
    validation: dict[str, Any]
    heldout: dict[str, Any] | None = None


def gate(candidate_val: EvalResult, parent_val: EvalResult, *, alpha: float = 0.05,
         max_safety_increase: float = 0.0, max_task_regression: float = 0.1) -> GateResult:
    cmp = paired_compare(candidate_val.per_key("success"), parent_val.per_key("success"))
    reasons: list[str] = []
    c_s, p_s = candidate_val.metrics["success_mean"] or 0.0, parent_val.metrics["success_mean"] or 0.0
    c_si = candidate_val.metrics.get("safety_interventions_mean") or 0.0
    p_si = parent_val.metrics.get("safety_interventions_mean") or 0.0
    if cmp.p_value >= alpha:
        reasons.append(f"not significant: McNemar p={cmp.p_value:.4f} (wins {cmp.candidate_wins}/{cmp.parent_wins})")
    if c_s < p_s:
        reasons.append(f"success regressed {p_s:.3f} -> {c_s:.3f}")
    if c_si > p_si + max_safety_increase:
        reasons.append(f"safety interventions increased {p_si:.2f} -> {c_si:.2f}")
    # Per-task regression suite: no task may drop by more than max_task_regression.
    def per_task(r: EvalResult) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for s in r.log["samples"]:
            v = s["scores"]["success"]["value"]
            if v is not None:
                acc.setdefault(s["task_id"], []).append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}
    pc, pp = per_task(candidate_val), per_task(parent_val)
    for task, ps in pp.items():
        if pc.get(task, 0.0) < ps - max_task_regression:
            reasons.append(f"task regression on {task}: {ps:.2f} -> {pc.get(task, 0.0):.2f}")
    return GateResult(passed=not reasons, reasons=reasons,
                      validation={"paired": cmp.model_dump(), "candidate_success": c_s, "parent_success": p_s,
                                  "candidate_safety": c_si, "parent_safety": p_si})


Evaluate = Callable[[ExperimentConfig, str, str], EvalResult]   # (cfg, split, label) -> result


class CycleReport(Contract):
    diagnosis: FailureReport
    evaluated: list[dict[str, Any]]
    promoted: str | None
    heldout: dict[str, Any] | None


def improvement_cycle(base_cfg: ExperimentConfig, registry: CandidateRegistry, generator: CandidateGenerator,
                      evaluate: Evaluate, *, approve: Callable[[Candidate, GateResult], bool] = lambda c, g: True,
                      alpha: float = 0.05, gate_kwargs: Mapping[str, Any] | None = None) -> CycleReport:
    parent_id = registry.active
    parent_cfg = base_cfg.with_overrides(registry.effective_patch())
    train = evaluate(parent_cfg, "train", "parent-train")
    report = diagnose(train)
    parent_val = evaluate(parent_cfg, "validation", "parent-validation")
    evaluated: list[dict[str, Any]] = []
    best: tuple[Candidate, GateResult, EvalResult] | None = None
    for cand in generator.propose(report, parent_cfg, parent_id):
        try:
            check_protected_scope(cand)
        except ProtectedScopeViolation as exc:
            evaluated.append({"id": cand.id, "patch": dict(cand.patch), "rejected": str(exc)})
            continue
        registry.register(cand, base_config=None)
        cand_cfg = parent_cfg.with_overrides(dict(cand.patch))
        val = evaluate(cand_cfg, "validation", f"{cand.id}-validation")
        g = gate(val, parent_val, alpha=alpha, **dict(gate_kwargs or {}))
        evaluated.append({"id": cand.id, "patch": dict(cand.patch), "gate": g.model_dump(mode="json")})
        if not g.passed:
            registry.reject(cand.id, gate=g.model_dump(mode="json"))
            continue
        if best is None or val.metrics["success_mean"] > best[2].metrics["success_mean"]:
            best = (cand, g, val)
    promoted, heldout = None, None
    if best is not None:
        cand, g, _ = best
        cand_cfg = parent_cfg.with_overrides(dict(cand.patch))
        ho_c = evaluate(cand_cfg, "heldout", f"{cand.id}-heldout")
        ho_p = evaluate(parent_cfg, "heldout", "parent-heldout")
        heldout = {"candidate_success": ho_c.metrics["success_mean"], "parent_success": ho_p.metrics["success_mean"],
                   "paired": paired_compare(ho_c.per_key(), ho_p.per_key()).model_dump(), "used_for_selection": False}
        g = g.model_copy(update={"heldout": heldout})
        if approve(cand, g):
            registry.promote(cand.id, gate={"passed": True, **g.validation, "heldout": heldout}, approved_by="approver")
            promoted = cand.id
    return CycleReport(diagnosis=report, evaluated=evaluated, promoted=promoted, heldout=heldout)


class GainProposer:
    """`Proposer`-protocol version of GainRuleGenerator for campaigns: scale one numeric config path."""

    name = "rule.gain@0.2"

    def __init__(self, path: str = "agent.params.kp", factors: tuple[float, ...] = (2.0, 4.0)) -> None:
        self.path, self.factors = path, tuple(factors)

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[Any]) -> list[tuple[Candidate, str]]:
        node: Any = cfg.model_dump(mode="json")
        for part in self.path.split("."):
            node = node[part]
        return [(Candidate(patch={self.path: round(float(node) * f, 4)}, rationale="", parent_id=parent_id,
                           generator=self.name, evidence={"buckets": dict(diagnosis.get("buckets", {}))}),
                 f"scaling {self.path} by {f:g} reduces the failures in {dict(diagnosis.get('buckets', {}))}")
                for f in self.factors]
