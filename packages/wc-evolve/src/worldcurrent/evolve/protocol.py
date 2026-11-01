"""RSI evaluation protocol v2 (RFC-0002 §4): four-way split, cascade evaluation, archive, confirmatory gate.

    search split      -> diagnosis and proposer feedback (the proposer may read these traces)
    search_val split  -> candidate selection (metrics never shown to the proposer)
    gate split        -> one confirmatory paired test per selected candidate (Holm if several)
    heldout split     -> report only

Lessons applied (research/comparisons/rsi-frameworks-deep-dive.md): no single-shot acceptance; selection and
confirmation on different seeds; held-out sealed; novelty check before spending rollouts; every candidate kept in an
archive with its hypothesis and evidence; promotion only through the gate; budget accounting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from worldcurrent.eval.benchmark import RSI_SPLITS, gate_split_names, rsi_splits  # noqa: F401  (re-exported)
from worldcurrent.eval.runner import EvalResult
from worldcurrent.eval.scoring import paired_compare  # noqa: F401  (re-exported)
from worldcurrent.core.artifact import referenced_artifacts
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.contract import Contract
from worldcurrent.core.errors import ProtectedScopeViolation
from worldcurrent.core.improvement import ArchiveEntry, Proposer, StageResult  # noqa: F401 - the contract, re-exported

from .candidate import Candidate, CandidateRegistry, check_protected_scope

class Archive:
    """Every evaluated candidate with lineage, hypothesis and outcome (append-only JSONL)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, entry: ArchiveEntry) -> None:
        with open(self.path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def entries(self) -> list[ArchiveEntry]:
        if not self.path.exists():
            return []
        return [ArchiveEntry.model_validate_json(l) for l in self.path.read_text().splitlines() if l.strip()]

    def seen(self, candidate: Candidate) -> bool:
        key = json.dumps(candidate.patch, sort_keys=True, default=str)
        return any(json.dumps(e.candidate.patch, sort_keys=True, default=str) == key for e in self.entries())


def holm(p_values: list[float], alpha: float) -> list[bool]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    passed = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (len(p_values) - rank):
            passed[i] = True
        else:
            break
    return passed


Evaluate = Callable[[ExperimentConfig, str, str], EvalResult]          # (cfg, split, label) -> result
Diagnose = Callable[[EvalResult], dict[str, Any]]


class ProtocolConfig(Contract):
    alpha: float = 0.05
    stage1_pairs: int = 10            # first pairs of search_val used for early rejection
    stage1_max_deficit: int = 2       # reject if parent wins exceed candidate wins by this much in stage 1
    gate_top_k: int = 1               # candidates confirmed on the gate split (Holm-corrected)
    gate_looks: int = 1               # group-sequential looks (benchmark must provide gate, gate_r1, ...)
    spending_rho: float = 3.0         # alpha spent by look k: alpha * (k/K)^rho (rho=3 ~ O'Brien-Fleming-like)
    futility_stop: bool = True        # stop a candidate once it no longer leads the parent on the gate
    max_task_regression: float = 0.15
    max_safety_increase: float = 1.0
    max_candidates: int = 8
    budget_s: float | None = None     # wall-clock budget for the cycle
    # optional world-model pre-screen (an instrument): {env: {type: world_model, params: {path}}, split: search,
    # margin: 0.1}. A candidate predicted below parent - margin is rejected before any real rollout; the gate never
    # sees model results. Predictions are compared with search_val outcomes in the cycle report (calibration).
    prescreen: dict[str, Any] | None = None
    # the paired statistic (plugin group "gate_tests"): mcnemar on success by default; e.g. {type: sign, params:
    # {metric: steps, direction: lower}}. Regression and safety vetoes stay on success and safety.
    gate_test: dict[str, Any] = Field(default_factory=lambda: {"type": "mcnemar"})


class CycleReport(Contract):
    parent_id: str | None
    diagnosis: dict[str, Any]
    entries: list[ArchiveEntry]
    promoted: str | None
    heldout: dict[str, Any] | None
    elapsed_s: float
    prescreen: dict[str, Any] | None = None
    # the paired statistic (plugin group "gate_tests"): mcnemar on success by default; e.g. {type: sign, params:
    # {metric: steps, direction: lower}}. Regression and safety vetoes stay on success and safety.
    gate_test: dict[str, Any] = Field(default_factory=lambda: {"type": "mcnemar"})


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    import numpy as np

    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _per_task(r: EvalResult) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for s in r.log["samples"]:
        v = s["scores"]["success"]["value"]
        if v is not None:
            acc.setdefault(s["task_id"], []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def spend(alpha: float, k: int, looks: int, rho: float) -> float:
    """Nominal level for look k (1-based) under the power-family spending function; sums to alpha over all looks."""
    return alpha * ((k / looks) ** rho - ((k - 1) / looks) ** rho)


def _merge(results: list[EvalResult], metric: str = "success"
           ) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, float], float]:
    """(metric per key, success per key, success per task, mean safety interventions) over several runs."""
    keys: dict[str, float | None] = {}
    succ: dict[str, float | None] = {}
    per: dict[str, list[float]] = {}
    safety = []
    for r in results:
        keys.update(r.per_key(metric))
        succ.update(r.per_key())
        safety.append(r.metrics.get("safety_interventions_mean") or 0.0)
        for smp in r.log["samples"]:
            v = smp["scores"]["success"]["value"]
            if v is not None:
                per.setdefault(smp["task_id"], []).append(v)
    return keys, succ, {k: sum(v) / len(v) for k, v in per.items()}, sum(safety) / max(len(safety), 1)


def _rate(keys: dict[str, float | None]) -> float:
    vals = [v for v in keys.values() if v is not None]
    return sum(vals) / max(len(vals), 1)


def gate_sequential(finalists: list[ArchiveEntry], parent_cfg: ExperimentConfig, evaluate: Evaluate,
                    p: ProtocolConfig) -> dict[str, tuple[list[StageResult], bool]]:
    """Evaluate look by look; at look k test the cumulative pairs with Holm at the spent level; stop early on
    success or futility. Overall type-I error <= alpha by the union bound over looks."""
    names = gate_split_names(p.gate_looks)
    test = gate_test(p)
    parent_runs: list[EvalResult] = []
    cand_runs: dict[str, list[EvalResult]] = {e.candidate.id: [] for e in finalists}
    out: dict[str, tuple[list[StageResult], bool]] = {e.candidate.id: ([], False) for e in finalists}
    active = list(finalists)
    for k, split in enumerate(names, start=1):
        if not active:
            break
        parent_runs.append(evaluate(parent_cfg, split, f"parent-{split}"))
        pk, psucc, ptask, psafe = _merge(parent_runs, test.metric)
        level = spend(p.alpha, k, len(names), p.spending_rho)
        rows = []
        for e in active:
            cand_runs[e.candidate.id].append(evaluate(parent_cfg.with_overrides(dict(e.candidate.patch)), split,
                                                      f"{e.candidate.id}-{split}"))
            ck, csucc, ctask, csafe = _merge(cand_runs[e.candidate.id], test.metric)
            c = test.compare(ck, pk)
            reasons = [f"regression on {t.split('/')[-1]}" for t, v in ptask.items()
                       if ctask.get(t, 0.0) < v - p.max_task_regression]
            if csafe > psafe + p.max_safety_increase:
                reasons.append("more safety interventions")
            rows.append((e, c, reasons, _rate(csucc), _rate(psucc)))
        ok = holm([r[1].p_value for r in rows], level)
        still = []
        for (e, c, reasons, cs, ps), passed in zip(rows, ok):
            looks, _ = out[e.candidate.id]
            looks.append(StageResult(stage=3, success=cs, parent_success=ps, wins=c.candidate_wins, losses=c.parent_wins,
                                     n_pairs=c.n_pairs, p_value=c.p_value,
                                     notes=f"look {k}/{len(names)} at level {level:.4f}" + ("; " + "; ".join(reasons) if reasons else "")))
            if passed and not reasons:
                out[e.candidate.id] = (looks, True)
            elif reasons or (p.futility_stop and c.candidate_wins <= c.parent_wins):
                out[e.candidate.id] = (looks, False)
            else:
                still.append(e)
        active = still
    return out


def gate_test(p: ProtocolConfig) -> Any:
    from worldcurrent.core.registry import registry

    return registry.build("gate_tests", p.gate_test)


def run_cycle(base_cfg: ExperimentConfig, registry: CandidateRegistry, archive: Archive, proposer: Proposer,
              evaluate: Evaluate, diagnose: Diagnose, *, protocol: ProtocolConfig | None = None,
              approve: Callable[[Candidate, dict[str, Any]], bool] = lambda c, g: True) -> CycleReport:
    p = protocol or ProtocolConfig()
    t0 = time.time()
    parent_id = registry.active
    parent_cfg = base_cfg.with_overrides(registry.effective_patch())
    diagnosis = diagnose(evaluate(parent_cfg, "search", "parent-search"))
    parent_val = evaluate(parent_cfg, "search_val", "parent-search_val")
    test = gate_test(p)
    pv = parent_val.per_key(test.metric)
    entries: list[ArchiveEntry] = []
    pre = dict(p.prescreen or {})
    parent_pred = None
    if pre:
        pre_split, margin = pre.get("split", "search"), float(pre.get("margin", 0.1))
        parent_pred = evaluate(parent_cfg.with_overrides({"env": pre["env"]}), pre_split, "parent-prescreen").metrics["success_mean"]
    for cand, hypothesis in proposer.propose(diagnosis, parent_cfg, parent_id, archive.entries())[: p.max_candidates]:
        if p.budget_s is not None and time.time() - t0 > p.budget_s:
            break
        entry = ArchiveEntry(candidate=cand, hypothesis=hypothesis, stage_reached=0)
        try:                                                       # stage 0: scope, schema, provenance, novelty
            check_protected_scope(cand)
            cand_cfg = parent_cfg.with_overrides(dict(cand.patch))
            arts = referenced_artifacts(cand.patch)              # sealed, unmodified, no gate / held-out data
            unknown = [a for a, m in arts.items() if m.role == "behaviour" and m.provenance.params.get("unknown_provenance")]
            if unknown:                                          # adopted without stated sources: a leak cannot be ruled out
                raise ValueError(f"behaviour artifacts with unknown provenance: {unknown}")
            if arts:
                entry = entry.model_copy(update={"evidence": {"artifacts": {
                    p: {"kind": m.kind, "role": m.role, "digest": m.digest, "splits": sorted(m.provenance.splits())}
                    for p, m in arts.items()}}})
            if archive.seen(cand):
                raise ValueError("duplicate of an archived candidate")
            if pre:
                pred = evaluate(cand_cfg.with_overrides({"env": pre["env"]}), pre_split, f"{cand.id}-prescreen").metrics["success_mean"]
                entry = entry.model_copy(update={"evidence": {**entry.evidence, "prescreen": {
                    "predicted": pred, "parent_predicted": parent_pred, "split": pre_split, "margin": margin}}})
                if pred < parent_pred - margin:
                    raise ValueError(f"prescreen: world model predicts {pred:.2f} vs parent {parent_pred:.2f}")
        except (ProtectedScopeViolation, ValueError) as exc:
            entry = entry.model_copy(update={"verdict": "rejected", "evidence": {**entry.evidence, "stage0": str(exc)}})
            archive.add(entry)
            entries.append(entry)
            continue
        registry.register(cand)
        val = evaluate(cand_cfg, "search_val", f"{cand.id}-search_val")   # stages 1-2 share one evaluation
        cv = val.per_key(test.metric)
        first = list(pv)[: p.stage1_pairs]
        s1 = test.compare({k: cv.get(k) for k in first}, {k: pv[k] for k in first})
        stages = [StageResult(stage=1, wins=s1.candidate_wins, losses=s1.parent_wins, n_pairs=s1.n_pairs,
                              notes="early rejection check on the first pairs")]
        if s1.parent_wins - s1.candidate_wins >= p.stage1_max_deficit:
            entry = entry.model_copy(update={"stage_reached": 1, "stages": stages, "verdict": "rejected",
                                             "evidence": {**entry.evidence, "stage1": "clearly worse than parent"}})
        else:
            s2 = test.compare(cv, pv)
            stages.append(StageResult(stage=2, success=val.metrics["success_mean"], parent_success=parent_val.metrics["success_mean"],
                                      wins=s2.candidate_wins, losses=s2.parent_wins, n_pairs=s2.n_pairs, p_value=s2.p_value))
            entry = entry.model_copy(update={"stage_reached": 2, "stages": stages, "verdict": "archived",
                                             "evidence": {**entry.evidence, "search_val_run": str(val.run_dir),
                                                          "safety": val.metrics.get("safety_interventions_mean")}})
        entries.append(entry)
    # stage 3: group-sequential confirmation on the gate split(s); fresh init states, never seen by selection
    ranked = sorted([e for e in entries if e.stage_reached == 2],
                    key=lambda e: (e.stages[-1].wins - e.stages[-1].losses, e.stages[-1].success or 0), reverse=True)
    finalists = [e for e in ranked if (e.stages[-1].wins - e.stages[-1].losses) > 0][: p.gate_top_k]
    promoted, heldout = None, None
    if finalists:
        verdicts = gate_sequential(finalists, parent_cfg, evaluate, p)
        for e in finalists:
            i = entries.index(e)
            looks, passed = verdicts[e.candidate.id]
            entries[i] = e.model_copy(update={"stage_reached": 3, "stages": [*e.stages, *looks],
                                              "verdict": "gate_passed" if passed else "gated_fail"})
        winners = [e for e in entries if e.verdict == "gate_passed"]
        if winners:
            w = winners[0]
            ho_c = evaluate(parent_cfg.with_overrides(dict(w.candidate.patch)), "heldout", f"{w.candidate.id}-heldout")
            ho_p = evaluate(parent_cfg, "heldout", "parent-heldout")
            hc = test.compare(ho_c.per_key(test.metric), ho_p.per_key(test.metric))
            heldout = {"candidate": w.candidate.id, "candidate_success": ho_c.metrics["success_mean"],
                       "parent_success": ho_p.metrics["success_mean"], "paired": hc.model_dump(), "used_for_selection": False}
            gate = {"passed": True, "stage3": w.stages[-1].model_dump(), "heldout": heldout}
            if approve(w.candidate, gate):
                registry.promote(w.candidate.id, gate=gate, approved_by="approver")
                promoted = w.candidate.id
                entries[entries.index(w)] = w.model_copy(update={"verdict": "promoted", "evidence": {**w.evidence, "heldout": heldout}})
    for e in entries:
        if e.stage_reached > 0:                      # stage-0 rejections were archived when they happened
            archive.add(e)
            if e.verdict != "promoted":
                registry.reject(e.candidate.id, gate={"verdict": e.verdict})
    pre_report = None
    if pre:
        pairs = [(e.evidence["prescreen"]["predicted"], e.stages[1].success) for e in entries
                 if "prescreen" in e.evidence and len(e.stages) > 1 and e.stages[1].success is not None]
        pre_report = {"parent_predicted": parent_pred, "parent_search_val": parent_val.metrics["success_mean"],
                      "rejected": sum(1 for e in entries if str(e.evidence.get("stage0", "")).startswith("prescreen")),
                      "pairs": pairs, "spearman": _spearman([a for a, _ in pairs], [b for _, b in pairs])}
    return CycleReport(parent_id=parent_id, diagnosis=diagnosis, entries=entries, promoted=promoted, heldout=heldout,
                       elapsed_s=round(time.time() - t0, 1), prescreen=pre_report)
