"""Conformance for gate-test plugins (hylo.eval.stats.GateTest)."""

from __future__ import annotations

from typing import Any

from hylo.testing.conformance import ConformanceReport, _run

from .stats import binomial_check


def check_gate_test(test: Any, *, trials: int = 200) -> ConformanceReport:
    rep = ConformanceReport(subject=f"gate test {getattr(test, 'name', type(test).__name__)}")

    def interface() -> str:
        if not isinstance(getattr(test, "metric", None), str):
            raise AssertionError("gate tests declare the metric they compare (test.metric)")
        r = test.compare({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0})
        for f in ("n_pairs", "candidate_wins", "parent_wins", "p_value"):
            getattr(r, f)
        if not 0.0 <= r.p_value <= 1.0:
            raise AssertionError(f"p-value {r.p_value} outside [0, 1]")
        return f"metric {test.metric}"

    def no_evidence() -> None:
        same = {f"k{i}": float(i % 2) for i in range(20)}
        if test.compare(same, same).p_value < 0.5:
            raise AssertionError("identical results must not look significant")

    def level() -> str:
        r = binomial_check(test, trials=trials)
        if r["type1_rate"] > 0.08:
            raise AssertionError(f"type-I error {r['type1_rate']:.3f} > 0.08 at alpha 0.05 (identical agents)")
        return f"type-I {r['type1_rate']:.3f}, power {r['power']:.2f} (binary metric, 40 pairs, 0.5 -> 0.8)"

    _run(rep, "returns a PairedComparison with a p-value in [0, 1]", interface)
    _run(rep, "identical results are not significant", no_evidence)
    _run(rep, "holds alpha 0.05 under H0", level)
    return rep
