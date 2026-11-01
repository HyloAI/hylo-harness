"""Gate tests (plugin group "gate_tests"): the paired statistic the improvement protocol uses to decide.

A gate test compares candidate and parent on the same (task, seed) keys and returns a PairedComparison:
`candidate_wins` / `parent_wins` (pairs where one side is better, used by early rejection and futility), and a
one-sided `p_value` for H1 "the candidate is better". The protocol around it (cascade, group-sequential looks,
alpha spending, Holm, regression and safety vetoes, held-out report) is unchanged whichever test is chosen.

    protocol: {gate_test: {type: sign, params: {metric: steps, direction: lower}}}

Built in:
- mcnemar (default): exact one-sided McNemar on a binary metric (success);
- sign: exact one-sided sign test on any metric, with a direction and an optional minimum difference;
- paired_permutation: sign-flip permutation test on the mean paired difference of a continuous metric.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from hylo.core.scoring import GateTest, PairedComparison  # noqa: F401 - the contract, re-exported

from .scoring import one_sided_exact_mcnemar, paired_compare


def _pairs(candidate: Mapping[str, float | None], parent: Mapping[str, float | None]) -> list[tuple[float, float]]:
    return [(float(candidate[k]), float(parent[k])) for k in candidate
            if k in parent and candidate[k] is not None and parent[k] is not None]


class McNemar:
    name = "mcnemar"

    def __init__(self, metric: str = "success") -> None:
        self.metric = metric

    def compare(self, candidate, parent) -> PairedComparison:
        return paired_compare(candidate, parent)


class SignTest:
    """Pairs where the candidate is better by more than `min_diff` against pairs where it is worse."""

    name = "sign"

    def __init__(self, metric: str = "success", direction: Literal["higher", "lower"] = "higher", min_diff: float = 0.0) -> None:
        self.metric, self.sign, self.min_diff = metric, 1.0 if direction == "higher" else -1.0, float(min_diff)

    def compare(self, candidate, parent) -> PairedComparison:
        d = [self.sign * (c - p) for c, p in _pairs(candidate, parent)]
        cw = sum(1 for x in d if x > self.min_diff)
        pw = sum(1 for x in d if x < -self.min_diff)
        return PairedComparison(n_pairs=len(d), candidate_wins=cw, parent_wins=pw, p_value=one_sided_exact_mcnemar(cw, pw))


class PairedPermutation:
    """Sign-flip permutation test of the mean paired difference (exact when 2^n is small, Monte Carlo otherwise)."""

    name = "paired_permutation"

    def __init__(self, metric: str = "success", direction: Literal["higher", "lower"] = "higher",
                 n_resamples: int = 20000, seed: int = 0) -> None:
        self.metric, self.sign, self.n, self.seed = metric, 1.0 if direction == "higher" else -1.0, n_resamples, seed

    def compare(self, candidate, parent) -> PairedComparison:
        d = np.array([self.sign * (c - p) for c, p in _pairs(candidate, parent)])
        cw, pw = int((d > 0).sum()), int((d < 0).sum())
        nz = d[d != 0]
        if len(nz) == 0:
            return PairedComparison(n_pairs=len(d), candidate_wins=cw, parent_wins=pw, p_value=1.0)
        observed = nz.sum()
        if len(nz) <= 16:                                          # exact: every sign assignment
            signs = ((np.arange(2 ** len(nz))[:, None] >> np.arange(len(nz))) & 1) * 2 - 1
            p = float(np.mean((signs * np.abs(nz)).sum(axis=1) >= observed - 1e-12))
        else:                                                      # Monte Carlo, with the observed assignment counted
            signs = np.random.default_rng(self.seed).choice([-1, 1], size=(self.n, len(nz)))
            p = float((np.sum((signs * np.abs(nz)).sum(axis=1) >= observed - 1e-12) + 1) / (self.n + 1))
        return PairedComparison(n_pairs=len(d), candidate_wins=cw, parent_wins=pw, p_value=min(1.0, p))


def binomial_check(test: GateTest, n: int = 40, trials: int = 200, seed: int = 0) -> dict[str, float]:
    """Calibration helper for new tests: rejection rate at alpha 0.05 under H0 (identical agents) and under a
    clear effect. A valid one-sided test stays near or below 0.05 under H0."""
    rng = np.random.default_rng(seed)
    h0 = h1 = 0
    for _ in range(trials):
        parent = {f"k{i}": float(rng.random() < 0.5) for i in range(n)}
        same = {f"k{i}": float(rng.random() < 0.5) for i in range(n)}
        better = {f"k{i}": float(rng.random() < 0.8) for i in range(n)}
        h0 += test.compare(same, parent).p_value <= 0.05
        h1 += test.compare(better, parent).p_value <= 0.05
    return {"type1_rate": h0 / trials, "power": h1 / trials}


__all__ = ["GateTest", "McNemar", "SignTest", "PairedPermutation", "binomial_check"]
