import pytest

from worldcurrent.eval.scoring import paired_compare
from worldcurrent.eval.stats import McNemar, PairedPermutation, SignTest, binomial_check


@pytest.mark.parametrize("test", [McNemar(), SignTest(), PairedPermutation(n_resamples=2000)])
def test_builtin_tests_hold_their_level_and_detect_a_clear_effect(test):
    r = binomial_check(test, n=40, trials=150)
    assert r["type1_rate"] <= 0.08 and r["power"] >= 0.7


def test_mcnemar_is_the_previous_gate_and_sign_handles_continuous_lower_is_better():
    c, p = {"a": 1.0, "b": 0.0, "c": 1.0}, {"a": 0.0, "b": 0.0, "c": 1.0}
    assert McNemar().compare(c, p) == paired_compare(c, p)
    fewer_steps = SignTest(metric="steps", direction="lower").compare({f"k{i}": 40.0 for i in range(8)}, {f"k{i}": 60.0 for i in range(8)})
    assert fewer_steps.candidate_wins == 8 and fewer_steps.p_value == pytest.approx(1 / 256)


def test_exact_permutation_p_value():
    # differences +1, +1, +1, -1: 5 of the 16 sign assignments reach a sum >= 2
    r = PairedPermutation().compare({"a": 2, "b": 2, "c": 2, "d": 0}, {"a": 1, "b": 1, "c": 1, "d": 1})
    assert r.p_value == pytest.approx(5 / 16) and (r.candidate_wins, r.parent_wins) == (3, 1)
