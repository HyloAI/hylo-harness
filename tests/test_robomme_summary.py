import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "robomme_summary", Path(__file__).parents[1] / "examples/memory/robomme/summarize.py")
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


def row(episode, condition, status, **changes):
    return {"task": "task", "split": "val", "episode": episode, "seed": episode,
            "condition": condition, "status": status, "seconds": 1., "steps": 10,
            "reset_observation_sha256": str(episode), **changes}


def test_paired_statistics_keep_invalid_attempt_denominators():
    rows = []
    for i, (off, on) in enumerate([("fail", "success"), ("fail", "success"),
                                   ("success", "fail"), ("success", "agent_error")]):
        rows.extend([row(i, "memory_off", off), row(i, "memory_on", on)])
    result = summary_module.summarize(rows)
    assert result["overall"]["memory_on"]["success_per_attempt"] == .5
    assert result["paired"]["memory_wins"] == 2
    assert result["paired"]["memory_losses"] == 2
    assert result["paired"]["invalid_pairs"] == 0
    assert result["paired"]["exact_mcnemar_one_sided"] == .6875


def test_new_conditions_infrastructure_and_incomplete_pairs():
    result = summary_module.summarize([
        row(0, "history", "success"), row(0, "working", "agent_error"),
        row(1, "history", "infra_error", reset_observation_sha256=None), row(1, "working", "success", comparison_fingerprint="configured"),
        row(2, "working", "success")], baseline="history", candidate="working")
    assert result["paired"]["memory_losses"] == 1
    assert result["paired"]["invalid_pairs"] == 1
    assert result["paired"]["incomplete_pairs"] == 1
    assert result["overall"]["working"]["success_per_attempt"] == 2 / 3


@pytest.mark.parametrize("changes", [{"seed": 99}, {"reset_observation_sha256": "different"},
                                      {"reset_observation_sha256": None}, {"comparison_fingerprint": "changed"}])
def test_pairs_require_matching_reset_evidence(changes):
    with pytest.raises(ValueError):
        summary_module.summarize([row(0, "memory_off", "fail"), row(0, "memory_on", "success", **changes)])
