"""The `scorers` and `gate_tests` plugin contracts (hylo.scorer, hylo.gate_test) and the data they exchange.

A scorer reads one episode's trace (never the agent or the environment) and returns a Score; a Score of None is an
abstention (infra failure, agent error) and is excluded from metrics. A gate test compares per-episode values of one
metric between a candidate and its parent, paired on (task, seed), and returns a PairedComparison. Implementations
(success, steps, McNemar, sign, permutation) live in hylo-eval.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import Field

from .contract import Contract
from .task import TaskInstance
from .trace import EpisodeTrace


class Score(Contract):
    value: float | None
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scorer(Protocol):
    name: str

    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score: ...


class PairedComparison(Contract):
    n_pairs: int
    candidate_wins: int          # candidate success, parent failure
    parent_wins: int             # parent success, candidate failure
    p_value: float               # one-sided, H1: candidate better


@runtime_checkable
class GateTest(Protocol):
    name: str
    metric: str                  # the scorer whose per-episode value is compared

    def compare(self, candidate: Mapping[str, float | None], parent: Mapping[str, float | None]) -> PairedComparison: ...


__all__ = ["GateTest", "PairedComparison", "Score", "Scorer"]
