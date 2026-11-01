"""Scorers read the trace only; they never touch the Agent or the Environment.
A Score of None is an abstention (infra failure, agent error) and is excluded from metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Protocol

from pydantic import Field

from worldcurrent.core.contract import Contract
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.trace import EpisodeTrace


class Score(Contract):
    value: float | None
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scorer(Protocol):
    name: str

    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score: ...


class SuccessScorer:
    name = "success"

    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score:
        end = trace.last("episode.end")
        if end is None:
            return Score(value=None, explanation="episode has no end event")
        status = end["payload"]["status"]
        if status in ("infra_invalid", "error"):
            return Score(value=None, explanation=f"abstain: {status}")
        return Score(value=1.0 if status == "success" else 0.0, explanation=status)


class SafetyInterventionScorer:
    name = "safety_interventions"

    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score:
        n = sum(1 for e in trace.of_type("safety.review") if e["payload"]["modified"])
        aborted = any(e["payload"].get("phase") == "safety" for e in trace.of_type("error"))
        return Score(value=float(n + (1 if aborted else 0)), explanation=f"{n} modified actions, aborted={aborted}")


class StepsScorer:
    name = "steps"

    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score:
        end = trace.last("episode.end")
        return Score(value=float(end["payload"]["steps"]) if end else None)


DEFAULT_SCORERS: tuple[Scorer, ...] = (SuccessScorer(), SafetyInterventionScorer(), StepsScorer())


def mean_ignoring_none(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class PairedComparison(Contract):
    n_pairs: int
    candidate_wins: int          # candidate success, parent failure
    parent_wins: int             # parent success, candidate failure
    p_value: float               # one-sided exact McNemar, H1: candidate better


def one_sided_exact_mcnemar(candidate_wins: int, parent_wins: int) -> float:
    """P[X >= candidate_wins], X ~ Binomial(discordant, 0.5). Same test as Zetta gating.py."""
    discordant = candidate_wins + parent_wins
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(candidate_wins, discordant + 1))
    return tail / 2 ** discordant


def paired_compare(candidate: Mapping[str, float | None], parent: Mapping[str, float | None]) -> PairedComparison:
    """Pairs on TaskInstance.key (task id + seed). Pairs where either side abstained are dropped."""
    keys = [k for k in candidate if k in parent and candidate[k] is not None and parent[k] is not None]
    cw = sum(1 for k in keys if candidate[k] and not parent[k])
    pw = sum(1 for k in keys if parent[k] and not candidate[k])
    return PairedComparison(n_pairs=len(keys), candidate_wins=cw, parent_wins=pw,
                            p_value=one_sided_exact_mcnemar(cw, pw))
