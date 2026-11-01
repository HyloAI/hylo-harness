"""Instruments (RFC-0004): learned components that measure or label instead of act.

A success judge or progress model labels episodes (so memory and training data can be written without simulator
ground truth); a world model pre-screens candidates. Instruments are trusted through calibration, not the gate:
`hylo.learn.calibrate` scores a judge against simulator outcomes on runs the judge was not built from and
stores the CalibrationReport in the judge's artifact manifest. An instrument never decides a gate verdict: in simulation the gate always
scores with ground truth, and a world model may only choose which candidates are worth real evaluation budget.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import Field

from .contract import Contract, JsonAny, NDArray
from .trace import EpisodeTrace
from .types import Observation


class Judgement(Contract):
    p_success: float = Field(ge=0.0, le=1.0)
    progress: list[float] | None = None          # per decision point, if the instrument estimates it
    info: dict[str, JsonAny] = Field(default_factory=dict)


@runtime_checkable
class SuccessJudge(Protocol):
    """Judges an episode from what a real robot would have: observations and actions, no privileged state."""

    name: str

    def judge(self, episode: EpisodeTrace) -> Judgement: ...


@runtime_checkable
class ValueModel(Protocol):
    """Expected (discounted) success from a state, optionally after executing a chunk."""

    name: str

    def value(self, obs: Observation, chunk: np.ndarray | None = None) -> float: ...


@runtime_checkable
class WorldModel(Protocol):
    """Action-conditioned predictor; used to pre-screen candidates, never to replace their evaluation."""

    name: str

    def rollout(self, obs: Observation, actions: np.ndarray) -> list[Observation]: ...


class CalibrationReport(Contract):
    n: int
    base_rate: float
    auc: float | None
    brier: float
    ece: float                                   # expected calibration error, 10 equal-width bins
    threshold: float | None                      # lowest threshold reaching target_precision (None: unreachable)
    precision: float | None                      # at `threshold`
    recall: float | None
    target_precision: float
    ready: bool                                  # may label data without simulator ground truth
    runs: list[str] = Field(default_factory=list)
    scores: NDArray | None = None                # kept for re-analysis; excluded from the manifest
