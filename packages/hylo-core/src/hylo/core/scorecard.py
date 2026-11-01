"""Scorecards (hylo.scorecard/v0.1): the result of running one policy under one benchmark protocol.

Written by hylo-bench (`hylo bench run`), read by leaderboards, the catalog, the playground and the project site, so
the format lives here. A scorecard is self-contained: the protocol and policy it measured, the headline metrics
with a 95% interval on success, per-task success, and the per-episode outcomes (`episodes`, keyed
"task#seed") that let two scorecards of the same protocol be compared pairwise.

    {"schema_version": "hylo.scorecard/v0.1", "protocol": "libero-10", "protocol_version": "1",
     "protocol_digest": "...", "policy": "smolvla", "agent": {"type": ..., "params": ...},
     "metrics": {"success_mean": 0.54, "steps_mean": 312.0, ...}, "success_ci95": [0.40, 0.67],
     "per_task": {"libero_10/0": {"success": 0.6, "n": 5}, ...}, "episodes": {"libero_10/0#30000": 1.0, ...},
     "run_dir": "runs/...", "config_digest": "...", "created_at": ...}
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from .contract import Contract, JsonAny

SCORECARD_SCHEMA = "hylo.scorecard/v0.1"


class TaskScore(Contract):
    success: float | None
    n: int


class Scorecard(Contract):
    schema_version: Literal["hylo.scorecard/v0.1"] = SCORECARD_SCHEMA
    protocol: str
    protocol_version: str
    protocol_digest: str
    policy: str                                # display name, unique per protocol on a leaderboard
    agent: dict[str, JsonAny]                  # the agent spec that was run
    metrics: dict[str, float | int | None]
    success_ci95: tuple[float, float] | None = None
    per_task: dict[str, TaskScore] = Field(default_factory=dict)
    episodes: dict[str, float | None] = Field(default_factory=dict)   # "task#seed" -> success
    run_dir: str | None = None
    config_digest: str | None = None
    code: dict[str, JsonAny] = Field(default_factory=dict)            # versions of what ran
    notes: str | None = None
    created_at: float = Field(default_factory=time.time)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Scorecard":
        return cls.model_validate_json(Path(path).read_text())


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a success rate (well behaved at 0 and 1, unlike the normal approximation)."""
    if n <= 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


__all__ = ["SCORECARD_SCHEMA", "Scorecard", "TaskScore", "wilson_interval"]
