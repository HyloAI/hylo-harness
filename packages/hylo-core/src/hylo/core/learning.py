"""Learning pipeline contracts (RFC-0004): data source -> labeler -> learner -> sealed artifact -> candidate -> gate.

Contracts only. A learner turns development data into one artifact directory and seals it
(`artifact.write_manifest`). Running jobs (in-process or with another venv's interpreter), calibration and data
ingestion live in hylo-learn (`hylo.learn`).
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator, Protocol, runtime_checkable

from pydantic import Field

from .artifact import DEV_SPLITS, ArtifactManifest
from .contract import Contract, JsonAny
from .trace import EpisodeTrace


class TrainJob(Contract):
    learner: str                                   # "learners" plugin name or module:attr
    inputs: list[str]                              # run directories or roots to search
    out: str                                       # the artifact directory this job seals
    splits: tuple[str, ...] = DEV_SPLITS
    params: dict[str, JsonAny] = Field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json(exclude={"out"}).encode()).hexdigest()[:16]


@runtime_checkable
class Learner(Protocol):
    """Builds and seals one artifact from a job. Must refuse forbidden splits (use artifact.development_runs)."""

    name: str

    def fit(self, job: TrainJob) -> ArtifactManifest: ...


@runtime_checkable
class OnlineLearner(Protocol):
    """Learns by acting (policy search, on-policy RL, iterative self-training): each iteration it proposes agent
    specs to roll out, the runner evaluates them on a development split (recorded as ordinary runs) and hands back
    the results, the learner updates. `seal` writes the artifact; its provenance lists every rollout run.

    Run by hylo.learn.online.run_online; a TrainJob whose learner has `policies` / `update` is online and
    carries the experiment config in params["experiment"]."""

    name: str

    def start(self, job: TrainJob, agent: dict[str, Any]) -> None: ...          # the base agent spec
    def policies(self) -> list[dict[str, Any]]: ...                              # agent specs for this iteration
    def update(self, results: list[dict[str, Any]]) -> dict[str, Any]: ...      # one entry per policy: agent, run_dir, metrics, samples
    def finished(self) -> bool: ...
    def seal(self, out: str, sources: list[Any]) -> ArtifactManifest: ...


@runtime_checkable
class DataSource(Protocol):
    """Episodes from outside the campaign's own runs (demonstrations, videos, other robots).

    `core.datasource.ingest` turns them into a run under split external:<name> with origin and licence. Episodes may
    lack actions (videos) or outcomes; labelers fill those in and say who produced each label."""

    name: str
    origin: str
    licence: str

    def episodes(self) -> Iterator[Any]: ...          # core.datasource.ExternalEpisode


@runtime_checkable
class Labeler(Protocol):
    """Adds labels an episode is missing (outcome, progress, latent actions). Labels from an instrument carry its
    artifact digest; an instrument never labels the data it was calibrated on."""

    name: str

    def label(self, episode: EpisodeTrace) -> dict[str, Any]: ...
