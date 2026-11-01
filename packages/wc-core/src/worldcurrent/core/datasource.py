"""External data contracts (RFC-0004): demonstrations, videos and other robots' logs.

A DataSource yields ExternalEpisodes. `worldcurrent.learn.data.ingest` writes them as a run under split
`external:<name>` (sealed, with origin and licence); every label says who produced it ("source", "judge:<digest>"
or none) and an unlabeled episode is never treated as a success.
"""

from __future__ import annotations

from pydantic import Field

from .contract import Contract, JsonAny, NDArray


class ExternalFrame(Contract):
    state: dict[str, NDArray] = Field(default_factory=dict)
    images: dict[str, NDArray] = Field(default_factory=dict)
    action: NDArray | None = None                         # None: the source has no action labels (e.g. video)


class ExternalEpisode(Contract):
    episode_id: str
    task_id: str
    instruction: str = ""
    frames: list[ExternalFrame]
    success: bool | None = None                           # the source's own outcome label, if it has one
    meta: dict[str, JsonAny] = Field(default_factory=dict)


def external_split(name: str) -> str:
    return f"external:{name}"
