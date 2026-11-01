"""Task contracts. A TaskSpec covers both LIBERO-style declarative tasks with fixed init states and
RoboCasa-style code tasks with seeded/snapshot init and dynamic instructions."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Union

from pydantic import Field

from .contract import Contract, JsonAny


class SeedInit(Contract):
    kind: Literal["seed"] = "seed"


class InitStateRef(Contract):
    kind: Literal["state_ref"] = "state_ref"
    uri: str                     # e.g. "libero://libero_10/<task>.pruned_init"
    count: int = Field(ge=1)     # stored init states; index = episode_index % count


class SnapshotInit(Contract):
    kind: Literal["snapshot"] = "snapshot"
    uri: str                     # serialized simulator snapshot (RoboCasa ep_meta + xml + states)


InitSpec = Annotated[Union[SeedInit, InitStateRef, SnapshotInit], Field(discriminator="kind")]


class Predicate(Contract):
    """Success predicate: builtin name, declarative `expr` (e.g. BDDL goal) or callable `ref`."""

    kind: Literal["builtin", "expr", "callable"]
    name: str = ""
    expr: str | None = None
    ref: str | None = None       # "package.module:Class.method"


class TaskSpec(Contract):
    id: str                      # globally unique, e.g. "testing/reach_center"
    version: str
    instruction: str             # static text or template; the env reports the realised text
    init: InitSpec = Field(default_factory=SeedInit)
    success: Predicate = Field(default_factory=lambda: Predicate(kind="builtin", name="goal_reached"))
    max_steps: int = Field(default=200, ge=1)
    params: dict[str, JsonAny] = Field(default_factory=dict)  # env-specific task parameters
    tags: tuple[str, ...] = ()

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class TaskInstance(Contract):
    """One concrete episode request: a task, a seed, and where it sits in the protocol."""

    spec: TaskSpec
    seed: int
    episode_index: int = 0
    split: str = "train"

    @property
    def key(self) -> str:
        """Stable pairing key used for same-seed comparisons across agent versions."""
        return f"{self.spec.id}#{self.seed}"
