"""Benchmark = versioned set of TaskSpecs + evaluation protocol (splits, seeds, episodes)."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, model_validator

from worldcurrent.core.contract import Contract
from worldcurrent.core.task import TaskInstance, TaskSpec

STANDARD_SPLITS = ("train", "validation", "heldout")


class Benchmark(Contract):
    id: str
    version: str
    tasks: tuple[TaskSpec, ...]
    splits: dict[str, tuple[int, ...]]             # split -> seeds; each seed runs on every task
    requires_privileged: bool = False              # scored with privileged information?
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _seeds_are_disjoint(self) -> "Benchmark":
        seen: dict[int, str] = {}
        for split, seeds in self.splits.items():
            for s in seeds:
                if s in seen and seen[s] != split:
                    raise ValueError(f"seed {s} appears in both {seen[s]!r} and {split!r} splits")
                seen[s] = split
        return self

    def instances(self, split: str) -> list[TaskInstance]:
        seeds = self.splits[split]
        return [TaskInstance(spec=t, seed=s, episode_index=i, split=split)
                for t in self.tasks for i, s in enumerate(seeds)]

    def digest(self) -> str:
        blob = json.dumps({"id": self.id, "version": self.version,
                           "tasks": [t.digest() for t in self.tasks],
                           "splits": {k: list(v) for k, v in sorted(self.splits.items())}},
                          sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def standard_splits(train: int, validation: int, heldout: int) -> dict[str, tuple[int, ...]]:
    """Disjoint seed ranges: train 0.., validation 10000.., heldout 20000.."""
    return {"train": tuple(range(0, train)),
            "validation": tuple(range(10_000, 10_000 + validation)),
            "heldout": tuple(range(20_000, 20_000 + heldout))}


RSI_SPLITS = ("search", "search_val", "gate", "heldout")


def rsi_splits(n_states: int = 50, bounds: tuple[int, int, int] = (20, 30, 40), replicates: int = 1,
               gate_looks: int = 1) -> dict[str, tuple[int, ...]]:
    """RFC-0002 split of init-state indices: search (diagnosis), search_val (selection), gate (confirmation),
    heldout (report only). Replicate r reuses init state (seed % n_states) with new policy noise (seed + r * n_states).
    gate_looks > 1 adds gate_r1 .. gate_r{looks-1}: further noise replicates of the gate init states, evaluated only
    when an earlier look is inconclusive (group-sequential gate)."""
    a, b, c = bounds
    ranges = {"search": range(0, a), "search_val": range(a, b), "gate": range(b, c), "heldout": range(c, n_states)}
    splits = {k: tuple(i + r * n_states for r in range(replicates) for i in v) for k, v in ranges.items()}
    for look in range(1, gate_looks):
        r = replicates + look - 1
        splits[f"gate_r{look}"] = tuple(i + r * n_states for i in ranges["gate"])
    return splits


def gate_split_names(gate_looks: int) -> list[str]:
    return ["gate", *[f"gate_r{k}" for k in range(1, gate_looks)]]
