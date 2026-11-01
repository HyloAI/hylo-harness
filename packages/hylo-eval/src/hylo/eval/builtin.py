"""The test benchmark "testing/reach-v0" over the test double environment (hylo.testing): for unit tests
only. Real benchmarks live in hylo-bench."""

from __future__ import annotations

from hylo.testing.point_reach import reach_tasks

from .benchmark import Benchmark, rsi_splits, standard_splits


def testing_benchmark(tags: list[str] | None = None, train: int = 10, validation: int = 20, heldout: int = 20,
                   protocol: str = "standard", gate_looks: int = 1) -> Benchmark:
    tasks = tuple(t for t in reach_tasks() if not tags or set(tags) & set(t.tags))
    splits = rsi_splits(100, (40, 60, 80), gate_looks=gate_looks) if protocol == "rsi" else standard_splits(train, validation, heldout)
    return Benchmark(id="testing/reach-v0", version="0.1", tasks=tasks,
                     splits=splits,
                     metadata={"tags": ",".join(tags or [])})
