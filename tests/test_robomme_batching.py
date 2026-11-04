from pathlib import Path
import sys
from threading import Event

import pytest

directory = str(Path(__file__).parents[1] / "examples/memory/robomme")
sys.path.insert(0, directory)
from microbatch import BatchOverloaded, Microbatcher
sys.path.remove(directory)


def test_requests_batch_without_mixing_outputs():
    seen = []
    def generate(items):
        seen.append(items)
        return ["reply:" + item for item in items]
    batcher = Microbatcher(generate, size=2, wait_seconds=1, capacity=4)
    try:
        futures = [batcher.submit(item) for item in ("episode-a", "episode-b", "episode-c", "episode-d")]
        replies = [future.result(timeout=3) for future in futures]
        assert [r.value for r in replies] == ["reply:" + item for item in ("episode-a", "episode-b", "episode-c", "episode-d")]
        assert seen == [["episode-a", "episode-b"], ["episode-c", "episode-d"]]
        assert all(r.batch_size == 2 and r.queue_seconds >= 0 and r.batch_seconds >= 0 for r in replies)
    finally:
        batcher.close()


@pytest.mark.parametrize("bad_length", [False, True])
def test_batch_error_reaches_every_request_and_next_batch_recovers(bad_length):
    calls = []
    def generate(items):
        calls.append(items)
        if len(calls) == 1:
            if bad_length:
                return []
            raise ValueError("model failure")
        return items
    batcher = Microbatcher(generate, size=2, wait_seconds=0.05, capacity=2)
    try:
        futures = [batcher.submit("a"), batcher.submit("b")]
        for future in futures:
            with pytest.raises((ValueError, RuntimeError)):
                future.result(timeout=3)
        assert batcher.submit("fresh").result(timeout=3).value == "fresh"
    finally:
        batcher.close()


def test_capacity_counts_running_requests_and_cancelled_queue_releases_slot():
    entered, release = Event(), Event()
    calls = []
    def generate(items):
        calls.append(items)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=3)
        return items
    batcher = Microbatcher(generate, size=1, capacity=2)
    try:
        first = batcher.submit("first")
        assert entered.wait(timeout=3)
        second = batcher.submit("cancelled")
        with pytest.raises(BatchOverloaded):
            batcher.submit("overload")
        assert second.cancel()
        release.set()
        assert first.result(timeout=3).value == "first"
        assert batcher.submit("next").result(timeout=3).value == "next"
        assert calls == [["first"], ["next"]]
    finally:
        release.set()
        batcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        batcher.submit("late")
