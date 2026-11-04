"""Bounded request microbatches with one model caller and per-request futures."""
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import BoundedSemaphore, Lock, Thread
import math
import time


class BatchOverloaded(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchReply:
    value: object
    batch_size: int
    queue_seconds: float
    batch_seconds: float


class Microbatcher:
    def __init__(self, call_batch, *, size=2, wait_seconds=0.2, capacity=16):
        if size < 1 or not math.isfinite(wait_seconds) or wait_seconds < 0 or capacity < size:
            raise ValueError("invalid batching limits")
        self.call_batch, self.size, self.wait_seconds = call_batch, size, wait_seconds
        self.queue, self.slots, self.lock = Queue(), BoundedSemaphore(capacity), Lock()
        self.closed = False
        self.thread = Thread(target=self._run, name="vision-microbatch", daemon=True)
        self.thread.start()

    def submit(self, value):
        with self.lock:
            if self.closed:
                raise RuntimeError("batcher is closed")
            if not self.slots.acquire(blocking=False):
                raise BatchOverloaded("vision request capacity exhausted")
            future = Future()
            self.queue.put((value, future, time.monotonic()))
            return future

    def _run(self):
        stopping = False
        while not stopping:
            first = self.queue.get()
            if first is None:
                break
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.size:
                try:
                    item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                except Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            active = []
            for item in batch:
                if item[1].set_running_or_notify_cancel():
                    active.append(item)
                else:
                    self.slots.release()
            if not active:
                continue
            started = time.monotonic()
            failure = None
            try:
                values = list(self.call_batch([item[0] for item in active]))
                if len(values) != len(active):
                    raise RuntimeError("model returned a different number of batch results")
            except Exception as exc:
                failure = exc
            elapsed = time.monotonic() - started
            for index, (_, future, queued) in enumerate(active):
                self.slots.release()
                if failure:
                    future.set_exception(failure)
                else:
                    future.set_result(BatchReply(values[index], len(active), started - queued, elapsed))

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join(timeout=5)
