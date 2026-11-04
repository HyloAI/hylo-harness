"""CPU event-memory diagnostics across ten demands, NOT an external robot benchmark.

Inputs are reference observable event streams: perception and manipulation are deliberately
outside this diagnostic. Compare complete evidence with an eight-event FIFO and a last-event
baseline. Random seed controls cues, identities, counts and delays; answers come from the
independent task construction below. No model, simulator privilege or external source is used.

    python examples/memory/06_event_memory.py --quick --out /tmp/hylo-events
"""
from __future__ import annotations

import random
import time

from _common import parse_args, write_result
from hylo.core.episode_memory import EpisodeScope, MemoryEvent, RecallQuery
from hylo.memory.episode import EventMemory

SLUG = "event_memory"
TYPES = ("spatial", "object", "capacity", "sequential", "tracking", "negative", "checklist", "temporal",
         "procedural", "prospective")


def case(kind: str, seed: int):
    rng = random.Random(seed)
    events = []

    def add(value, category="cue", subject="target"):
        step = len(events) + 1
        events.append(MemoryEvent(id=f"event:{step}", step=step, time=float(step), kind=category,
                                  subject=subject, value=value, evidence=(f"visible-frame:{step}",), source="diagnostic/v1"))

    def distract():
        for _ in range(rng.randint(16, 40)):
            add(rng.randint(0, 100), "distractor")

    query = dict(name=kind, op="latest", kind="cue", complete=True)
    if kind in ("spatial", "object"):
        expected = [rng.uniform(-1, 1), rng.uniform(-1, 1)] if kind == "spatial" else rng.choice(["red", "green", "blue"])
        add(expected)
        distract()
    elif kind in ("capacity", "sequential", "procedural"):
        if kind == "capacity":
            expected = rng.sample(list("abcdefgh"), rng.randint(3, 7))
        elif kind == "sequential":
            expected = [rng.choice(["red", "blue", "red"]) for _ in range(rng.randint(4, 12))]
        else:
            expected = [{"frame_id": "table", "position": [rng.random(), rng.random(), 0], "orientation": [0, 0, 0, 1]}
                        for _ in range(rng.randint(4, 8))]
        for value in expected:
            add(value)
            distract()
        query["op"] = "sequence"
    elif kind == "tracking":
        slots = ["a", "b", "c"]
        target = rng.choice(slots)
        add(slots.index(target))
        for _ in range(rng.randint(2, 8)):
            a, b = rng.sample(range(3), 2)
            slots[a], slots[b] = slots[b], slots[a]
            add(slots.index(target))  # Reference result of an observed identity-preserving swap.
        expected = slots.index(target)
        distract()
    elif kind == "negative":
        universe = list("abcdef")
        missing = rng.choice(universe)
        expected = [missing]
        for subject in universe:
            if subject != missing:
                add("seen", subject=subject)
        distract()
        query.update(op="unseen", universe=tuple(universe))
    elif kind == "checklist":
        outcomes = {str(i): bool(rng.randrange(2)) for i in range(6)}
        target = rng.choice(list(outcomes))
        for subject, value in outcomes.items():
            add(value, subject=subject)
        expected = outcomes[target]
        distract()
        query["subject"] = target
    elif kind == "temporal":
        expected = rng.randint(2, 9)
        for _ in range(expected):
            add("completed transition")
            distract()
        query["op"] = "count"
    else:
        expected = [{"when": "work completed", "action": rng.choice(["press red", "press blue"])}]
        add(expected[0], "intent", "remember-later")
        add({"when": "now", "action": "other"}, "intent", "other")
        distract()
        add("observed completion", "intent.done", "other")
        distract()
        query.update(op="pending", kind="intent")
    return events, RecallQuery(**query), expected


def run_case(events, query, expected, window=None):
    m = EventMemory()
    m.reset(EpisodeScope(task_id="diagnostic", episode_id="one"))
    if window is not None and len(events) > window:
        # A truncated stream cannot honestly claim complete counting/negative evidence.
        m.append([MemoryEvent(id="missing-prefix", step=0, time=0, kind="_gap", subject="stream",
                              evidence=("diagnostic:missing-prefix",), source="diagnostic/v1")])
        events = events[-window:]
    m.append(events)
    answer = m.recall(query)
    correct = answer.status == "known" and answer.value == expected
    m.close()
    return correct


def main():
    args = parse_args(SLUG, __doc__)
    seeds = 4 if args.quick else 100
    started = time.perf_counter()
    results = {}
    failures = []
    for kind in TYPES:
        counts = {"full_events": 0, "fifo_8": 0, "last_event": 0}
        for seed in range(seeds):
            events, query, expected = case(kind, 10000 + seed)
            for label, window in [("full_events", None), ("fifo_8", 8), ("last_event", 1)]:
                correct = run_case(events, query, expected, window)
                counts[label] += correct
                if label == "full_events" and not correct:
                    failures.append({"kind": kind, "seed": 10000 + seed})
        results[kind] = {"cases": seeds, "correct": counts}
    output = {"example": SLUG, "official_benchmark": False, "inputs": "reference observable events, no RGB or motor policy",
              "seed_start": 10000, "seeds_per_type": seeds, "results": results, "failures": failures,
              "runtime_s": time.perf_counter() - started}
    write_result(args.out, output)
    print(f"Event diagnostics: {10*seeds-len(failures)}/{10*seeds} complete-evidence cases correct")
    print(f"Reference event inputs only; not RoboMME/MIKASA performance. Results: {args.out / 'result.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
