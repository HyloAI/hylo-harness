import pytest

from hylo.core.episode_memory import EpisodeScope, MemoryEvent, RecallQuery
from hylo.memory.episode import EventMemory


def event(i, value, kind="cue", subject="object", **kw):
    return MemoryEvent(id=f"e{i}", step=i, time=float(i), kind=kind, subject=subject, value=value,
                       evidence=(f"frame:{i}",), source="test-visible/v1", **kw)


def memory(**kw):
    m = EventMemory(**kw)
    m.reset(EpisodeScope(task_id="task", episode_id="one"))
    return m


def query(op, kind="cue", **kw):
    return RecallQuery(name="answer", op=op, kind=kind, **kw)


def test_ordered_recall_retains_repetitions_and_event_relative_reference():
    m = memory()
    es = [event(0, "red"), event(1, "blue"), event(2, "button", kind="press"), event(3, "red")]
    assert m.append(es) == 4 and m.append(es) == 0
    assert m.recall(query("sequence", complete=True)).value == ["red", "blue", "red"]
    assert m.recall(query("nth", index=1)).value == "blue"
    assert m.recall(query("latest", before="e2")).value == "blue"
    assert m.recall(query("first", after="e2")).value == "red"
    assert m.recall(query("latest", as_of_step=0)).value == "red"
    assert m.recall(query("latest", after="e2", as_of_step=1)).status == "unknown"
    assert m.recall(query("count", complete=True)).value == 3
    assert m.recall(query("count")).status == "partial"


def test_updated_location_does_not_erase_history():
    m = memory()
    m.append([event(0, {"frame_id": "map", "xyz": [0, 1, 0]}),
              event(4, {"frame_id": "map", "xyz": [2, 1, 0]})])
    assert m.recall(query("latest")).value["xyz"][0] == 2
    assert m.recall(query("latest", as_of_step=0)).value["xyz"][0] == 0


def test_negative_recall_requires_closed_world():
    m = memory()
    m.append([event(0, "seen", subject="red"), event(1, "seen", subject="blue")])
    q = query("unseen", universe=("red", "blue", "green"))
    assert m.recall(q).status == "unknown"
    assert m.recall(q.model_copy(update={"complete": True})).value == ["green"]


def test_intention_survives_delay_and_only_observed_completion_removes_it():
    m = memory()
    m.append([event(0, {"trigger": "all cubes placed", "action": "press red"}, kind="intent", subject="reminder")])
    q = query("pending", kind="intent", complete=True)
    assert m.recall(q).value[0]["action"] == "press red"
    m.append([event(200, "done", kind="intent.done", subject="reminder")])
    assert m.recall(q).value == []
    assert len(m.recall(q.model_copy(update={"as_of_step": 199})).value) == 1


def test_gaps_and_ambiguous_updates_do_not_become_confident_answers():
    m = memory()
    m.append([event(0, "left"), event(1, None, kind="_gap")])
    assert m.recall(query("latest")).status == "unknown"
    assert m.recall(query("count", complete=True)).status == "unknown"
    m.append([event(2, "right", confidence=0.3)])
    assert m.recall(query("latest")).status == "unknown"  # Do not fall back to confident but stale "left".
    m.append([event(3, "center")])
    assert m.recall(query("latest")).value == "center"


def test_episode_reset_and_explicit_resume(tmp_path):
    path = str(tmp_path / "journal.sqlite")
    m = memory(path=path)
    m.append([event(0, "secret")])
    digest = m.snapshot()
    m.reset(EpisodeScope(task_id="task", episode_id="two"))
    assert m.recall(query("latest")).status == "unknown"
    with pytest.raises(ValueError, match="resume"):
        m.reset(EpisodeScope(task_id="task", episode_id="one"))
    m.close()
    m = EventMemory(path)
    m.reset(EpisodeScope(task_id="task", episode_id="one"), resume=True)
    assert m.snapshot() == digest and m.recall(query("latest")).value == "secret"


def test_batches_fail_atomically_and_payloads_are_not_mutable_aliases():
    m = memory(max_events=2)
    e = event(0, {"x": [1]})
    m.append([e])
    e.value["x"][0] = 99
    assert m.recall(query("latest")).value == {"x": [1]}
    with pytest.raises(OverflowError):
        m.append([event(1, "a"), event(2, "b")])
    assert m.recall(query("count", complete=True)).value == 1
    with pytest.raises(ValueError, match="ID reused"):
        m.append([event(0, "different")])
    with pytest.raises(ValueError, match="increasing"):
        m.append([event(2, "a"), event(1, "b")])
    assert m.recall(query("count", complete=True)).value == 1


def test_context_limit_reports_partial_instead_of_complete_sequence():
    m = memory()
    m.append([event(i, i) for i in range(4)])
    result = m.recall(query("sequence", complete=True, limit=2))
    assert result.status == "partial" and result.value == [0, 1]
    assert result.evidence == ("frame:0", "frame:1")


def test_elapsed_time_uses_clock_not_number_of_policy_calls():
    m = memory()
    m.append([event(0, "start", kind="timer")])
    q = query("elapsed", kind="timer", at_time=13.25)
    assert m.recall(q).value == 13.25
    assert m.recall(q.model_copy(update={"at_time": None})).status == "unknown"
    m.append([event(14, "restart", kind="timer")])
    assert m.recall(q).status == "unknown"


def test_returned_values_cannot_mutate_retained_evidence():
    m = memory()
    m.append([event(0, {"coordinates": [1, 2, 3]})])
    digest = m.snapshot()
    answer = m.recall(query("latest"))
    answer.value["coordinates"][0] = 999
    assert m.snapshot() == digest


def test_reversed_event_bounds_are_unknown_not_zero():
    m = memory()
    m.append([event(0, "a"), event(1, "b")])
    assert m.recall(query("count", before="e0", after="e1", complete=True)).status == "unknown"
