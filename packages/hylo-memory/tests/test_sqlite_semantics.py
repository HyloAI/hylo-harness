import sqlite3

import pytest

from hylo.core.memory import MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan
from hylo.memory.sqlite import SQLiteMemoryStore, _SCHEMA


def fact(**updates):
    fields = dict(kind=MemoryKind.SPATIAL, key="cup", content="cup", scope={"episode_id": "one"},
                  event_time=TimeSpan(start=0), pose=Pose(frame_id="map", position=(0, 0, 0)))
    return MemoryItem(**{**fields, **updates})


def test_scope_isolates_duplicates_and_replacements():
    m = SQLiteMemoryStore()
    m.write([fact(), fact(scope={"episode_id": "two"})])
    assert len(m.query(MemoryQuery())) == 2
    m.write([fact(content="moved", event_time=TimeSpan(start=1))])
    assert m.query(MemoryQuery(scope={"episode_id": "two"}))[0].item.content == "cup"
    assert m.query(MemoryQuery(scope={"episode_id": "one"}))[0].item.content == "moved"


def test_roundtrip_pose_clock_and_simulation_zero(tmp_path):
    p = tmp_path / "memory.sqlite"
    m = SQLiteMemoryStore(p)
    a = fact(pose=Pose(frame_id="map", position=(0, 0, 0), orientation=(0, 0, 1, 0)),
             event_time=TimeSpan(start=0, end=2, clock="wall"))
    m.write([a])
    digest = m.snapshot()
    m.close()
    m = SQLiteMemoryStore(p)
    got = m.query(MemoryQuery(as_of=0, clock="wall"))[0].item
    assert got.pose == a.pose and got.event_time == a.event_time
    assert m.query(MemoryQuery(clock="sim")) == []
    assert m.snapshot() == digest


@pytest.mark.parametrize("change", [
    {"pose": Pose(frame_id="camera", position=(0, 0, 0))},
    {"pose": Pose(frame_id="map", position=(0, 0, 0), orientation=(0, 0, 1, 0))},
    {"event_time": TimeSpan(start=1)},
    {"event_time": TimeSpan(start=0, clock="wall")},
    {"entities": ("other",)},
])
def test_distinct_evidence_is_not_deduplicated(change):
    m = SQLiteMemoryStore()
    a, b = fact(key=None), fact(key=None, **change)
    assert a.content_hash() != b.content_hash()
    m.write([a, b])
    assert len(m.query(MemoryQuery())) == 2


def test_forget_obeys_scope_entities_and_exact_radius_without_top_k():
    m = SQLiteMemoryStore()
    items = [fact(key=str(i), entities=("target",), pose=Pose(frame_id="map", position=(0.1*i, 0.1*i, 0)))
             for i in range(4)]
    m.write(items + [fact(key="other-episode", entities=("target",), scope={"episode_id": "two"}),
                     fact(key="other-entity", entities=("other",))])
    q = MemoryQuery(scope={"episode_id": "one"}, entities=("target",),
                    near=(Pose(frame_id="map", position=(0, 0, 0)), 0.35), k=1)
    assert m.forget(q) == 3  # Fourth target is in bbox but outside sphere; k is not a deletion cap.
    assert {h.item.key for h in m.query(MemoryQuery())} == {"3", "other-episode", "other-entity"}


def test_backdated_batch_rolls_back_without_reversed_intervals():
    m = SQLiteMemoryStore()
    m.write([fact(event_time=TimeSpan(start=5))])
    digest = m.snapshot()
    with pytest.raises(ValueError, match="backdated"):
        m.write([fact(key="another"), fact(content="late", event_time=TimeSpan(start=2))])
    assert m.snapshot() == digest


def test_snapshot_includes_intervals_not_only_live_boolean():
    m = SQLiteMemoryStore()
    m.write([fact(), fact(content="new", event_time=TimeSpan(start=5))])
    old = m.snapshot()
    m.db.execute("UPDATE items SET valid_to=4 WHERE valid_to IS NOT NULL")
    assert m.snapshot() != old


def test_legacy_schema_migrates_without_losing_rows(tmp_path):
    path = tmp_path / "legacy.sqlite"
    db = sqlite3.connect(path)
    db.executescript(_SCHEMA)
    db.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               ("old", "spatial", "cup", "cup", "{}", "[]", '{"b": "2", "a": "1"}',
                "map", 0, 0, 0, 0, None, 0, None, 10, None, 1, "old", "legacy-hash"))
    db.commit()
    db.close()
    m = SQLiteMemoryStore(path)
    got = m.query(MemoryQuery(as_of=0))[0].item
    assert got.id == "old" and got.event_time.clock == "sim" and got.pose.orientation is None
    duplicate = got.model_copy(update={"id": "retry", "scope": {"a": "1", "b": "2"}})
    assert m.write([duplicate]).events[0][0] == "SKIP_DUP"


def test_literal_search_and_empty_kinds():
    m = SQLiteMemoryStore()
    m.write([fact(content="100% seen"), fact(key="other", content="other")])
    assert len(m.query(MemoryQuery(text="%"))) == 1
    assert m.query(MemoryQuery(kinds=frozenset())) == []


@pytest.mark.parametrize("query", [{"time_range": (2, 1)}, {"as_of": float("nan")},
                                    {"near": (Pose(frame_id="map", position=(0, 0, 0)), -1)}])
def test_invalid_query_ranges_rejected(query):
    with pytest.raises(ValueError):
        MemoryQuery(**query)


@pytest.mark.parametrize("factory", [lambda: TimeSpan(start=float("nan")),
                                      lambda: TimeSpan(start=2, end=1),
                                      lambda: Pose(frame_id="map", position=(0, float("inf"), 0))])
def test_invalid_evidence_rejected_before_storage(factory):
    with pytest.raises(ValueError):
        factory()
