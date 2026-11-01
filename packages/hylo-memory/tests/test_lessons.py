import pytest

from hylo.memory.action_bank import FrozenMemoryError
from hylo.memory.lessons import LessonBook

EV = [{"run_id": "r", "episode_id": "e1", "split": "search"}]


def test_delta_ops_dedup_votes_and_retirement(tmp_path):
    book = LessonBook(str(tmp_path), retire_margin=2)
    [add] = book.apply([{"op": "ADD", "scope": "global", "kind": "pitfall", "title": "Regrasping the same spot fails",
                         "content": "After two failed grasps at one spot, back off.", "evidence": EV}], source="reflect:a")
    lid = add["id"]
    dup = book.apply([{"op": "ADD", "scope": "global", "kind": "pitfall", "title": "Regrasping the same spot fails.",
                       "content": "x"}])
    assert dup[0]["op"] == "UPVOTE" and dup[0]["id"] == lid and book.all()[0].helpful == 1
    other_scope = book.apply([{"op": "ADD", "scope": "task:t/a", "kind": "pitfall", "title": "Regrasping the same spot fails",
                               "content": "y"}])
    assert other_scope[0]["op"] == "ADD"                           # dedup is per scope
    bad = book.apply([{"op": "UPVOTE", "id": "nope"}, {"op": "SHOUT"}])
    assert all("error" in d for d in bad)
    book.apply([{"op": "EDIT", "id": lid, "content": "Back off 5 cm after two failed grasps."}])
    assert book.query(("global",))[0].content.startswith("Back off") and book.query(("global",))[0].helpful == 1
    out = book.apply([{"op": "DOWNVOTE", "id": lid}] * 3)
    assert out[-1].get("retired") and book.query(("global",)) == []
    assert "reflect:a" in book.sources()


def test_ranking_persistence_and_freeze(tmp_path):
    book = LessonBook(str(tmp_path))
    ids = [book.apply([{"op": "ADD", "scope": "global", "kind": "strategy", "title": t, "content": t}])[0]["id"]
           for t in ("approach from above the handle", "close the gripper only when centred")]
    book.apply([{"op": "UPVOTE", "id": ids[1]}])
    assert [l.id for l in book.query(("global",))] == [ids[1], ids[0]]
    frozen = LessonBook(str(tmp_path), frozen=True)
    assert frozen.snapshot() == book.snapshot() and len(frozen) == 2
    with pytest.raises(FrozenMemoryError):
        frozen.apply([{"op": "UPVOTE", "id": ids[0]}])
