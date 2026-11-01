import json

import pytest

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.evolve.steering import diagnose_traces
from worldcurrent.memory.lessons import LessonBook
from worldcurrent.reasoners.llm.client import LLMClient
from worldcurrent.proposers.llm.reflector import LLMReflector


def _diag(tmp_path, split):
    cfg = ExperimentConfig(name="m", benchmark={"type": "testing/reach-v0", "params": {"protocol": "rsi"}},
                           env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.3}})
    return diagnose_traces(run_experiment(cfg, split=split, run_root=tmp_path))


def test_reflector_ops_must_cite_shown_episodes_and_stay_in_scope(tmp_path, monkeypatch):
    diag = _diag(tmp_path, "search")
    ev_ids = []

    def fake_post(self, path, body, headers):
        user = json.loads(body["messages"][-1]["content"])
        ev_ids.extend(e["id"] for t in user["evidence"]["tasks"] for e in t["episodes"])
        task = user["evidence"]["tasks"][0]["task_id"]
        return {"message": {"content": json.dumps({"ops": [
            {"op": "ADD", "scope": f"task:{task}", "kind": "pitfall", "title": "slow approach times out",
             "content": "Low gain never reaches the goal in time.", "evidence": [ev_ids[0]]},
            {"op": "ADD", "scope": "global", "kind": "strategy", "title": "uncited", "content": "x", "evidence": []},
            {"op": "ADD", "scope": "task:unknown/task", "kind": "pitfall", "title": "t", "content": "c", "evidence": [ev_ids[0]]},
            {"op": "UPVOTE", "id": "L-does-not-exist", "evidence": [ev_ids[0]]}]})}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    book = LessonBook(str(tmp_path / "lessons"))
    refl = LLMReflector()
    ops = refl.reflect(diag, book)
    assert [o["op"] for o in ops] == ["ADD"] and ops[0]["evidence"][0]["split"] == "search"
    assert len(refl.rejected) == 3
    assert book.apply(ops)[0]["op"] == "ADD"


def test_reflector_refuses_selection_splits(tmp_path):
    with pytest.raises(ValueError, match="search-split"):
        LLMReflector().evidence(_diag(tmp_path, "search_val"))
