import json

import pytest

from hylo.core.episode_memory import EpisodeScope, MemoryEvent, RecallQuery
from hylo.memory.episode import EventMemory
from hylo.memory.visual import VisualClaim, VisualPlan, VisualPoint, VisualReportMemory, render_subgoal


def plan(topic, claim, frame, confidence=0.6):
    return VisualPlan(subgoal="Lift the red cube.", observations=(VisualClaim(
        topic=topic, claim=claim, frame_ids=(frame,), confidence=confidence,
    ),))


def test_visual_reports_preserve_uncertainty_and_both_locations(tmp_path):
    journal = EventMemory(str(tmp_path / "reports.sqlite"))
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="hidden", episode_id="0"))
    memory.write(plan("red cube", "left bin", 0), supplied_frame_ids=[0], step=0)
    memory.write(plan("red cube", "right bin", 4), supplied_frame_ids=[4], step=4)
    context = memory.context("red cube")
    assert context["status"] == "unverified_visual_reports"
    assert [r["claim"] for r in context["reports"]] == ["left bin", "right bin"]
    assert all(r["confidence"] == 0.6 for r in context["reports"])
    journal.close()


def test_bad_citations_reject_whole_response():
    journal = EventMemory()
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="test", episode_id="0"))
    before = journal.snapshot()
    response = VisualPlan(subgoal="Lift cube", observations=(
        plan("cube", "left", 0).observations[0], plan("cube", "right", 999).observations[0],
    ))
    with pytest.raises(ValueError, match="absent"):
        memory.write(response, supplied_frame_ids=[0], step=0)
    assert journal.snapshot() == before
    with pytest.raises(ValueError, match="causal"):
        memory.write(response, supplied_frame_ids=[999], step=0)
    journal.close()


def test_budget_and_episode_isolation(tmp_path):
    journal = EventMemory(str(tmp_path / "memory.sqlite"))
    memory = VisualReportMemory(journal, max_chars=700)
    memory.reset(EpisodeScope(task_id="a", episode_id="0"))
    for i in range(20):
        memory.write(plan(str(i), "observed cue " * 10, i), supplied_frame_ids=[i], step=i)
    context = memory.context("cue")
    assert len(json.dumps(context, ensure_ascii=False)) <= 700
    assert context["omitted_reports"] > 0
    memory.reset(EpisodeScope(task_id="a", episode_id="1"))
    assert memory.context("cue")["reports"] == []
    journal.close()


def test_grounding_coordinates_and_stale_frame_rejection():
    response = VisualPlan(subgoal="Pick up the cube at <point_1>.", points=(
        VisualPoint(id="point_1", frame_id=4, x=999, y=0),
    ))
    assert render_subgoal(response, current_frame_id=4, width=320, height=240) == "Pick up the cube at <0, 319>."
    with pytest.raises(ValueError, match="current"):
        render_subgoal(response, current_frame_id=5, width=320, height=240)
    with pytest.raises(ValueError, match="unresolved"):
        render_subgoal(VisualPlan(subgoal="Pick <point_1>"), current_frame_id=4, width=320, height=240)


def test_sensor_event_and_visual_report_share_one_observation_step():
    journal = EventMemory()
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="test", episode_id="0"))
    journal.append([MemoryEvent(id="cycle:0", step=5, time=5, kind="motion.pick_place",
                                subject="red", evidence=("frame:5",), source="sensor")])
    memory.write(plan("red cube", "cube is on the target", 5), supplied_frame_ids=[5], step=5)
    assert journal.recall(RecallQuery(name="cycles", op="count", kind="motion.pick_place")).value == 1
    assert len(memory.context("red cube")["reports"]) == 1
    journal.close()
