"""CPU contract checks for the optional GPU pilot; no third-party repositories are imported."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hylo.core.episode_memory import EpisodeScope
from hylo.memory.episode import EventMemory
from hylo.memory.frames import FrameEvidence
from hylo.memory.visual import VisualReportMemory


spec = importlib.util.spec_from_file_location(
    "robomme_memory_pilot", Path(__file__).parents[1] / "examples/memory/robomme/run_paired.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_separate_grounding_and_observation_allowlist(monkeypatch):
    journal = EventMemory()
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="test", episode_id="0"))
    bank = FrameEvidence()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    bank.append(0, frame, demonstration=True)
    args = SimpleNamespace(condition="memory_on", vision_url="http://127.0.0.1:1",
                           frame_evidence=bank, demo_end=1, oracle_subgoal="SECRET")
    requests = []
    replies = iter([
        {"observations": [{"topic": "cue", "claim": "a red cube was visible", "frame_ids": [0], "confidence": 0.5}],
         "subgoal": "pick up the red cube", "points": [{"id": "point_1", "frame_id": 2, "x": 0, "y": 0}]},
        {"points": [{"id": "point_1", "frame_id": 2, "x": 999, "y": 500}]},
    ])
    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.StringIO(json.dumps({"text": "JSON follows:\n```json\n" + json.dumps(next(replies)) + "\n```"}))
    monkeypatch.setattr(pilot.urllib.request, "urlopen", respond)
    command, record = pilot.plan(args, "pick cube", [frame], [frame], [2], memory)
    assert command == "pick up the red cube at <8, 15>"
    assert len(requests[0]["images"]) == 4
    assert len(requests[1]["images"]) == 1 and requests[1]["mode"] == "ground"
    assert "SECRET" not in json.dumps(requests)
    assert record["request"]["frame_ids"] == [0, 2]
    assert memory.context("cue")["reports"][0]["confidence"] == 0.5
    journal.close()


@pytest.mark.parametrize("bad", [
    {"observations": [{"topic": "cue", "claim": "future", "frame_ids": [5], "confidence": 1}],
     "subgoal": "remain static", "points": []},
    {}, {"subgoal": 7, "points": []}, {"subgoal": "pick cube", "points": [{}]},
])
def test_invalid_plans_fail_without_writing_memory(monkeypatch, bad):
    journal = EventMemory()
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="test", episode_id="0"))
    before = journal.snapshot()
    calls = []
    def respond(request, **kwargs):
        calls.append(request)
        return io.StringIO(json.dumps({"text": json.dumps(bad)}))
    monkeypatch.setattr(pilot.urllib.request, "urlopen", respond)
    args = SimpleNamespace(condition="memory_on", vision_url="http://127.0.0.1:1")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="failed validation"):
        pilot.plan(args, "pick cube", [frame], [frame], [0], memory)
    assert len(calls) == 3 and journal.snapshot() == before
    journal.close()


def test_composite_proposal_compiles_only_first_stage():
    data = {"observations": [], "subgoal": "pick up the cube and place it on the target", "points": [
        {"id": "point_1", "frame_id": 0, "x": 400, "y": 400},
        {"id": "point_2", "frame_id": 0, "x": 500, "y": 500},
    ]}
    normalized = pilot.first_stage(data, 0)
    assert normalized["subgoal"] == "pick up the cube at <point_1>"
    assert len(normalized["points"]) == 1
    assert len(data["points"]) == 2


def test_native_qwen_points_keep_coordinate_checks_and_adapter_frame_binding():
    data = pilot.json_object('```json\n[{"point_2d": [306, 333], "label": "red cube"}]\n```')
    result = pilot.grounding_points(data, ["point_1"], 64)
    assert (result[0].x, result[0].y, result[0].frame_id) == (306, 333, 64)
    with pytest.raises(ValueError):
        pilot.grounding_points([{"point_2d": [-1, 333]}], ["point_1"], 64)
    with pytest.raises(ValueError):
        pilot.grounding_points(data * 2, ["point_1"], 64)


def test_rgb_marks_bind_only_valid_regions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/memory/robomme"))
    from grounding import propose
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[20:30, 25:35] = [255, 0, 0]
    frame[40:50, 40:50] = [255, 0, 0]
    candidates, _ = propose(frame)
    red = next(c for c in candidates if c["color"] == "red")
    selected = [red["id"]]
    def respond(request, **kwargs):
        payload = json.loads(request.data)
        assert payload["mode"] == "marks" and len(payload["images"]) == 2
        assert "depth" not in payload and "segmentation" not in payload
        return io.StringIO(json.dumps({"text": json.dumps({"choices": {"point_1": selected[0]}})}))
    monkeypatch.setattr(pilot.urllib.request, "urlopen", respond)
    args = SimpleNamespace(vision_url="http://127.0.0.1:1", rgb_regions=True)
    point = pilot.VisualPoint(id="point_1", frame_id=0, x=0, y=0)
    points, trace = pilot.ground(args, "pick up red cube at <point_1>", [point], frame, 0)
    assert points[0].x == round(29.5 * 999 / 63)
    assert points[0].y == round(24.5 * 999 / 63)
    selected[0] = 999
    points, trace = pilot.ground(args, "pick up red cube at <point_1>", [point], frame, 0)
    assert points is None and "abstention" in trace


def test_occluded_target_continuation_expires(monkeypatch):
    journal = EventMemory()
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id="test", episode_id="0"))
    args = SimpleNamespace(condition="memory_off", vision_url="http://127.0.0.1:1",
                           last_semantic="pick up the red cube at <target>",
                           last_grounded_command="pick up the red cube at <5, 8>", last_grounded_frame=0)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    current = [32]
    def respond(request, **kwargs):
        payload = json.loads(request.data)
        if payload.get("mode") == "ground":
            data = {"points": []}
        else:
            data = {"observations": [], "subgoal": "pick up the red cube at <point_1>",
                    "points": [{"id": "point_1", "frame_id": current[0], "x": 0, "y": 0}]}
        return io.StringIO(json.dumps({"text": json.dumps(data)}))
    monkeypatch.setattr(pilot.urllib.request, "urlopen", respond)
    command, record = pilot.plan(args, "pick cube", [frame], [frame], current, memory)
    assert command == args.last_grounded_command
    assert record["grounding"]["evidence_frame"] == 0
    current[0] = 129
    command, record = pilot.plan(args, "pick cube", [frame], [frame], current, memory)
    assert command == "remain static"
    journal.close()


def test_appearance_filter_does_not_treat_hidden_contents_as_visible_color(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/memory/robomme"))
    from grounding import filter_targets
    candidates = [{"color": "red", "white_fraction": 0, "id": 1},
                  {"color": "purple", "white_fraction": 0.3, "id": 2},
                  {"color": "white", "white_fraction": 1, "id": 3}]
    selected, rule = filter_targets(candidates, "container hiding the red cube", "object to contact", (256, 256, 3))
    assert selected == candidates and rule == "unfiltered"
    selected, rule = filter_targets(candidates, "the target", "placement DESTINATION", (256, 256, 3))
    assert [c["id"] for c in selected] == [2]
