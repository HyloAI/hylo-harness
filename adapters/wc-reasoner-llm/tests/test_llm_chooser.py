import json
import os

import pytest

from worldcurrent.agents.object_reasoner import SceneObject
from worldcurrent.reasoners.llm.chooser import LLMChooser
from worldcurrent.reasoners.llm.client import LLMClient, LLMError, _parse_json

OBJS = [SceneObject(id="obj_0", rgb=(0.85, 0.1, 0.1), position=(0.3, 0.1), source="seen"),
        SceneObject(id="obj_1", rgb=(0.15, 0.35, 0.9), position=(-0.2, 0.3), source="remembered")]


def fake_ollama(reply: dict, calls: list):
    def _post(self, path, body, headers):
        calls.append(body)
        return {"message": {"content": json.dumps(reply)}, "prompt_eval_count": 10, "eval_count": 5}
    return _post


def test_chooser_returns_a_valid_object(monkeypatch):
    calls: list = []
    monkeypatch.setattr(LLMClient, "_post", fake_ollama({"object_id": "obj_1", "rationale": "sky is blue"}, calls))
    c = LLMChooser().choose("touch the pad the colour of the sky", OBJS, (0.0, 0.0))
    assert c.object_id == "obj_1" and c.rationale == "sky is blue"
    body = calls[0]
    assert body["format"] == "json" and body["think"] is False
    assert "#2659e6" in body["messages"][-1]["content"]  # colours are given to the model as hex too


def test_unknown_id_is_rejected(monkeypatch):
    monkeypatch.setattr(LLMClient, "_post", fake_ollama({"object_id": "obj_9", "rationale": "?"}, []))
    assert LLMChooser().choose("x", OBJS, (0, 0)).object_id is None


def test_endpoint_failure_degrades_to_no_choice(monkeypatch):
    def boom(self, path, body, headers):
        raise LLMError("connection refused")
    monkeypatch.setattr(LLMClient, "_post", boom)
    c = LLMChooser().choose("x", OBJS, (0, 0))
    assert c.object_id is None and "unavailable" in c.rationale


def test_responses_are_cached_by_prompt(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(LLMClient, "_post", fake_ollama({"object_id": "obj_0", "rationale": "red"}, calls))
    chooser = LLMChooser(cache_dir=str(tmp_path))
    first = chooser.choose("touch red", OBJS, (0, 0))
    second = chooser.choose("touch red", OBJS, (0, 0))
    assert len(calls) == 1 and first.meta["cached"] is False and second.meta["cached"] is True


def test_json_is_extracted_from_chatty_output():
    assert _parse_json('Sure! {"object_id": "obj_0"} hope that helps') == {"object_id": "obj_0"}


@pytest.mark.skipif(os.environ.get("WC_TEST_LLM") != "1", reason="set WC_TEST_LLM=1 to call the local Ollama server")
def test_live_ollama_grounds_an_indirect_description():
    c = LLMChooser(timeout_s=120).choose("touch the pad that is the colour of a ripe tomato", OBJS, (0, 0))
    assert c.object_id == "obj_0"


# --- VLM (set-of-mark) chooser --------------------------------------------------------------------------
import numpy as np  # noqa: E402

from worldcurrent.core.types import Observation  # noqa: E402
from worldcurrent.reasoners.llm.vlm_chooser import VLMChooser, project  # noqa: E402

CAM = {"pos": [0.0, 0.0, 1.45], "xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1], "fovy_deg": 50.0, "width": 384, "height": 384}


def test_projection_of_a_top_down_camera():
    uv = project(np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]]), CAM)
    assert np.allclose(uv[0], [192, 192])
    assert uv[1][0] > 192 and np.isclose(uv[1][1], 192)      # +x is right
    assert uv[2][1] < 192 and np.isclose(uv[2][0], 192)      # +y is up


def test_vlm_sees_only_the_marked_image(monkeypatch):
    calls: list = []
    monkeypatch.setattr(LLMClient, "_post", fake_ollama({"marker": 1, "rationale": "the blue pad"}, calls))
    obs = Observation(state={}, images={"top": np.zeros((384, 384, 3), np.uint8)}, extra={"camera_params": {"top": CAM}})
    c = VLMChooser().choose("touch the pad the colour of the sky", OBJS, (0, 0), obs=obs)
    msg = calls[0]["messages"][-1]
    assert c.object_id == "obj_1" and len(msg["images"]) == 1
    assert "rgb" not in msg["content"] and "#" not in msg["content"]   # no colour values in the prompt
    assert c.meta["marked_image"].shape == (384, 384, 3)


def test_vlm_without_a_camera_image_degrades():
    assert VLMChooser().choose("x", OBJS, (0, 0), obs=Observation(state={})).object_id is None
