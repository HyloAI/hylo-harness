import json

import numpy as np
import pytest

from worldcurrent.core.events import Event
from worldcurrent.core.trace import TraceStore
from worldcurrent.reasoners.llm.client import LLMClient
from worldcurrent.reasoners.llm.judge import VLMSuccessJudge


def _episode(tmp_path, with_images=True):
    store = TraceStore(tmp_path / "run")
    img = np.zeros((8, 8, 3), np.uint8)
    for k in range(3):
        payload = {"state": {}, "instruction": "put the bowl on the plate"}
        if with_images and k < 2:                                  # the last observation has no frame
            payload["images"] = {"agentview": img + k, "wrist": img}
        store.write(Event(type="observation", episode_id="e", seq=k, payload=payload))
    store.close_episode("e")
    return store.read("e")


@pytest.mark.parametrize("reply,expected", [
    ({"completed": True, "confidence": 0.9}, 0.9),
    ({"completed": False, "confidence": 0.9}, pytest.approx(0.1)),    # confidence in its own "no" verdict
    ({"completed": False, "confidence": 0.2}, 0.2),
    ({"completed": "true"}, 1.0),
])
def test_vlm_judge_maps_the_reply_to_a_success_probability(tmp_path, monkeypatch, reply, expected):
    seen = {}

    def fake_post(self, path, body, headers):
        seen["images"] = len(body["messages"][-1]["images"])
        seen["text"] = body["messages"][-1]["content"]
        return {"message": {"content": json.dumps(reply)}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    j = VLMSuccessJudge().judge(_episode(tmp_path))
    assert j.p_success == expected
    assert seen["images"] == 2 and "put the bowl on the plate" in seen["text"]


def test_vlm_judge_without_frames_says_so(tmp_path):
    j = VLMSuccessJudge().judge(_episode(tmp_path, with_images=False))
    assert j.p_success == 0.0 and "error" in j.info
