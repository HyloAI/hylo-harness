"""Live smoke test: code.llm with a local Ollama vision model on the toy double. Skipped when Ollama is unreachable
or the model is not pulled (HYLO_OLLAMA_URL / HYLO_OLLAMA_MODEL override the defaults)."""

import json
import os
import urllib.request

import pytest

from hylo.codeagents.agents import CodeLLMAgent
from hylo.codeagents.fakes import ToyManipEnv, toy_benchmark
from hylo.core.kernel import EpisodeRunner
from hylo.core.trace import TraceStore

URL = os.environ.get("HYLO_OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("HYLO_OLLAMA_MODEL", "qwen3.5:35b-a3b")


def _available() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/tags", timeout=2) as r:
            return MODEL in {m["name"] for m in json.loads(r.read())["models"]}
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _available(), reason=f"Ollama at {URL} with {MODEL} is not reachable")
def test_code_llm_with_a_local_model_drives_the_toy_robot(tmp_path):
    model = {"backend": "ollama", "model": MODEL, "base_url": URL + "/v1", "vision": True, "reasoning_effort": "none",
             "max_tokens": 1200, "timeout_s": 300}
    agent = CodeLLMAgent(model=model, tools="hylo.codeagents.fakes:FakeTools", max_turns=4, block_timeout_s=30)
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(ToyManipEnv(), agent, trace=store).run(toy_benchmark().instances("search")[0])
    tr = store.read(rec.outcome.episode_id)
    turns = [e["payload"] for e in tr.of_type("agent.code.turn")]
    assert rec.outcome.status in ("success", "failure") and rec.outcome.termination_reason.startswith("agent_done")
    assert turns and all(t.get("tokens_in") and t.get("latency_s", 0) > 0 for t in turns if "model_error" not in t)
    assert any(t.get("code") for t in turns)                               # the model wrote at least one block
    print(f"\nlive {MODEL}: {rec.outcome.status} in {len(turns)} turns, {rec.outcome.steps} steps, "
          f"{tr.last('agent.code.summary')['payload']['usage']}")
