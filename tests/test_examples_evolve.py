"""Smoke test: every evolution example (examples/evolve/) runs end to end in --quick mode and writes its result.json."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "evolve"
OLLAMA = "http://localhost:11434"
LLM_MODEL = "qwen3.5:35b-a3b"

# example -> optional modules it needs (beyond hylo-core / hylo-eval / hylo-evolve)
NEEDS = {
    "01_campaign_reach.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
    "02_protected_scope.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
    "03_gate_power.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
    "04_field_rollback.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
    "05_llm_vs_rule.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference", "hylo.proposers.llm"],
}


def _missing(modules: list[str]) -> list[str]:
    out = []
    for m in modules:
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except ModuleNotFoundError:
            out.append(m)
    return out


def _ollama_model() -> str | None:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:  # noqa: BLE001
        return f"Ollama not reachable at {OLLAMA}"
    return None if LLM_MODEL in names else f"{LLM_MODEL} not pulled on {OLLAMA}"


def test_every_example_is_covered():
    assert sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py")) == sorted(NEEDS)


@pytest.mark.parametrize("script", sorted(NEEDS))
def test_example_runs_quick(script: str, tmp_path: Path):
    missing = _missing(NEEDS[script])
    if missing:
        pytest.skip(f"{script} needs {', '.join(missing)}")
    if script == "05_llm_vs_rule.py":
        why = _ollama_model()
        if why:
            pytest.skip(why)
    args = [sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path), "--workers", "8"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=EXAMPLES.parents[1])
    assert p.returncode == 0, f"{script} exited {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["quick"] is True
    if script == "02_protected_scope.py":
        assert result["refused"] >= 10 and all(c["episodes_spent"] == 0 for c in result["candidates"]
                                               if c["verdict"] == "refused at stage 0")
    if script == "03_gate_power.py":
        assert result["paired_test"]["power"]["A/A"]["40"] <= 0.1
    if script == "05_llm_vs_rule.py":
        assert "llm.steering" in result["arms"]
