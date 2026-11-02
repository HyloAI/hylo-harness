"""Smoke tests: every example in examples/agents runs with --quick and writes its result.json."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))
NEEDS = {"02_reasoner_swap.py": ("hylo.reasoners.llm", "hylo.reasoners.rules")}


def _missing(*modules: str) -> list[str]:
    out = []
    for m in modules:
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except ModuleNotFoundError:
            out.append(m)
    return out


def test_every_example_is_listed():
    assert len(SCRIPTS) >= 3


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_quick(script: str, tmp_path: Path):
    missing = _missing("mujoco", "hylo.envs.mujoco", "hylo.policies.reference", "hylo.learn", *NEEDS.get(script, ()))
    if missing:
        pytest.skip(f"optional packages missing: {missing}")
    # the reasoner example skips its LLM / VLM rows by itself when Ollama is unreachable
    proc = subprocess.run([sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path)],
                          capture_output=True, text=True, timeout=600, cwd=EXAMPLES.parents[1])
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["quick"] is True
