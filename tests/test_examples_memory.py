"""Smoke test: every memory example (examples/memory) runs with --quick and writes its result.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "memory"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))
NEEDS = {"envs": "mujoco.planar_reach", "benchmarks": "mujoco/table-v0"}   # hylo-bench's MuJoCo adapter


def _missing() -> str | None:
    """The examples reference the MuJoCo envs by registry name only; skip when that plugin package is absent."""
    from hylo.core.registry import registry

    for group, name in NEEDS.items():
        try:
            registry.resolve(group, name)
        except Exception as exc:                       # noqa: BLE001
            return f"{group} plugin {name!r} not available: {exc}"
    for mod in ("hylo.policies.reference", "hylo.learn", "hylo.evolve", "hylo.reasoners.rules"):
        try:
            __import__(mod)
        except ImportError as exc:
            return f"{mod} not importable: {exc}"
    return None


def test_examples_are_present():
    assert len(SCRIPTS) >= 4


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_quick(script: str, tmp_path: Path):
    reason = _missing()
    if reason:
        pytest.skip(reason)
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    proc = subprocess.run([sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path), "--workers", "8"],
                          capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["example"] == script[3:-3]
