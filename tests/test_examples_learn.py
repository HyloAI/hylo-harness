"""Smoke test: every learning example (examples/learn/) runs end to end in --quick mode and writes its result.json."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "learn"

# example -> optional modules it needs (beyond hylo-core / hylo-eval / hylo-learn)
NEEDS = {
    "01_learner_job.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
    "02_judge_calibration.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference", "hylo.instruments"],
    "03_online_cem.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.policies.reference"],
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


def test_every_example_is_covered():
    assert sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py")) == sorted(NEEDS)


@pytest.mark.parametrize("script", sorted(NEEDS))
def test_example_runs_quick(script: str, tmp_path: Path):
    missing = _missing(NEEDS[script])
    if missing:
        pytest.skip(f"{script} needs {', '.join(missing)}")
    args = [sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path), "--workers", "8"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=EXAMPLES.parents[1])
    assert p.returncode == 0, f"{script} exited {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["quick"] is True
    if script == "01_learner_job.py":
        assert result["checks"]["tampered"].startswith("refused") and result["checks"]["gate job"].startswith("refused")
