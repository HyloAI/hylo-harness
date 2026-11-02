"""Smoke test: every core example (examples/core/) runs end to end in --quick mode and writes its result.json."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "core"
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu" / "libOSMesa.so.8"

# example -> optional modules it needs (beyond hylo-core / hylo-eval)
NEEDS = {
    "01_first_episode.py": ["hylo.envs.mujoco", "hylo.agents"],
    "02_safety_shield.py": ["hylo.envs.mujoco", "hylo.agents"],
    "03_trace_replay.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.reasoners.rules", "hylo.memory"],
    "04_remote_env.py": ["hylo.envs.mujoco", "hylo.agents", "websockets", "msgpack"],
    "05_conformance_artifacts.py": ["hylo.envs.mujoco", "hylo.agents", "hylo.cli"],
    "06_realtime_timing.py": ["hylo.envs.mujoco"],
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
    args = [sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path)]
    if script == "03_trace_replay.py" and not (os.environ.get("MUJOCO_GL") or USER_MESA.exists()):
        args.append("--no-render")                                  # no OpenGL backend for MuJoCo here
    p = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=EXAMPLES.parents[1])
    assert p.returncode == 0, f"{script} exited {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["quick"] is True
