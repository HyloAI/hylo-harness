"""Free mode (motion_free) with a scripted model and the fake backend: scripts in the run directory, the kit's
requests, refusals, the tool-call budget, the final report (fast: no model, no rendering)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_characters as mc  # noqa: E402
from hylo.codeagents.motion_free import run_free  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402
from motion_testkit import FakeBackend, block  # noqa: E402


def _log(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]


def test_free_mode_with_a_scripted_model(tmp_path):
    rig = mc.names("rig")[0]
    replies = [
        block("from studio import *\nfrom scipy.ndimage import uniform_filter1d\n"
              "m = text_motion('a person waves', 2.0)\nlong = loop(concat(m, mirror(m)), 2)\n"
              "J = uniform_filter1d(long.joints, 5, axis=0)\nsmooth = motion_from_array(J, long.fps, 'smoothed')\n"
              "save_motion(smooth, 'work/routine.npz')\nprint(round(smooth.seconds, 1))\n"
              "print(retarget(smooth, 'h1')['scaling'])\nprint(hands('inputs/clip.mp4')['files'])\n"
              "open('lib/helper.py', 'w').write('def twice(x):\\n    return 2 * x\\n')"),
        block("from studio import *\nimport helper\nprint(helper.twice(21))\n"
              "for f in (lambda: open('/etc/hostname').read(), lambda: video_motion('/etc/hosts'),\n"
              "          lambda: tool('mesh', image='x.png'), lambda: render('work/routine.npz', 'c3po')):\n"
              "    try:\n        f()\n    except (PermissionError, ToolError, ValueError) as e:\n"
              "        print('refused:', type(e).__name__)"),
        block("import os\nos.system('touch HACK')"),
        block(f"from studio import *\nm = load_motion('work/routine.npz')\nprint(check(m, 'human')['problems'][:1])\n"
              f"finish(m, ['human', '{rig}'], 'a looped, mirrored, smoothed wave')"),
    ]
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not decoded by the fake")
    s = run_free("make a longer waving routine", tmp_path / "run", video=str(clip), model=ScriptedChat(replies),
                 backend=FakeBackend(), budget="turns=6")
    run = tmp_path / "run"
    assert s["stopped"] == "finished" and s["turns"] == 4 and s["scripts"] == 4
    assert s["characters"] == ["human", rig] and s["files"]["lineup"] == "lineup.mp4"
    assert (run / "characters.json").exists() and (run / "scripts" / "turn03.py").exists()
    assert not (run / "HACK").exists() and (run / "inputs" / "clip.mp4").exists()
    log = _log(run)
    turns = [r for r in log if "turn" in r]
    assert "42" in turns[1]["output"] and turns[1]["output"].count("refused:") == 4
    assert turns[2]["error"].startswith("rejected")
    req = [r for r in log if r.get("event") == "request"]
    assert {"text_motion", "retarget", "tool", "check", "finish"} <= {r["op"] for r in req}
    assert any("outside the run directory" in (r.get("refused") or "") for r in req)
    assert any("'mesh' is not allowed" in (r.get("refused") or "") for r in req)
    scripts = [r for r in log if r.get("event") == "script"]
    assert len(scripts) == 4 and scripts[2].get("rejected")
    z = np.load(run / "motion.npz")
    assert z["joints"].shape[1:] == (17, 3)


def test_tool_call_budget_and_unfinished_session(tmp_path):
    replies = [block("from studio import *\nfor i in range(3):\n    try:\n        text_motion('a person waves', 1.0)\n"
                     "        print('ok')\n    except ToolError as e:\n        print(e)")]
    s = run_free("wave", tmp_path, model=ScriptedChat(replies), backend=FakeBackend(), budget="turns=1",
                 max_tool_calls=2)
    assert s["final"] is None and s["tool_calls"] == 2 and s["refused_requests"] == 1
    out = [r for r in _log(tmp_path) if "turn" in r][0]["output"]
    assert out.count("ok") == 2 and "budget (2) is used up" in out
