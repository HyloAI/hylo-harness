"""The runner without a model: a worker in its own uv environment, one-shot and persistent, file exchange."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from hylo.tools import TOOLS
from hylo.tools.cli import tool
from hylo.tools.runner import ToolServer, run_worker, worker_path

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")


def test_one_shot(tmp_path):
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    res = run_worker("echo", {"a": 1}, inputs={"image": img}, workdir=tmp_path / "w")
    assert res.outputs["params"] == {"a": 1} and res.outputs["state"] == "loaded"
    assert res.outputs["shape"] == [4, 5, 3]
    assert res.files["flipped"].exists() and res.timing["wall_s"] > 0
    assert res.provenance["device"] == "cpu" and res.provenance["model"] == "echo"


def test_persistent():
    with ToolServer("echo") as srv:
        assert [srv.call({"i": i}).outputs["params"]["i"] for i in range(3)] == [0, 1, 2]


def test_every_tool_has_a_worker_and_cli(capsys):
    for name, mod in TOOLS.items():
        assert worker_path(name).exists(), name
        m = __import__(mod, fromlist=["cli"])
        assert callable(m.cli), name
    assert tool(["list"]) == 0
    out = capsys.readouterr().out
    assert all(n in out for n in TOOLS)
