"""motion_tools: the studio's tool layer with a fake backend (fast: synthetic motions, placeholder renders, the
MuJoCo worker in-process)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_characters as mc  # noqa: E402
from hylo.codeagents.motion_tools import (StudioTools, character_table, contact_sheet, finalize,  # noqa: E402
                                          parse_characters)
from motion_testkit import FakeBackend, motion  # noqa: E402


def test_parse_characters():
    assert parse_characters(None) == [] and sorted(parse_characters("all")) == mc.names()
    assert parse_characters("all")[:5] == list(mc.ROBOTS_DEFAULT)
    assert parse_characters("robots") == list(mc.ROBOTS_DEFAULT)
    assert parse_characters("humans") == mc.names("rig")
    assert parse_characters("g1, human,g1") == ["g1", "human"]
    with pytest.raises(ValueError, match="unknown characters"):
        parse_characters("g1,c3po")


def test_contact_sheet(tmp_path):
    import imageio.v2 as imageio
    with imageio.get_writer(str(tmp_path / "v.mp4"), fps=10, codec="libx264", macro_block_size=8,
                            ffmpeg_log_level="error") as w:
        for i in range(10):
            w.append_data(np.full((90, 160, 3), 20 * i, np.uint8))
    sheet = contact_sheet(tmp_path / "v.mp4", n=4)
    assert sheet.shape[1:] == (640, 3) and sheet.shape[0] in (90, 96)        # (the encoder pads to 16 px)
    assert sheet[50, 600].mean() > sheet[50, 20].mean()
    from PIL import Image
    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "p.png")
    assert contact_sheet(tmp_path / "p.png").shape == (40, 60, 3)


def test_retarget_report_and_render_for_robots_and_rigs(tmp_path):
    be = FakeBackend()
    t = StudioTools(be, tmp_path, "human", {})
    m = motion(seconds=1.0)
    rig = mc.names("rig")[0]
    rm = t.retarget(m, rig)
    assert rm.kind == "rig" and rm.qpos.shape == (30, 17, 3) and t.retarget(m, rig) is rm      # cached
    rep = t.report(m, rig)
    assert rep["physics"].startswith("n/a") and rep["mean_keypoint_err_m"] == 0.0
    assert t.render(m, robot=rig).endswith("video.mp4")
    rh = t.retarget(m, "human")
    assert rh.kind == "mjcf" and "limb_angle_err_deg" in rh.info and "limit_hit_frac" in rh.info
    chk = t.check(m, "human")
    assert chk["problems"] and "kinematics" in chk
    ops = [c[1] for c in be.calls if c[0] == "sim"]
    assert ops.count("rig") == 2 and "retarget" in ops and "check" in ops
    timing = t.timing
    assert {"retarget", "check", "render"} <= set(timing)


def test_lineup_composes_one_render_per_character(tmp_path):
    be = FakeBackend()
    t = StudioTools(be, tmp_path, "human", {}, characters=["human", mc.names("rig")[0]])
    m = motion(seconds=1.0)
    v = t.lineup(m)
    assert Path(v).exists() and t.lineups[-1]["characters"] == ["human", mc.names("rig")[0]]
    for c in ("human", mc.names("rig")[0]):                          # each tile is that character's own render
        rm = t.retarget(m, c)
        assert f"render-{c}" in t.renders[id(rm)]["video"] and t.retarget(rm) is rm
    assert [c for c in be.calls if c[0] == "render"] and ("sim", "compose") in be.calls
    with pytest.raises(ValueError, match="unknown characters"):
        t.lineup(motion(seconds=1.0), ["robocop"])


def test_finalize_with_characters_writes_the_report_table(tmp_path):
    be = FakeBackend()
    chars = ["human", mc.names("rig")[0]]
    t = StudioTools(be, tmp_path, "human", {}, characters=chars)
    t.final = motion(seconds=1.0)
    res = finalize(t, tmp_path, characters=chars)
    table = json.loads((tmp_path / "characters.json").read_text())
    assert set(table) == set(chars) and table["human"]["kind"] == "mjcf"
    assert {"mean_keypoint_err_m", "limb_angle_err_deg", "limit_hit_frac"} <= set(table["human"])
    assert res["files"]["lineup"] == "lineup.mp4" and (tmp_path / "motion.npz").exists()
    assert character_table(t, t.final, ["human"])["human"]["title"] == mc.title("human")
