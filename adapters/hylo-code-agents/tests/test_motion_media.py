"""motion_media: video windows, labelled grids (fast, no rendering) and rigged characters driven in Blender
(solve only, a few seconds; skipped when Blender or the character archives are not available)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_characters as mc  # noqa: E402
from hylo.codeagents import motion_ops as ops  # noqa: E402
from motion_testkit import media, skeleton, stepping  # noqa: E402


def _video(path: Path, colour, n: int = 6, size=(64, 96)) -> str:
    import imageio.v2 as imageio
    with imageio.get_writer(str(path), fps=10, codec="libx264", macro_block_size=8, ffmpeg_log_level="error") as w:
        for _ in range(n):
            w.append_data(np.full((*size, 3), colour, np.uint8))
    return str(path)


def test_compose_grid_with_labels(tmp_path):
    import imageio.v2 as imageio
    a = _video(tmp_path / "a.mp4", (200, 30, 30), 6)
    b = _video(tmp_path / "b.mp4", (30, 200, 30), 3)
    np.save(tmp_path / "c.npy", np.full((4, 48, 64, 3), 90, np.uint8))
    r = media.handle("compose", {"videos": [a, b, str(tmp_path / "c.npy")], "labels": ["a", "b", "c"], "cols": 2,
                                 "height": 64, "fps": 10}, {}, tmp_path / "out")
    assert r["outputs"] == {"frames": 6, "fps": 10, "tiles": 3, "cols": 2}
    frames = [f for f in imageio.get_reader(r["files"]["video"])]      # (its length hint is inf)
    assert len(frames) == 6 and frames[0].shape[0] == 128              # two rows of 64 px
    f = frames[-1]
    assert f[40, 48, 0] > 150 and f[40, 48 + f.shape[1] // 2, 1] > 150   # shorter clips hold their last frame


def test_clip_of_an_image(tmp_path):
    from PIL import Image
    Image.fromarray(np.zeros((900, 1200, 3), np.uint8)).save(tmp_path / "p.png")
    r = media.handle("clip", {}, {"media": str(tmp_path / "p.png")}, tmp_path / "out")
    assert np.load(r["files"]["frames"]).shape == (3, 480, 640, 3)


def test_unknown_op(tmp_path):
    with pytest.raises(ValueError, match="unknown op"):
        media.handle("export", {}, {}, tmp_path)


# ------------------------------------------------------------------------------------------------- rigs (Blender)
def _blender_or_skip():
    from hylo.codeagents.motion_tools import render_tool_dir
    exe, _env = media.render_worker(str(render_tool_dir())).blender_exe()
    if exe is None:
        pytest.skip("Blender is not installed")
    return str(render_tool_dir())


def _solve(name, K, out, rig=None):
    tool = _blender_or_skip()
    try:
        rig = rig or mc.rig_spec(name)
    except RuntimeError as exc:                       # offline and not cached
        pytest.skip(str(exc))
    np.save(out / "kp.npy", K)
    return media.handle("rig", {"rig": rig, "keypoints": str(out / "kp.npy"), "names": ops.NAMES, "fps": 30.0,
                                "render": False, "render_tool": tool}, {}, out)


@pytest.mark.parametrize("name", mc.names("rig"))
def test_rig_follows_synthetic_motions_with_feet_on_the_ground(name, tmp_path):
    for label, K in (("stand_wave", skeleton(1.0)), ("step", stepping(1.5))):
        out = tmp_path / label
        out.mkdir()
        r = _solve(name, ops.canonical(ops.Motion(K, 30.0)).joints, out)
        m = r["outputs"]["metrics"]
        J = np.load(r["files"]["joints"])
        assert J.shape == (len(K), 17, 3)
        assert m["mean_keypoint_err_m"] < 0.02 and m["limb_angle_err_deg"] < 2
        assert m["ground_penetration_max_m"] < 0.01 and m["limit_hit_frac"] == 0
        if label == "stand_wave":
            assert m["frames_floating_5cm"] == 0 and m["foot_skate_mean_speed_mps"] < 0.02
        assert abs(J[:, ops.IDX["Head"], 2].max() - J[:, ops.IDX["Hips"], 2].max()) > 0.2   # upright


def test_rig_mapping_with_a_missing_bone_fails_clearly(tmp_path):
    try:
        rig = copy.deepcopy(mc.rig_spec("kenney_survivor"))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    rig["drive"]["neck"] = "NoSuchBone"
    with pytest.raises(RuntimeError, match="NoSuchBone"):
        _solve("kenney_survivor", skeleton(0.2), tmp_path, rig)
