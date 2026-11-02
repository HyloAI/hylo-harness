"""motion_characters (mapping files, validation, assets) and retargeting onto every MJCF character (motion_sim).
The robot tests need the Menagerie folders (fetched on first use, then cached); offline and uncached they skip."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_characters as mc  # noqa: E402
from hylo.codeagents import motion_ops as ops  # noqa: E402
from motion_testkit import ms, skeleton, stepping  # noqa: E402

ROBOTS = ["g1", "h1", "t1", "apollo", "n1"]


# ------------------------------------------------------------------------------------------------- mappings
def test_every_mapping_file_is_valid_and_credited():
    names = mc.names()
    assert set(ROBOTS + ["human"]) <= set(mc.names("mjcf")) and len(mc.names("rig")) >= 2
    for n in names:
        c = mc.load(n)
        assert c["name"] == n and c["licence"] and c["credit"]
    cr = mc.credits(names)
    assert all(v["licence"] for v in cr.values())
    assert cr["h1"]["commit"] == "b0655057a9df2abe39ea7fc989914ba6f248301c" and cr["h1"]["files_verified"] > 10
    assert all(cr[r]["sha256"] and cr[r]["url"].startswith("https://") for r in mc.names("rig"))


def test_validation_names_each_problem():
    good = mc.load("h1")
    assert mc.validate(good) == []
    bad = copy.deepcopy(good)
    del bad["bodies"]["left_knee"]
    bad["bodies"]["head"] = ["torso_link", [0, 0]]
    bad["bodies"]["tail"] = "x"
    bad["feet"] = ["only_one"]
    bad["clearance"] = {"wrist": 3}
    msg = " | ".join(mc.validate(bad))
    for part in ("left_knee", "head", "unknown roles", "feet", "clearance"):
        assert part in msg
    rig = copy.deepcopy(mc.load("kenney_survivor"))
    assert mc.validate(rig) == []
    rig["joints"]["pelvis"] = "Hips"                                 # no .head / .tail
    rig["drive"]["spine"] = []
    rig["drive"]["neck"] = rig["drive"]["pelvis"]                    # a bone driven twice
    rig["asset"]["archive"]["sha256"] = "abc"
    msg = " | ".join(mc.validate(rig))
    for part in ("joints", "spine", "driven twice", "sha256"):
        assert part in msg
    assert mc.validate({"kind": "robot"})[-1].startswith("kind must be")
    with pytest.raises(ValueError, match="unknown character"):
        mc.load("no_such_character")


def test_archive_assets_verify_the_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("HYLO_ASSETS", str(tmp_path / "cache"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("pack/model.fbx", b"model")
    data = buf.getvalue()
    (tmp_path / "pack.zip").write_bytes(data)
    asset = {"archive": {"url": (tmp_path / "pack.zip").as_uri(), "sha256": hashlib.sha256(data).hexdigest(),
                         "members": {"model.fbx": "pack/model.fbx"}}, "file": "model.fbx"}
    d = mc.archive_assets(asset)
    assert (d / "model.fbx").read_bytes() == b"model" and not (d / "archive.zip").exists()
    assert mc.archive_assets(asset) == d                             # cached
    asset["archive"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest mismatch"):
        mc.archive_assets(asset)


def test_menagerie_folder_is_verified_file_by_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HYLO_ASSETS", str(tmp_path))
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"repo": "x/y", "commit": "c" * 40, "folder": "bot",
                               "sha256": {"a.xml": hashlib.sha256(b"<mujoco/>").hexdigest()}}))
    d = mc.menagerie_dir(man)
    d.mkdir(parents=True)
    (d / "a.xml").write_bytes(b"<mujoco/>")
    assert mc.menagerie_assets(man) == d and (d / ".hylo-verified").exists()
    (d / ".hylo-verified").unlink()
    (d / "a.xml").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differ"):
        mc.menagerie_assets(man)


# ------------------------------------------------------------------------------------------------- MJCF characters
@pytest.fixture(scope="module")
def specs(tmp_path_factory):
    out = {}
    for n in ROBOTS + ["human"]:
        try:
            out[n] = mc.mjcf_spec(n, tmp_path_factory.mktemp(n))
        except RuntimeError as exc:                   # offline and not cached
            pytest.skip(str(exc))
    return out


@pytest.mark.parametrize("name", ROBOTS + ["human"])
def test_mapping_matches_the_model(specs, name):
    r = ms.validate(specs[name])["outputs"]
    assert r["problems"] == []
    assert r["height_m"] == pytest.approx(mc.load(name)["height_m"], abs=0.05)


def test_validation_catches_swapped_sides(specs):
    s = copy.deepcopy(specs["h1"])
    t = s["targets"]
    t["LeftShin"], t["RightShin"] = t["RightShin"], t["LeftShin"]
    t["LeftForeArm"][0] = "right_shoulder_roll_link"
    problems = " | ".join(ms.validate(s)["outputs"]["problems"])
    assert "LeftShin is not on the left" in problems and "not above" in problems


def _retarget(spec, K, out):
    np.save(out / "kp.npy", K)
    r = ms.handle("retarget", {"robot": spec, "names": ops.NAMES, "fps": 30.0}, {"keypoints": str(out / "kp.npy")}, out)
    c = ms.handle("check", {"robot": spec, "fps": 30.0, "physics": False}, {"qpos": r["files"]["qpos"]}, out)
    return r["outputs"], c["outputs"], np.load(r["files"]["qpos"])


@pytest.mark.parametrize("name", ROBOTS)
def test_retarget_each_robot_within_limits_with_feet_on_the_ground(specs, name, tmp_path):
    import mujoco
    spec = specs[name]
    m = mujoco.MjModel.from_xml_path(spec["xml"])
    for label, K in (("stand_wave", skeleton(1.5)), ("step", stepping(2.0))):
        out = tmp_path / label
        out.mkdir()
        r, c, q = _retarget(spec, ops.canonical(ops.Motion(K, 30.0)).joints, out)
        assert r["scaling"] == "bones" and q.shape == (len(K), m.nq)
        lim = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)]
        a = m.jnt_qposadr[lim]
        assert np.all(q[:, a] >= m.jnt_range[lim, 0] - 1e-6) and np.all(q[:, a] <= m.jnt_range[lim, 1] + 1e-6)
        assert c["joint_limit_violation_frames"] == 0
        assert c["ground_penetration_max_m"] < 0.01                     # feet on, not in, the floor
        assert r["limb_angle_err_deg"] < 25 and r["keypoint_err_pct_height"] < 6
        assert 0 <= r["limit_hit_frac"] <= 1
        if label == "stand_wave":
            assert c["frames_floating_5cm"] == 0 and c["foot_skate_mean_speed_mps"] < 0.02


def test_uniform_scaling_is_still_available(specs, tmp_path):
    s = dict(specs["g1"], scaling="uniform")
    r, c, _q = _retarget(s, ops.canonical(ops.Motion(skeleton(1.0), 30.0)).joints, tmp_path)
    assert r["scaling"] == "uniform" and c["ground_penetration_max_m"] < 0.01
