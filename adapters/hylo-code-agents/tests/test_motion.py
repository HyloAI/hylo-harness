"""The motion studio: motion edits, the MuJoCo worker (retarget, check, rollout) run in-process, and the agent loop
with a scripted model and stubbed tool workers. HYLO_MOTION_RUN=1 also runs the real tools (Kimodo, MediaPipe,
the render tool and the motion_sim worker in its uv environment) with a scripted model; that takes minutes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_ops as ops  # noqa: E402
from hylo.codeagents.motion import run_studio  # noqa: E402
from hylo.codeagents.motion_tools import StudioTools, robot_spec  # noqa: E402
from motion_testkit import REST, FakeBackend, block, motion, ms, skeleton  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402

# ------------------------------------------------------------------------------------------------- edits
def test_canonical_faces_x_on_the_ground():
    J = skeleton(walk=0.5)
    c, s = np.cos(1.0), np.sin(1.0)
    turned = J @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [3, -2, 0.1]
    m = ops.canonical(ops.Motion(turned, 30.0))
    assert abs(ops.heading(m.joints[:1])[0]) < 1e-6
    assert np.allclose(m.joints[0, 0, :2], 0, atol=1e-9)
    assert abs(m.joints[:, ops.FEET, 2].min()) < 0.02
    assert m.summary()["displacement_xy_m"][0] == pytest.approx(0.5 * 59 / 30, abs=0.02)


def test_concat_aligns_and_blends():
    a, b = motion(walk=0.5), motion(walk=0.5)
    c = ops.concat(a, b, blend=0.3)
    assert c.frames == a.frames + b.frames - 9
    root = c.joints[:, 0, 0]
    assert np.all(np.diff(root) > -1e-6)                   # keeps walking forward, no jump back to the origin
    assert root[-1] == pytest.approx(2 * a.joints[-1, 0, 0] - 0.5 * 9 / 30, abs=0.05)
    assert ops.loop(a, 3).frames == 3 * a.frames - 2 * round(0.25 * 30)


def test_mirror_retime_trim_hold_keyframe():
    m = motion()
    mm = ops.mirror(m)
    assert np.allclose(mm.joint("LeftHand")[:, 2], m.joint("RightHand")[:, 2])
    assert np.allclose(ops.mirror(mm).joints, m.joints, atol=1e-9)
    assert ops.retime(m, speed=2).frames == 30 and ops.retime(m, seconds=3).frames == 90
    assert ops.trim(m, 0.5, 1.0).frames == 15
    h = ops.hold(m.pose(1.0), 2.0)
    assert h.frames == 60 and np.allclose(h.joints[0], h.joints[-1])
    pose = ops.mirror(m).pose(0.25)                      # left hand up
    k = ops.keyframe(m, pose, at=1.0, width=0.5)
    i = 30
    rel = lambda J: J - J[0] * [1, 1, 0]                 # noqa: E731
    assert np.allclose(rel(k.joints[i]), rel(pose.joints[0]) + [0, 0, k.joints[i, 0, 2] - pose.joints[0, 0, 2]],
                       atol=1e-6)
    assert np.allclose(k.joints[0], m.joints[0])          # outside the window: unchanged


def test_from_mediapipe_fills_missing_frames():
    T = 10
    W = np.zeros((T, 33, 3))
    mp = {11: "LeftArm", 12: "RightArm", 13: "LeftForeArm", 14: "RightForeArm", 15: "LeftHand", 16: "RightHand",
          23: "LeftLeg", 24: "RightLeg", 25: "LeftShin", 26: "RightShin", 27: "LeftFoot", 28: "RightFoot",
          31: "LeftToeBase", 32: "RightToeBase", 7: "Head", 8: "Head"}
    for i, n in mp.items():
        W[:, i] = REST[n]
    valid = np.ones(T, bool)
    valid[4:6] = False
    W[4:6] = 0
    m = ops.from_mediapipe(W, valid, 30.0)
    assert np.allclose(m.joints[4], m.joints[3], atol=1e-6)
    assert m.joint("Chest")[0, 2] > m.joint("Hips")[0, 2]


# ------------------------------------------------------------------------------------------------- worker
@pytest.fixture(scope="module")
def human_robot(tmp_path_factory):
    return robot_spec("human", tmp_path_factory.mktemp("human"))


def _retarget(robot, m, out):
    np.save(out / "kp.npy", m.joints)
    return ms.handle("retarget", {"robot": robot, "names": ops.NAMES, "fps": m.fps}, {"keypoints": str(out / "kp.npy")},
                     out)


def test_retarget_and_check_human_character(human_robot, tmp_path):
    r = _retarget(human_robot, motion(), tmp_path)
    q = np.load(r["files"]["qpos"])
    assert q.shape == (60, r["outputs"]["nq"]) and r["outputs"]["mean_keypoint_err_m"] < 0.12
    assert Path(r["files"]["scene"]).exists()
    c = ms.handle("check", {"robot": human_robot, "fps": 30.0}, {"qpos": r["files"]["qpos"]}, tmp_path)["outputs"]
    assert c["ground_penetration_max_m"] < 0.01 and c["frames_floating_5cm"] == 0
    assert c["physics"].startswith("n/a")                 # the character has no actuators
    assert c["foot_skate_mean_speed_mps"] < 0.05          # standing still


def test_check_flags_penetration_and_sliding(human_robot, tmp_path):
    r = _retarget(human_robot, motion(wave=False), tmp_path)
    q = np.load(r["files"]["qpos"])
    q[:, 2] -= 0.05                                       # sink into the floor
    q[:, 0] += np.linspace(0, 1.0, len(q))                # slide 1 m with the feet down
    np.save(tmp_path / "bad.npy", q)
    c = ms.handle("check", {"robot": human_robot, "fps": 30.0, "physics": False}, {"qpos": str(tmp_path / "bad.npy")},
                  tmp_path)["outputs"]
    assert c["ground_penetration_max_m"] > 0.04 and c["frames_below_floor_1cm"] == len(q)
    assert c["foot_skate_mean_speed_mps"] > 0.3


def test_support_polygon():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    assert ms.outside_distance(np.array([0.5, 0.5]), sq) < 0
    assert ms.outside_distance(np.array([1.5, 0.5]), sq) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def g1_robot(tmp_path_factory):
    try:
        return robot_spec("g1", tmp_path_factory.mktemp("g1"))
    except RuntimeError as exc:                           # offline and not cached
        pytest.skip(str(exc))


def test_g1_retarget_stands_with_pd_tracking(g1_robot, tmp_path):
    r = _retarget(g1_robot, motion(seconds=2.0), tmp_path)
    assert r["outputs"]["nq"] == 36 and r["outputs"]["mean_keypoint_err_m"] < 0.1
    c = ms.handle("check", {"robot": g1_robot, "fps": 30.0, "ankle_gain": 1.0}, {"qpos": r["files"]["qpos"]},
                  tmp_path)["outputs"]
    assert c["self_collision_frames"] == 0 and c["joint_limit_violation_frames"] == 0
    assert c["with_ankle_strategy"]["upright_s"] == pytest.approx(c["clip_s"], abs=0.05)


# ------------------------------------------------------------------------------------------------- the loop
def test_studio_run_with_scripted_model(tmp_path):
    replies = [block("m = text_motion(['a person waves', 'a person bows'], seconds=[1.0, 1.0])\nprint(m)\n"
                     "rep = check(m, robot='human')\nprint(rep['problems'], rep['kinematics']['ground_penetration_max_m'])"),
               block("w = concat(m, mirror(m), blend=0.2)\nprint(w.seconds)"),
               block("finish(w, 'wave, then mirrored')")]
    be = FakeBackend()
    s = run_studio("wave, then bow", tmp_path, robot="human", model=ScriptedChat(replies), backend=be,
                   side_by_side=False)
    assert s["stopped"] == "finished" and s["turns"] == 3 and s["finished_by_model"]
    assert (tmp_path / "motion.mp4").exists() and (tmp_path / "check.json").exists()
    z = np.load(tmp_path / "motion.npz")
    assert z["joints"].shape == (int(round(3.8 * 30)), 17, 3) and str(z["robot"]) == "human"
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    turns = [r for r in log if "turn" in r]
    assert len(turns) == 3 and not any(r.get("error") for r in turns)
    assert {r["tool"] for r in log if r.get("event") == "tool"} >= {"text_motion", "retarget", "check", "render"}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["cost_usd"] == 0 and summary["wall_s"] >= 0 and "text_motion" in summary["tool_seconds"]
    assert be.calls[-1] == ("close",)


def test_studio_inputs_budget_and_side_by_side(tmp_path, g1_robot):
    img = tmp_path / "pose.png"
    img.write_bytes(b"not used by the fake")
    replies = [block("p = pose_from_image('image')\nprint(p.frames)\nprint(video_motion('/etc/passwd'))"),
               block("m = keyframe(text_motion('a person stands', 2.0), p, at=1.0)\nprint(m)")]
    s = run_studio("strike this pose", tmp_path / "run", robot="g1", image=str(img), model=ScriptedChat(replies),
                   backend=FakeBackend(), budget="turns=2")
    assert s["stopped"].startswith("budget") and s["final"] is None and "did not finish" in s["error"]
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert "only the provided inputs" in log[2]["error"] or "only the provided inputs" in json.dumps(log)
    replies = [block("finish(hold(text_motion('a person stands', 1.0).pose(0.5), 1.0))")]
    s = run_studio("stand", tmp_path / "split", robot="g1", model=ScriptedChat(replies), backend=FakeBackend(),
                   side_by_side=True)
    assert s["files"]["split"] == "split.mp4" and (tmp_path / "split" / "split.mp4").exists()
    assert "upright_s" in s["check"]


def test_tools_reject_bad_input_paths(tmp_path):
    t = StudioTools(FakeBackend(), tmp_path, "human", {"video": str(tmp_path / "v.mp4")})
    with pytest.raises(ValueError):
        t.video_motion("/etc/hosts")
    with pytest.raises(ValueError):
        t.video_motion("/etc/v.mp4")                       # same file name elsewhere: refused
    assert t._media("v.mp4") == t._media("video") == str(tmp_path / "v.mp4")
    with pytest.raises(ValueError):
        t.motion_from_array(np.zeros((4, 3, 3)))


# ------------------------------------------------------------------------------------------------- real tools
REAL = pytest.mark.skipif(not os.environ.get("HYLO_MOTION_RUN"), reason="set HYLO_MOTION_RUN=1 to run the real tools")


@REAL
def test_real_text_to_g1(tmp_path):
    replies = [block("m = text_motion('a person waves with the right hand', 3.0)\nprint(m, check(m))"),
               block("finish(m)")]
    s = run_studio("wave", tmp_path, robot="g1", model=ScriptedChat(replies), style="mujoco")
    assert s["stopped"] == "finished" and (tmp_path / "motion.mp4").stat().st_size > 10_000
    assert s["check"]["mean_keypoint_err_m"] < 0.1 and s["check"]["self_collision_frames"] < 10
    assert "upright_s" in s["check"] and "with_ankle_strategy" in s["check"]


@REAL
def test_real_pose_from_image(tmp_path):
    import urllib.request
    img = tmp_path / "pose.jpg"
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-assets/pose.jpg", img)
    replies = [block("p = pose_from_image('image')\nprint(p.summary())\nfinish(hold(p, 1.0))")]
    s = run_studio("hold this pose", tmp_path / "run", robot="human", image=str(img), model=ScriptedChat(replies),
                   style="mujoco")
    assert s["stopped"] == "finished"
    z = np.load(tmp_path / "run" / "motion.npz")
    J = z["joints"][0]
    span = abs(J[ops.IDX["LeftHand"], 1] - J[ops.IDX["RightHand"], 1])
    assert span > 1.0                                      # warrior II: arms stretched out to the sides
