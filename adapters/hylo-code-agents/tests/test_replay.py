"""Replay-to-sim workflow (RFC-0006 part 8): deterministic pipeline on a synthetic recorded episode (no tool workers,
no OpenGL: depth and segmentation are ray-cast with mj_multiRay), event detection, registration, hand retargeting,
scene building, metrics and the agent loop with a scripted model.

The slow test (HYLO_REPLAY_RUN=1) records one mujoco.panda_table episode (needs an OpenGL backend) and runs the real
tools (SAM 3, mesh, convex, render) on it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hylo.codeagents.replay import ReplayConfig, ReplayPipeline, run_agent  # noqa: E402
from hylo.codeagents.replay_data import Recording, axis_angle_to_R, euler_xyz_to_R, from_fetched  # noqa: E402
from hylo.codeagents.replay_geom import (SIM_WORKER, box_mesh, final_state, pose_errors, size_error,  # noqa: E402
                                         write_obj)
from hylo.codeagents.replay_tools import StubTools, gt_segmenter  # noqa: E402
from hylo.codeagents.replay_track import (closed_intervals, detect_events, hand_to_gripper,  # noqa: E402
                                          labels_from_text, match_events, propose_labels, register_similarity, rotz)

W, H, FPS = 96, 72, 10.0


def _load_worker():
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_sim_worker", SIM_WORKER)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod                 # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


sim = _load_worker()
TOP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])      # TCP approach (+x) pointing down


def _pose(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def _camera():
    pos, target = np.array([0.45, -0.05, 0.45]), np.array([0.0, 0.02, 0.0])
    z = (target - pos) / np.linalg.norm(target - pos)                     # OpenCV: z forward, y down, x right
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    f = 0.5 * H / math.tan(math.radians(45.0) / 2)
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    return K, _pose(np.stack([x, y, z], 1), pos)


def _tcp_path():
    """Approach the block, close, lift, carry 12 cm in y, lower, open, rise (10 Hz)."""
    keys = [(0.0, [0, 0, 0.16], 1), (1.0, [0, 0, 0.02], 1), (1.6, [0, 0, 0.02], 0), (2.6, [0, 0, 0.12], 0),
            (3.8, [0, 0.12, 0.12], 0), (4.8, [0, 0.12, 0.023], 0), (5.4, [0, 0.12, 0.023], 1), (6.4, [0, 0.12, 0.16], 1)]
    t = np.arange(0, keys[-1][0] + 1e-9, 1 / FPS)
    kt = np.array([k[0] for k in keys])
    kp = np.array([k[1] for k in keys], float)
    pos = np.stack([np.interp(t, kt, kp[:, i]) for i in range(3)], 1)
    cmd = np.array([keys[max(0, np.searchsorted(kt, x, side="right") - 1)][2] for x in t], float)
    return t, np.stack([_pose(TOP, p) for p in pos]), cmd


def _raycast(m, d, K, Tc, ids):
    v, u = np.mgrid[0:H, 0:W]
    dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    vec = dirs @ Tc[:3, :3].T
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    n = len(vec)
    geomid, dist = np.zeros(n, np.int32), np.zeros(n)
    mujoco.mj_multiRay(m, d, Tc[:3, 3].copy(), vec.ravel().copy(), None, 1, -1, geomid, dist, None, n, 5.0)
    hit = geomid >= 0
    zc = dist * (vec @ Tc[:3, 2])                                          # range -> depth along the optical axis
    depth = np.where(hit, zc, 0.0).reshape(H, W).astype(np.float32)
    seg = np.where(hit, ids[np.clip(geomid, 0, None)], 0).reshape(H, W)
    return depth, seg


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> Recording:
    """A floating gripper picks a red 4 cm block and puts it down 12 cm away; RGB-D + segmentation ray-cast."""
    work = tmp_path_factory.mktemp("synthetic")
    V, F = box_mesh([0.04, 0.04, 0.04], 0.0, [0, 0, 0])
    part = str(write_obj(work / "block.obj", V, F))
    block = sim.SimObject("block", [part], None, _pose(rotz(0.3), [0, 0, 0.02]), (0.85, 0.1, 0.1, 1), True, 0.05)
    t, tcp, cmd = _tcp_path()
    scene = sim.build_scene(work / "gen" / "scene.xml", table_z=0.0, table_extent=np.array([[-0.3, 0.3], [-0.3, 0.3]]),
                            objects=[block], agent="gripper", gripper_T0=tcp[0])
    phys = sim.physics_replay(scene, t, None, tcp, cmd, {"block": block.pose0})
    m = scene.load()
    d = mujoco.MjData(m)
    lay = sim.Layout(scene)
    ids = np.zeros(m.ngeom, np.int64)
    ids[m.geom("table").id] = 1
    ids[list(lay.agent_geoms)] = 2
    ids[list(lay.obj_geoms["block"])] = 3
    colours = {0: (40, 40, 40), 1: (180, 150, 110), 2: (200, 200, 205), 3: (215, 30, 30)}
    K, Tc = _camera()
    rgb, depth, seg = [], [], []
    for k in range(len(t)):
        d.qpos[:] = phys["qpos"][k]
        mujoco.mj_forward(m, d)
        z, s = _raycast(m, d, K, Tc, ids)
        depth.append(z)
        seg.append(s)
        rgb.append(np.array([colours[int(i)] for i in s.ravel()], np.uint8).reshape(H, W, 3))
    gt = {"objects": {"block": {"label": "red block", "kind": "block", "half_size": [0.02, 0.02, 0.02],
                                "poses": phys["poses"]["block"]}},
          "seg": np.stack(seg), "segments": {"1": "table", "2": "robot", "3": "red block"}, "success": True}
    return Recording("synthetic-pick", np.stack(rgb), t, "put the red block to the left", "gripper",
                     depth=np.stack(depth), K=K, T_world_cam=Tc, tcp=tcp, gripper=phys["opening"], gripper_cmd=cmd,
                     meta={"source": "synthetic", "fps": FPS, "frame": "world", "success": True, "licence": "ours"},
                     gt=gt)


def test_synthetic_episode_moves_the_block(synthetic):
    p = synthetic.gt["objects"]["block"]["poses"][:, :3, 3]
    assert p[-1][1] - p[0][1] > 0.10 and p[:, 2].max() > 0.09
    assert (synthetic.gt["seg"][0] == 3).sum() > 30


def test_pipeline_on_synthetic_episode(synthetic, tmp_path):
    tools = StubTools(gt_segmenter(synthetic.gt["seg"][0], synthetic.gt["segments"]))
    pipe = ReplayPipeline(synthetic, tmp_path, tools, config=ReplayConfig())
    rep = pipe.run(labels=["red block"], video=False)
    steps = rep["steps"]
    assert [m["label"] for m in steps["segment"]["masks"]] == ["red block"]
    track = steps["track"]["objects"][0]
    assert track["lost_frames"] <= 3 and track["moved_m"] > 0.1
    obj = steps["objects"]["objects"][0]
    assert obj["dynamic"] and obj["size_error"]["max_abs_err_m"] < 0.012
    c = steps["checks"]
    assert c["pose_error_tracked_vs_gt"]["red_block"]["trans_mean_m"] < 0.01
    types = [e["type"] for e in c["events"]["recorded"]]
    assert types[:2] == ["grasp", "lift"] and "place" in types
    assert c["events"]["match"]["matched"] == c["events"]["match"]["of"] == 3
    assert c["final_state"]["success"] and c["success_reproduced"]
    assert c["pose_error_sim_vs_gt"]["red_block"]["trans_final_m"] < 0.03
    assert (tmp_path / "scene.xml").is_file() and (tmp_path / "report.json").is_file()
    z = np.load(tmp_path / "tracks.npz")
    assert z["tracked_red_block"].shape == (synthetic.T, 4, 4) and z["sim_red_block"].shape == (synthetic.T, 4, 4)
    json.loads((tmp_path / "report.json").read_text())


def test_recording_round_trip_and_subsample(synthetic, tmp_path):
    synthetic.save(tmp_path / "rec")
    back = Recording.load(tmp_path / "rec")
    assert back.T == synthetic.T and np.allclose(back.tcp, synthetic.tcp)
    assert np.allclose(back.gt["objects"]["block"]["poses"], synthetic.gt["objects"]["block"]["poses"])
    sub = back.subsample(2)
    assert sub.T == (synthetic.T + 1) // 2 and sub.gt["seg"].shape[0] == sub.T and sub.fps == pytest.approx(FPS / 2)


def test_closed_intervals_and_events():
    g = np.r_[np.ones(10), np.linspace(1, 0.55, 3), np.full(15, 0.55), np.linspace(0.55, 1, 3), np.ones(10)]
    ivs = closed_intervals(g)
    assert ivs == [(11, 30)]
    t = np.arange(len(g)) / 10
    z = np.r_[np.full(15, 0.02), np.linspace(0.02, 0.12, 5), np.full(6, 0.12), np.linspace(0.12, 0.02, 5), np.full(10, 0.02)]
    obj = np.stack([np.zeros(len(g)), np.r_[np.zeros(18), np.linspace(0, 0.1, 10), np.full(13, 0.1)], z], 1)
    still = np.tile([0.2, 0.0, 0.02], (len(g), 1))
    ev = detect_events(t, g, {"cube": obj, "plate": still})
    assert [e["type"] for e in ev] == ["grasp", "lift", "release", "place"]
    assert all(e["object"] == "cube" for e in ev)
    # closing on nothing (fully closed) is a missed grasp
    g2 = np.r_[np.ones(10), np.zeros(10), np.ones(10)]
    assert detect_events(np.arange(30) / 10, g2, {"cube": np.zeros((30, 3))})[0]["type"] == "missed_grasp"
    m = match_events(ev, [{**e, "t": e["t"] + 0.3} for e in ev if e["type"] != "place"])
    assert m["matched"] == 2 and m["of"] == 3


def test_registration_recovers_a_similarity():
    rng = np.random.default_rng(0)
    obj = np.cumsum(rng.normal(0, 0.02, (40, 3)), 0)
    s, yaw, t = 1.7, 0.6, np.array([0.3, -0.2, 0.9])
    tcp = s * obj @ rotz(yaw).T + t + rng.normal(0, 1e-4, obj.shape)
    reg = register_similarity(obj, tcp)
    assert reg["scale"] == pytest.approx(s, rel=1e-2) and reg["yaw"] == pytest.approx(yaw, abs=1e-2)
    assert np.allclose(reg["t"], t, atol=2e-3) and reg["rmse"] < 1e-3


def test_hand_to_gripper_aperture_and_frame():
    T = 6
    kp = np.zeros((T, 21, 3))
    kp[:, 0] = [0.0, 0.0, 0.5]                                  # wrist
    for j in (5, 9, 17):
        kp[:, j] = [0.0, -0.08, 0.5]
    gap = np.linspace(0.10, 0.03, T)
    kp[:, 4] = np.stack([-gap / 2, np.full(T, -0.15), np.full(T, 0.5)], 1)      # thumb tip
    kp[:, 8] = np.stack([gap / 2, np.full(T, -0.15), np.full(T, 0.5)], 1)       # index tip
    kp2d = np.zeros((T, 21, 2))
    depth = np.zeros((T, 10, 10))
    out = hand_to_gripper(kp, kp2d, np.ones(T, bool), depth, np.eye(3), np.eye(4), max_opening=0.12, smooth=1)
    assert np.allclose(out["aperture"], gap, atol=1e-6)
    assert np.allclose(out["tcp"][:, :3, 1], [1, 0, 0], atol=1e-6)          # closing axis thumb -> index
    assert np.allclose(out["tcp"][:, :3, 0], [0, -1, 0], atol=1e-6)         # approach: wrist -> tips
    assert out["gripper"][0] == pytest.approx(0.10 / 0.12)


def test_scene_with_fixed_and_moving_objects(tmp_path):
    V, F = box_mesh([0.04, 0.04, 0.04], 0.0, [0, 0, 0])
    cube, plate = str(write_obj(tmp_path / "cube.obj", V, F)), str(write_obj(tmp_path / "plate.obj", V * [3, 3, 0.2], F))
    a = sim.SimObject("cube", [cube], cube, _pose(None, [0.0, 0.0, 0.02]), dynamic=True)
    b = sim.SimObject("plate", [plate], None, _pose(None, [0.2, 0.0, 0.004]), dynamic=False)
    K, Tc = _camera()
    sc = sim.build_scene(tmp_path / "scene.xml", table_z=0.0, table_extent=np.array([[-0.3, 0.3], [-0.3, 0.3]]),
                         objects=[a, b], agent="gripper", camera={"K": K, "T_world_cam": Tc, "width": W, "height": H})
    m = mujoco.MjModel.from_xml_path(str(sc.path))
    lay = sim.Layout(sc)
    assert set(lay.obj_q) == {"cube"} and "plate" in lay.obj_body
    assert m.camera("replay_cam").id >= 0 and m.nmocap == 1
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=500)
    assert abs(d.xpos[lay.obj_body["cube"]][2] - 0.02) < 0.005          # rests on the table


def test_metrics():
    ref = np.stack([_pose(rotz(0.0), [0.1 * k, 0, 0]) for k in range(5)])
    est = np.stack([_pose(rotz(math.radians(10)), [0.1 * k, 0.01, 0]) for k in range(5)])
    e = pose_errors(ref, est)
    assert e["trans_mean_m"] == pytest.approx(0.01) and e["rot_mean_deg"] == pytest.approx(10.0, abs=1e-3)
    s = size_error([0.05, 0.04, 0.045], [0.045, 0.045, 0.045])
    assert s["max_abs_err_m"] == pytest.approx(0.005)
    fs = final_state({"a": ref[-1]}, {"a": est[-1]}, ["a"], tol=0.02)
    assert fs["success"]


def test_profiles_convert_robot_state(tmp_path):
    # LIBERO: a top-down eef (axis-angle pi about x) is a TCP approaching along -z; fingers 0.04 / -0.04 open
    state = np.array([[0.1, 0.0, 1.0, math.pi, 0.0, 0.0, 0.04, -0.04], [0.1, 0.0, 0.95, math.pi, 0.0, 0.0, 0.01, -0.01]], np.float32)
    np.savez(tmp_path / "recording.npz", rgb=np.zeros((2, 8, 8, 3), np.uint8), time=np.array([0, 0.1]), state=state)
    (tmp_path / "source.json").write_text(json.dumps({"repo": "x/libero", "episode": 3, "camera": "image", "fps": 10,
                                                      "task": "put the bowl on the plate", "licence": "cc-by-4.0"}))
    rec = from_fetched(tmp_path, "libero")
    assert np.allclose(rec.tcp[0][:3, 0], [0, 0, -1], atol=1e-6) and rec.gripper[0] == pytest.approx(1.0)
    assert rec.gripper[1] == pytest.approx(0.25) and rec.meta["licence"] == "cc-by-4.0"
    assert np.allclose(axis_angle_to_R([0, 0, math.pi / 2]) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(euler_xyz_to_R([0, 0, math.pi / 2]), axis_angle_to_R([0, 0, math.pi / 2]))


def test_labels_from_text_and_vlm():
    from hylo.reasoners.llm.chat import ScriptedChat

    assert labels_from_text("put the red block on the white plate") == ["red block", "white plate"]
    model = ScriptedChat(['Objects: ["red block", "the robot gripper", "plate"]'])
    res = propose_labels(model, np.zeros((8, 8, 3), np.uint8), "put the red block on the plate")
    assert res["labels"] == ["red block", "plate"] and res["by"] == "vlm"
    assert propose_labels(None, np.zeros((8, 8, 3), np.uint8), "pick up the mug")["labels"] == ["mug"]


def test_agent_drives_the_steps(synthetic, tmp_path):
    from hylo.reasoners.llm.chat import ScriptedChat

    blocks = [
        "print(replay.info())",
        'r = replay.labels(["red block", "blue bowl"])\nprint(replay.segment())',
        "print(replay.lift())\nprint(replay.track(mode='yaw'))",
        "print(replay.register())\nprint(replay.events())",
        "print(replay.objects())\nprint(replay.scene())\nprint(replay.replay())",
        "c = replay.checks()\nprint(c['final_state'])",
        'finish("block grasped, carried and placed in simulation")',
    ]
    model = ScriptedChat([f"Next step.\n```python\n{b}\n```" for b in blocks])
    tools = StubTools(gt_segmenter(synthetic.gt["seg"][0], synthetic.gt["segments"]))
    pipe = ReplayPipeline(synthetic, tmp_path, tools)          # labels and retries without the scripted model
    rep = run_agent(pipe, model, turns=10)
    assert rep["agent"]["stopped"] == "finish()" and rep["agent"]["turns"] == len(blocks)
    assert rep["steps"]["checks"]["final_state"]["success"]
    turns = [json.loads(line) for line in (tmp_path / "agent_turns.jsonl").read_text().splitlines()]
    assert not any(t.get("error") for t in turns), [t.get("error") for t in turns]
    assert "masks" in turns[1]["images"][0] or turns[1]["images"]                  # the overlay went to the model


@pytest.mark.skipif(os.environ.get("HYLO_REPLAY_RUN") != "1", reason="set HYLO_REPLAY_RUN=1 (records a Panda episode "
                    "with OpenGL and runs the SAM 3 / mesh / convex / render workers)")
def test_real_tools_on_a_panda_episode(tmp_path):
    from hylo.codeagents.replay import replay_command

    out = tmp_path / "pp"
    assert replay_command(["panda:pick_place:1", "--out", str(out), "--no-vlm"]) == 0
    rep = json.loads((out / "report.json").read_text())
    c = rep["steps"]["checks"]
    assert rep["steps"]["segment"]["masks"]
    assert all(v["trans_mean_m"] < 0.03 for v in c["pose_error_tracked_vs_gt"].values())
    assert Path(rep["steps"]["video"]["path"]).stat().st_size <= 4_000_000


# ---------------------------------------------------------------------------------------------------- multi-finger hand
def _hand_kit():
    """The synthetic three-finger hand of hylo-manip's tests (MJCF text and HandSpec)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[3] / "packages" / "hylo-manip" / "tests" / "test_manip_dexgrasp.py"
    if not path.exists():
        pytest.skip("hylo-manip tests not found")
    spec = importlib.util.spec_from_file_location("hylo_manip_dexgrasp_kit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hand_params(kit, model_file):
    return {"xml": str(model_file), "ee": "palm", "joints": list(kit.JOINTS), "tcp_in_ee": kit.TCP,
            "open": list(kit.OPEN), "closed": list(kit.POWER), "kp_scale": 1.0}


def test_hand_scene_kinematic_and_physics_replay(tmp_path):
    kit = _hand_kit()
    model = tmp_path / "hand3.xml"
    model.write_text(kit.HAND_XML)
    t, tcp, cmd = _tcp_path()
    hand = _hand_params(kit, model)
    sc = sim.build_scene(tmp_path / "scene.xml", table_z=0.0, table_extent=np.array([[-0.3, 0.3], [-0.3, 0.3]]),
                         objects=[], agent="hand", gripper_T0=tcp[0], hand=hand)
    lay = sim.Layout(sc)
    assert len(lay.finger_q) == 6 and (lay.finger_act >= 0).all() and lay.agent_bodies
    q = np.tile(np.asarray(kit.OPEN), (len(t), 1))
    q[len(t) // 2:] = kit.POWER
    kin = sim.kinematic_replay(sc, q, tcp, np.ones(len(t)), {}, [], {})
    m = sc.load()
    d = mujoco.MjData(m)
    d.qpos[:] = kin["qpos"][-1]
    mujoco.mj_forward(m, d)
    assert np.allclose(d.qpos[lay.finger_q], np.clip(kit.POWER, m.jnt_range[[m.joint(j).id for j in kit.JOINTS], 0],
                                                    m.jnt_range[[m.joint(j).id for j in kit.JOINTS], 1]))
    assert np.allclose(lay.tcp_point(d), tcp[-1][:3, 3], atol=1e-6)
    ph = sim.physics_replay(sc, t, q, tcp, cmd, {})
    assert not ph["diverged"]
    d.qpos[:] = ph["qpos"][-1]
    mujoco.mj_forward(m, d)
    assert np.linalg.norm(lay.tcp_point(d) - tcp[-1][:3, 3]) < 0.01          # the floating hand follows its path
    assert ph["opening"][0] > 0.9 and ph["opening"][-1] < 0.2                 # open, then closed on nothing


def test_pipeline_with_a_multi_finger_hand_and_stub_tools(synthetic, tmp_path):
    """The replay steps with a multi-finger hand in place of the gripper: the recorded TCP path stands in for the
    retargeted motion, the grasp is refined with the dexterous planner, and the hand replays it in physics."""
    from hylo.manip.handmodel import HandModel
    from hylo.manip.retarget import RetargetResult

    kit = _hand_kit()
    model = tmp_path / "hand3.xml"
    model.write_text(kit.HAND_XML)
    hm = HandModel(str(model), "palm", kit.SPEC, name="hand3_right")
    tools = StubTools(gt_segmenter(synthetic.gt["seg"][0], synthetic.gt["segments"]))
    pipe = ReplayPipeline(synthetic, tmp_path / "out", tools, config=ReplayConfig(hand="hand3"))
    pipe.propose_labels(["red block"])
    pipe.segment()
    pipe.lift_to_3d()
    pipe.track()
    pipe.register()
    pipe.events()
    pipe.build_objects()
    T = synthetic.T
    tcp = synthetic.tcp @ np.linalg.inv(np.asarray(kit.TCP)) @ np.asarray(kit.TCP)
    res = RetargetResult(np.tile(hm.open, (T, 1)), tcp @ np.linalg.inv(hm.tcp_in_ee), tcp, np.zeros(T), np.zeros(T),
                         1.0, 1.0, {}, False, np.ones(T, bool))
    pipe.dex = {"name": "hand3_right", "model": hm, "model_file": str(model), "ret": res, "kp": np.zeros((T, 21, 3)),
                "tcp": res.tcp.copy(), "fingers": res.q.copy(), "side": "right", "grasps": []}
    g = pipe.refine_grasps()
    assert g["refined"] and g["refined"][0]["object"] == "red_block" and g["refined"][0]["candidates"] > 0
    a = g["refined"][0]["frame"]
    assert not np.allclose(pipe.dex["fingers"][a + 3], hm.open)               # closing on the block
    scene = pipe.build_scene()
    assert scene["nq"] > 7
    rep = pipe.replay()
    assert not rep["physics"]["diverged"]
    c = pipe.checks()
    assert c["events"]["recorded"] and "match" in c["events"]
    saved = pipe.save()
    assert "grasps" in saved["steps"]
    z = np.load(tmp_path / "out" / "tracks.npz")
    assert z["hand_fingers"].shape == (T, 6) and z["hand_tcp"].shape == (T, 4, 4)
