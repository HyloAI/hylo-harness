"""text_motion: Kimodo text -> SOMA skeleton motion on the CPU, retargeted to the built-in MuJoCo humanoid; the
retargeting hook on a synthetic motion (no model)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tools_testkit import copy_example, needs_run, record, save_example_png

from hylo.tools.text_motion import LICENCE, MODEL, KeypointIKRetargeter, retarget, text_motion

SOMA_BODY = ["Hips", "Chest", "Head", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot",
             "LeftToeBase", "RightToeBase", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand",
             "RightHand"]


def _standing(T: int = 20) -> np.ndarray:
    """A 1.7 m person standing, arms down, then raising the right hand (z up, x forward, metres)."""
    p = {"Hips": (0, 0, 0.95), "Chest": (0, 0, 1.35), "Head": (0, 0, 1.62), "LeftLeg": (0, 0.1, 0.92),
         "RightLeg": (0, -0.1, 0.92), "LeftShin": (0.02, 0.1, 0.5), "RightShin": (0.02, -0.1, 0.5),
         "LeftFoot": (0, 0.1, 0.08), "RightFoot": (0, -0.1, 0.08), "LeftToeBase": (0.14, 0.1, 0.02),
         "RightToeBase": (0.14, -0.1, 0.02), "LeftArm": (0, 0.2, 1.42), "RightArm": (0, -0.2, 1.42),
         "LeftForeArm": (0, 0.22, 1.13), "RightForeArm": (0, -0.22, 1.13), "LeftHand": (0, 0.23, 0.88),
         "RightHand": (0, -0.23, 0.88)}
    J = np.tile(np.array([p[n] for n in SOMA_BODY], float), (T, 1, 1))
    th = np.linspace(0, np.pi * 0.9, T)              # the right arm swings sideways from down to almost straight up
    arm = np.c_[np.zeros(T), -np.sin(th), -np.cos(th)]
    sh = J[:, SOMA_BODY.index("RightArm")]
    J[:, SOMA_BODY.index("RightForeArm")] = sh + 0.29 * arm
    J[:, SOMA_BODY.index("RightHand")] = sh + 0.54 * arm
    return J


def test_retarget_hook_synthetic():
    pytest.importorskip("mujoco")
    J = _standing()
    rob = retarget((J, SOMA_BODY, 20.0), KeypointIKRetargeter(iters=40))
    assert rob.qpos.shape[0] == len(J) and rob.qpos.shape[1] == 7 + len(rob.joint_names)
    assert rob.err.max() < 0.08, rob.err            # keypoints within 8 cm after scaling to the robot
    # the right hand goes up: the robot's right wrist ends above its head
    import mujoco
    rt = KeypointIKRetargeter()
    d = rt.data
    d.qpos[:] = rob.qpos[-1]
    mujoco.mj_kinematics(rt.model, d)
    wrist = d.site_xpos[rt.model.site("r_wrist").id]
    head = d.site_xpos[rt.model.site("head").id]
    assert wrist[2] > head[2]


@needs_run
def test_text_motion_kimodo():
    t0 = time.perf_counter()
    mo = text_motion("a person walks forward and waves with the right hand", duration=5.0, seed=0)
    wall = time.perf_counter() - t0
    T = len(mo.joints)
    assert T == int(round(5.0 * mo.fps)) and mo.joints.shape[1] == len(mo.names) == 77
    assert mo.parents[0] == -1 and "Hips" in mo.names
    hips = mo.joints[:, mo.names.index("Hips")]
    assert 0.7 < np.median(hips[:, 2]) < 1.2                       # z up, standing pelvis height
    assert hips[-1, 0] - hips[0, 0] > 0.5                          # walked forward (+x)
    feet = mo.joints[:, [mo.names.index("LeftFoot"), mo.names.index("RightFoot")], 2]
    assert feet.min() > -0.05 and np.median(feet.min(1)) < 0.2      # feet on the floor
    assert mo.robot is not None and mo.robot.qpos.shape[0] == T
    assert mo.result.outputs["retarget"]["mean_keypoint_err_m"] < 0.1
    from PIL import Image
    files = mo.result.files
    examples = [save_example_png(Image.open(files["skeleton_strip"]), "text_motion_skeleton", max_side=1600)]
    if "robot_strip" in files:
        examples.append(save_example_png(Image.open(files["robot_strip"]), "text_motion_robot", max_side=1600))
        examples.append(copy_example(files["robot_video"], "text_motion_robot.mp4"))
    examples.append(copy_example(files["skeleton_video"], "text_motion_skeleton.mp4"))
    tm = mo.result.timing
    record("text_motion", status="works", model=MODEL, licence=LICENCE, wall_s=wall, timing=tm, examples=examples,
           notes=f"Kimodo-SOMA-RP-v1.2, 5 s ({T} frames at {mo.fps:g} fps), 100 denoising steps, text encoder "
                 f"LLM2Vec-Llama-3-8B fp32 on CPU; retargeted to the built-in MuJoCo humanoid by keypoint IK "
                 f"(mean keypoint error {mo.result.outputs['retarget']['mean_keypoint_err_m']} m); "
                 f"model load {tm.get('model_load_s')} s (once per worker), generate {tm.get('generate_s')} s, "
                 f"retarget {tm.get('retarget_s')} s; rest: rendering and environment check")


@needs_run
def test_text_motion_g1():
    """Kimodo-G1: text -> Unitree G1 joint angles directly (MuJoCo qpos for Kimodo's G1 MJCF)."""
    mo = text_motion("a person walks forward and waves with the right hand", duration=4.0, model="g1", seed=0)
    assert mo.robot is not None and mo.robot.qpos.shape == (len(mo.joints), 36)
    assert len(mo.robot.joint_names) == 29 and mo.robot.joint_names[0] == "left_hip_pitch_joint"
    assert mo.robot.qpos[-1, 0] - mo.robot.qpos[0, 0] > 0.5         # the base walked forward (+x)
    if "robot_strip" in mo.result.files:
        from PIL import Image
        save_example_png(Image.open(mo.result.files["robot_strip"]), "text_motion_g1", max_side=1600)
