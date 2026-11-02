"""motion_retarget: the shared retargeting math (pure numpy, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))      # motion_testkit (pytest runs in importlib mode)

from hylo.codeagents import motion_ops as ops  # noqa: E402
from hylo.codeagents import motion_retarget as rt  # noqa: E402
from motion_testkit import REST, skeleton, stepping  # noqa: E402

IDX = ops.IDX


def character(scale_legs: float = 0.7, scale_arms: float = 0.5) -> dict[str, np.ndarray]:
    """A rest pose with other proportions than REST: shorter legs and arms, wider hips."""
    R = {k: np.array(v, float) for k, v in REST.items()}
    for s in ("Left", "Right"):
        R[f"{s}Leg"][1] *= 1.3
        for a, b in (("Leg", "Shin"), ("Shin", "Foot"), ("Foot", "ToeBase")):
            R[f"{s}{b}"] = R[f"{s}{a}"] + (np.array(REST[f"{s}{b}"]) - np.array(REST[f"{s}{a}"])) * scale_legs
        for a, b in (("Arm", "ForeArm"), ("ForeArm", "Hand")):
            R[f"{s}{b}"] = R[f"{s}{a}"] + (np.array(REST[f"{s}{b}"]) - np.array(REST[f"{s}{a}"])) * scale_arms
    return R


def test_body_frames_are_rotations_facing_forward():
    J = skeleton(1.0)
    Fp, Fc = rt.body_frames(J, IDX)
    for F in (Fp, Fc):
        assert np.allclose(np.einsum("tji,tjk->tik", F, F), np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(F), 1.0, atol=1e-6)
    assert np.allclose(Fp[0, :, 0], [1, 0, 0], atol=0.05)          # forward is +x, left +y, up +z
    assert np.allclose(Fp[0, :, 2], [0, 0, 1], atol=0.05)


def test_proportional_targets_keep_the_characters_lengths_and_the_sources_directions():
    K = stepping(2.0)
    R = character()
    Tg, s = rt.proportional_targets(K, IDX, R)
    assert s == pytest.approx(rt.leg_length(np.stack([R[n] for n in ops.NAMES]), IDX) / rt.leg_length(K, IDX))
    for p, c in rt.LIMBS:
        want = np.linalg.norm(R[c] - R[p])
        assert np.allclose(np.linalg.norm(Tg[:, IDX[c]] - Tg[:, IDX[p]], axis=1), want, atol=1e-9)
    assert rt.segment_angles(Tg, K, IDX).max() < 0.05                # limbs point like the source's
    assert np.allclose(Tg[:, IDX["Hips"]], K[:, IDX["Hips"]] * s)   # the root path is scaled
    hip_w = np.linalg.norm(Tg[:, IDX["LeftLeg"]] - Tg[:, IDX["RightLeg"]], axis=1)
    assert np.allclose(hip_w, np.linalg.norm(R["LeftLeg"] - R["RightLeg"]), atol=1e-9)   # the character's hips


def test_uniform_scale_and_leg_length():
    R = {k: np.array(v) * 0.5 for k, v in REST.items()}
    K = skeleton(0.5)
    assert rt.uniform_scale(K, IDX, R) == pytest.approx(0.5, rel=1e-6)
    assert rt.leg_length(K, IDX) == pytest.approx(rt.leg_length(K[0], IDX))


def test_lock_feet_removes_sliding_of_a_planted_foot():
    K = skeleton(1.0, wave=False)
    K[:, IDX["LeftFoot"], 0] += np.linspace(0, 0.1, len(K))          # the planted foot slides 10 cm
    L = rt.lock_feet(K, IDX)
    assert np.ptp(L[3:-3, IDX["LeftFoot"], 0]) < 0.01
    assert np.allclose(L[:, IDX["Head"]], K[:, IDX["Head"]])


def test_smooth_is_a_moving_average():
    K = np.zeros((20, 17, 3))
    K[10] = 1.0
    S = rt.smooth(K, 2)
    assert S[10, 0, 0] == pytest.approx(0.2) and S.sum() == pytest.approx(K.sum())
    assert rt.smooth(K[:3], 2) is not None and np.array_equal(rt.smooth(K[:3], 2), K[:3])


def test_ground_correction_keeps_feet_on_the_floor_and_jumps_in_the_air():
    T = 60
    src = np.full(T, 0.02)
    src[20:30] = 0.3                                                  # a jump
    low = np.full(T, 0.05)                                            # the character floats 5 cm
    low[25] = -0.02
    corr = rt.ground_correction(low, src, 1.0)
    assert np.all(low + corr >= -1e-9)                               # never below the floor
    assert np.allclose((low + corr)[:15], 0, atol=1e-9)              # grounded frames touch it
    assert rt.feet_height(stepping(1.0), IDX).min() == pytest.approx(0.02, abs=1e-6)
