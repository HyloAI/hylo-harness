"""`text_motion`: text -> humanoid motion (NVIDIA Kimodo on the CPU) with a retargeting hook for robot humanoids.

    hylo tool text_motion "a person walks forward and waves" [--duration 4] [--model soma|g1] [--steps 100]
                          [--seed 0] [--no-retarget] [--no-render] [--out DIR]

Python:
    mo = text_motion("a person jumps twice", duration=3)    # Motion: joints (T, J, 3), names, parents, fps, robot
    rob = retarget(mo, KeypointIKRetargeter(xml, mapping))  # any MuJoCo humanoid: map skeleton joints to its sites

Models (open weights, NVIDIA Open Model License; code Apache-2.0): Kimodo-SOMA-RP-v1.2 (77-joint SOMA skeleton,
joint positions directly, no SMPL files needed) and Kimodo-G1-RP-v1 (Unitree G1 joint angles directly: `model="g1"`
returns MuJoCo qpos). The text encoder is LLM2Vec on Meta-Llama-3-8B-Instruct (gated on Hugging Face: accept the
Llama 3 licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and log in); it runs on the CPU in
fp32 (about 32 GB RAM). Output coordinates: metres, z up, x forward (MuJoCo convention).

Retargeting: the worker maps SOMA motion to a built-in MuJoCo humanoid by keypoint IK (`retarget=True`); in Hylo,
`retarget(motion, retargeter)` runs any `Retargeter` (`KeypointIKRetargeter` needs mujoco).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .runner import ToolResult, ToolServer, run_worker
from .workers.text_motion_retarget import (HUMANOID_XML, SOMA_TO_HUMANOID, KeypointIKRetargeter, Retargeter,
                                           RobotMotion, render_robot)

MODEL = "nvidia/Kimodo-SOMA-RP-v1.2 (+ nvidia/Kimodo-G1-RP-v1)"
LICENCE = "NVIDIA Open Model License (weights), Apache-2.0 (code); text encoder Llama 3 Community License (gated)"
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"

__all__ = ["HUMANOID_XML", "SOMA_TO_HUMANOID", "KeypointIKRetargeter", "Motion", "Retargeter", "RobotMotion",
           "render_robot", "retarget", "text_motion"]


@dataclass
class Motion:
    """joints (T, J, 3) metres, z up, x forward; names; parents (-1 for the root); fps; foot_contacts (T, 4);
    robot: the retargeted humanoid motion (worker side), or for model="g1" the G1 qpos (xml: Kimodo's G1 MJCF)."""
    joints: np.ndarray
    names: list[str]
    parents: np.ndarray
    fps: float
    foot_contacts: np.ndarray
    robot: RobotMotion | None
    result: ToolResult
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def gl_env() -> dict[str, str]:
    """MuJoCo offscreen rendering for the worker: the caller's MUJOCO_GL, else the user-space OSMesa when present."""
    if os.environ.get("MUJOCO_GL"):
        return {}
    if (USER_MESA / "libOSMesa.so.8").exists():
        return {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}
    return {}


def _parse(res: ToolResult) -> Motion:
    a = res.arrays("motion")
    robot = None
    if "robot" in res.files:
        r = res.arrays("robot")
        robot = RobotMotion(qpos=r["qpos"], fps=float(r["fps"]), xml=str(r["xml"]),
                            joint_names=[str(x) for x in r["joint_names"]], err=r["err"])
    elif "qpos" in a:
        robot = RobotMotion(qpos=a["qpos"], fps=float(a["fps"]), xml=res.outputs.get("g1_xml", ""),
                            joint_names=[str(x) for x in a.get("qpos_names", [])])
    keep = {"joints", "names", "parents", "fps", "foot_contacts"}
    return Motion(a["joints"], [str(x) for x in a["names"]], a["parents"], float(a["fps"]), a["foot_contacts"], robot,
                  res, {k: v for k, v in a.items() if k not in keep})


def text_motion(prompt: str | Sequence[str], *, duration: float | Sequence[float] = 4.0, model: str = "soma",
                steps: int = 100, seed: int = 0, postprocess: bool = True, retarget: bool = True, render: bool = True,
                server: ToolServer | None = None, **run_kw: Any) -> Motion:
    """Generate a motion. `prompt`: one text, or a list of texts for consecutive segments (with `duration` per
    segment). `model`: "soma" (human skeleton), "g1" (Unitree G1 qpos) or a Kimodo model name."""
    req = {"prompt": prompt if isinstance(prompt, str) else list(prompt),
           "duration": duration if isinstance(duration, (int, float)) else list(duration), "model": model,
           "steps": steps, "seed": seed, "postprocess": postprocess, "retarget": retarget, "render": render}
    if server:
        return _parse(server.call(req))
    env = {**gl_env(), **run_kw.pop("env", {})}
    return _parse(run_worker("text_motion", req, env=env, **run_kw))


def retarget(motion: Motion | tuple[np.ndarray, Sequence[str], float], retargeter: Retargeter | None = None) -> RobotMotion:
    """The retargeting hook: map a skeleton motion to robot joint targets. Default: keypoint IK onto the built-in
    humanoid (needs mujoco in this environment). Pass a `KeypointIKRetargeter(xml, mapping)` for another robot, or
    any callable (joints, names, fps) -> RobotMotion."""
    joints, names, fps = (motion.joints, motion.names, motion.fps) if isinstance(motion, Motion) else motion
    return (retargeter or KeypointIKRetargeter())(np.asarray(joints), list(names), float(fps))


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("text_motion", __doc__)
    ap.add_argument("prompt", nargs="+", help="one text, or several for consecutive segments")
    ap.add_argument("--duration", type=float, nargs="+", default=[4.0], help="seconds (one value, or one per prompt)")
    ap.add_argument("--model", default="soma", help="soma | g1 | soma-seed | g1-seed | a Kimodo model name")
    ap.add_argument("--steps", type=int, default=100, help="denoising steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--no-retarget", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    dur = a.duration[0] if len(a.duration) == 1 else a.duration
    prompt = a.prompt[0] if len(a.prompt) == 1 else a.prompt
    return _cli.guard(lambda: _cli.finish(text_motion(prompt, duration=dur, model=a.model, steps=a.steps, seed=a.seed,
                                                      postprocess=not a.no_postprocess, retarget=not a.no_retarget,
                                                      render=not a.no_render, **_cli.run_kw(a)).result, a.out))
