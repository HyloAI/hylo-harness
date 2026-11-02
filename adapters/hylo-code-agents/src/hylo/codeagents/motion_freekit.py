# hylo: worker-side (imported as `studio` by free-mode scripts inside the guarded runner; no Hylo imports)
"""The studio kit of free mode: what a model's script imports (`from studio import *`).

Pure helpers run in the script's own process (motions as numpy arrays, the edits of motion_ops, saving and loading
motions in the run directory). Everything that runs a model, a simulator or a renderer is a request to the studio
process over a pipe: the studio checks it (allowed tool, paths inside the run directory, budget), runs it, logs it
in log.jsonl and answers with the result. A refused or failed request raises ToolError in the script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ops: Any = None                 # motion_ops, set by the runner before the script starts (bind)
# motion_ops names re-exported for scripts; bound by bind()
Motion = NAMES = IDX = PARENTS = concat = blend = loop = mirror = retime = trim = hold = keyframe = None
canonical = heading = None
RUN = Path.cwd()
_n = [0]


class ToolError(RuntimeError):
    """A studio request that was refused or failed."""


def _channel():
    spec = os.environ.get("HYLO_STUDIO_RPC", "")
    if not spec:
        raise ToolError("no studio connection (run this script with `hylo motion --free`)")
    w, r = (int(x) for x in spec.split(","))
    if not hasattr(_channel, "io"):
        _channel.io = (os.fdopen(w, "w", buffering=1), os.fdopen(r, "r"))
    return _channel.io


def _call(op: str, **kw: Any) -> Any:
    wf, rf = _channel()
    wf.write(json.dumps({"op": op, "args": kw}, default=_jsonable) + "\n")
    wf.flush()
    line = rf.readline()
    if not line:
        raise ToolError("the studio closed the connection")
    res = json.loads(line)
    if "error" in res:
        raise ToolError(res["error"])
    return res.get("result")


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if ops is not None and isinstance(v, ops.Motion):
        return save_motion(v)
    raise TypeError(f"cannot send {type(v).__name__} to the studio")


# ------------------------------------------------------------------------------------------------- motions
def save_motion(m: Any, path: str | None = None) -> str:
    """Write a Motion to an .npz in the run directory (default work/motions/NNN.npz); returns the path."""
    if path is None:
        _n[0] += 1
        path = f"work/motions/{os.getpid()}-{_n[0]:03d}.npz"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, joints=np.asarray(m.joints, np.float32), fps=np.float32(m.fps), source=np.array(str(m.source)))
    return str(p)


def load_motion(path: str) -> Any:
    """A Motion from an .npz written by save_motion or a studio tool."""
    z = np.load(path)
    return ops.Motion(np.asarray(z["joints"], float), float(z["fps"]), str(z["source"]) if "source" in z else "")


def motion_from_array(joints: Any, fps: float = 30.0, source: str = "array") -> Any:
    """Your own motion: joints (T, 17, 3) in NAMES order, metres, z up, ground at z = 0."""
    J = np.asarray(joints, float)
    if J.ndim != 3 or J.shape[1:] != (len(ops.NAMES), 3):
        raise ValueError(f"joints must be (T, {len(ops.NAMES)}, 3)")
    return ops.Motion(J, float(fps), source)


def _m(res: Any) -> Any:
    return load_motion(res["motion"])


# ------------------------------------------------------------------------------------------------- sources
def text_motion(prompt: Any, seconds: Any = 4.0, seed: int = 0) -> Any:
    """Text -> Motion (NVIDIA Kimodo; minutes on this CPU). A list of prompts with a list of seconds makes one
    continuous motion of consecutive segments."""
    return _m(_call("text_motion", prompt=prompt, seconds=seconds, seed=seed))


def video_motion(path: str, start: float = 0.0, seconds: float | None = None) -> Any:
    """A video of one person -> Motion (the `human` tool: 3D pose per frame)."""
    return _m(_call("video_motion", path=path, start=start, seconds=seconds))


def pose_from_image(path: str) -> Any:
    """A photo of a person -> a one-frame Motion (their pose)."""
    return _m(_call("pose_from_image", path=path))


def tool(name: str, **kwargs: Any) -> dict:
    """Any other allowed hylo tool by name (hands, depth, segment, human, ...): kwargs as the tool's Python API
    (paths relative to the run directory). Returns {"outputs", "files"}; files are in the run directory."""
    return _call("tool", name=name, kwargs=kwargs)


def hands(video: str, **kw: Any) -> dict:
    """Video -> 3D hand keypoints per frame (the `hands` tool)."""
    return tool("hands", video=video, **kw)


def depth(image: str, **kw: Any) -> dict:
    """Image -> metric depth map, points and intrinsics (the `depth` tool)."""
    return tool("depth", image=image, **kw)


def segment(image: str, prompts: Any, **kw: Any) -> dict:
    """Image + text prompts -> instance masks (the `segment` tool)."""
    return tool("segment", image=image, prompts=prompts, **kw)


def human(video: str, **kw: Any) -> dict:
    """The raw `human` tool: per-frame 33 3D body landmarks and validity (video_motion converts these)."""
    return tool("human", video=video, **kw)


# ------------------------------------------------------------------------------------------------- characters
def characters() -> list[dict]:
    """The target characters: name, title, kind ("mjcf": a robot or simulated humanoid; "rig": a skinned human
    character rendered in Blender) and height."""
    return _call("characters")


def retarget(m: Any, character: str) -> dict:
    """Motion -> the character's motion; returns the retargeting metrics (keypoint error, limb angle error,
    joint-limit hits)."""
    return _call("retarget", motion=m, character=character)


def check(m: Any, character: str) -> dict:
    """Plausibility on a character: {"problems": [...], "kinematics": {...}}."""
    return _call("check", motion=m, character=character)


def render(m: Any, character: str = "g1", camera: str = "track") -> str:
    """Render the motion on one character (Blender Cycles; minutes). Returns the MP4 path; use look(path)."""
    return _call("render", motion=m, character=character, camera=camera)


def lineup(m: Any, characters: list[str] | None = None) -> str:
    """The motion on several characters: one render each, composed into a labelled grid. Returns the MP4 path."""
    return _call("lineup", motion=m, characters=characters)


def look(path: str, n: int = 4, note: str = "") -> str:
    """Show yourself n frames of a video (or an image) in your next feedback, as one contact sheet."""
    return _call("look", path=path, n=n, note=note)


def finish(m: Any, characters: list[str] | None = None, note: str = "") -> str:
    """The final motion and the characters to show it on (the studio checks and renders them); ends the session."""
    return _call("finish", motion=m, characters=characters, note=note)


def bind(motion_ops: Any) -> None:
    """Called by the runner: make motion_ops (Motion, NAMES, the edits) available as `ops` and by name."""
    global ops
    ops = motion_ops
    g = globals()
    for name in EDITS:
        g[name] = getattr(motion_ops, name)


EDITS = ["Motion", "NAMES", "IDX", "PARENTS", "concat", "blend", "loop", "mirror", "retime", "trim", "hold", "keyframe",
         "canonical", "heading"]                     # from motion_ops, bound by the runner (bind)
__all__ = ["ToolError", "check", "characters", "depth", "finish", "hands", "human", "lineup", "load_motion", "look",
           "motion_from_array", "np", "ops", "pose_from_image", "render", "retarget", "save_motion", "segment",
           "text_motion", "tool", "video_motion", "Motion", "NAMES", "IDX", "PARENTS", "concat", "blend", "loop",
           "mirror", "retime", "trim", "hold", "keyframe", "canonical", "heading"]
