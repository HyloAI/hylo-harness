"""Shared helpers of the motion-studio tests: the workers loaded by path, synthetic skeleton motions and a fake
tool backend (synthetic text and video motion, placeholder renders, the MuJoCo worker in-process)."""

from __future__ import annotations

import importlib.util

import numpy as np

from hylo.codeagents import motion_ops as ops
from hylo.codeagents.motion_tools import MEDIA_WORKER, SIM_WORKER


def load_worker(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_worker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ms = load_worker(SIM_WORKER)
media = load_worker(MEDIA_WORKER)

# a standing skeleton (metres, z up, facing +x), SOMA-like proportions
REST = {"Hips": (0, 0, 0.95), "Chest": (0, 0, 1.15), "Head": (0, 0, 1.55), "LeftLeg": (0, 0.09, 0.88),
        "LeftShin": (0.01, 0.1, 0.48), "LeftFoot": (0, 0.1, 0.08), "LeftToeBase": (0.13, 0.11, 0.02),
        "LeftArm": (0, 0.18, 1.38), "LeftForeArm": (0, 0.22, 1.1), "LeftHand": (0.02, 0.24, 0.84)}
REST.update({k.replace("Left", "Right"): (x, -y, z) for k, (x, y, z) in list(REST.items()) if k.startswith("Left")})


def skeleton(seconds: float = 2.0, fps: float = 30.0, walk: float = 0.0, wave: bool = True) -> np.ndarray:
    """(T, 17, 3): standing (optionally gliding forward at `walk` m/s), the right hand waving above the head."""
    T = int(seconds * fps)
    base = np.array([REST[n] for n in ops.NAMES], float)
    J = np.repeat(base[None], T, 0)
    t = np.arange(T) / fps
    J[..., 0] += (walk * t)[:, None]
    if wave:
        r = ops.IDX["RightHand"], ops.IDX["RightForeArm"]
        J[:, r[1]] = J[:, ops.IDX["RightArm"]] + [0.0, -0.12, 0.22]
        J[:, r[0]] = J[:, r[1]] + np.stack([0 * t, 0.12 * np.sin(2 * np.pi * t), 0.22 + 0 * t], 1)
    return J


def motion(**kw) -> ops.Motion:
    return ops.canonical(ops.Motion(skeleton(**kw), 30.0, "test"))


class FakeBackend:
    """Stub tool workers: synthetic text motion and video poses, renders as placeholder files; motion_sim runs
    in-process."""

    def __init__(self):
        self.calls = []

    def text_motion(self, prompt, seconds, seed, out):
        self.calls.append(("text_motion", prompt, seconds))
        secs = sum(seconds) if isinstance(seconds, list) else seconds
        J = skeleton(seconds=float(secs), walk=0.3 if "walk" in str(prompt) else 0.0)
        return J, list(ops.NAMES), 30.0, {"model": "fake"}

    def human(self, frames, fps, out):
        self.calls.append(("human", str(frames)))
        J = skeleton(seconds=1.0)
        W = np.zeros((len(J), 33, 3))
        for i, j in ops.MEDIAPIPE.items():
            for k in j:
                W[:, k] = J[:, ops.IDX[i]]
        W[:, [11, 12]] = J[:, [ops.IDX["LeftArm"], ops.IDX["RightArm"]]]
        return W, np.ones(len(J), bool), fps, {"backend": "fake"}, None

    def render(self, scene, qpos, fps, out, *, camera, style, distance):
        self.calls.append(("render", len(qpos)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "video.mp4").write_bytes(b"fake mp4")
        return {"video": str(out / "video.mp4"), "still": None, "backend": "fake"}

    def sim(self, op, params, inputs, out):
        self.calls.append(("sim", op))
        out.mkdir(parents=True, exist_ok=True)
        if op == "clip":
            np.save(out / "frames.npy", np.zeros((3, 8, 8, 3), np.uint8))
            return {"ok": True, "outputs": {"fps": 30.0, "frames": 3}, "files": {"frames": str(out / "frames.npy")}}
        if op == "compose":
            (out / "split.mp4").write_bytes(b"fake split")
            return {"ok": True, "outputs": {}, "files": {"video": str(out / "split.mp4")}}
        if op == "rig":                                    # a rigged character: its joints are the keypoints
            np.save(out / "joints.npy", np.load(params["keypoints"]))
            files = {"joints": str(out / "joints.npy")}
            if params.get("render"):
                (out / "video.mp4").write_bytes(b"fake rig video")
                files["video"] = str(out / "video.mp4")
            return {"ok": True, "outputs": {"metrics": {"mean_keypoint_err_m": 0.0, "limb_angle_err_deg": 0.0,
                                                        "limit_hit_frac": 0.0}}, "files": files}
        return ms.handle(op, params, inputs, out)

    def tool(self, name, kwargs, out):
        self.calls.append(("tool", name))
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.npz").write_bytes(b"fake")
        return {"outputs": {"tool": name}, "files": {"result": str(out / "result.npz")}}

    def close(self):
        self.calls.append(("close",))


def block(code: str) -> str:
    return f"Plan.\n```python\n{code}\n```"




def stepping(seconds: float = 2.0, fps: float = 30.0, speed: float = 0.5, lift: float = 0.08) -> np.ndarray:
    """(T, 17, 3): walking in place forward at `speed` m/s with alternating foot lifts (one step per 0.5 s)."""
    J = skeleton(seconds, fps, walk=0.0, wave=False)
    T = len(J)
    t = np.arange(T) / fps
    J[..., 0] += (speed * t)[:, None]
    for side, phase in (("Left", 0.0), ("Right", 0.5)):
        up = np.clip(np.sin(2 * np.pi * (t - phase)), 0, None) * lift
        for n in ("Foot", "ToeBase", "Shin"):
            J[:, ops.IDX[side + n], 2] += up * (0.5 if n == "Shin" else 1.0)
    return J
