"""Recordings for the replay-to-sim workflow (RFC-0006 part 8): one camera's video plus whatever the source knows.

A `Recording` is the common input of every replay step: RGB frames and times, and when the source has them depth,
intrinsics, the camera pose, the robot's joint positions, TCP poses (hylo.core.robot convention: +x approach, +y
closing axis, origin between the pads) and gripper opening (1 open, 0 closed), the task text, and for our own
simulated episodes ground truth (object poses and sizes, segmentation, success) used only for evaluation.

Sources (`load_source`):
    DIR                     a saved recording (DIR/recording.npz + DIR/recording.json)
    panda:TASK:SEED         record an episode now: mujoco.panda_table with its privileged oracle, RGB-D camera on
    hf:REPO:EPISODE         one episode of a LeRobot dataset on Hugging Face (fetched by replay_fetch.py in its own uv
                            environment), converted with a profile: libero, droid or video (camera images only)
    FILE.mp4 | FRAME_DIR    a video (e.g. a human hand demonstration); the hand becomes a parallel gripper later

Profiles say how a dataset's state maps to the robot: LIBERO (robosuite Panda: eef position, axis-angle, two finger
positions; world frame with the robot base at about (-0.66, 0, 0.912)), DROID (Franka: end-effector position and
XYZ Euler angles in the robot base frame, gripper position 0 open .. 1 closed).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FETCH_SCRIPT = Path(__file__).with_name("replay_fetch.py")
# the TCP in a robosuite / Franka end-effector frame whose z axis is the approach direction: +x approach = ee z,
# +y closing = ee y, +z = ee -x (same convention as mujoco.panda_table's hand)
EE_Z_APPROACH = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


@dataclass
class Recording:
    """One recorded episode. Arrays are per frame (T). Poses are 4x4 in the replay world (metres, +z up), except
    where `meta["frame"]` says the robot data is in its own base frame (external datasets before registration)."""
    name: str
    rgb: np.ndarray                                # (T, H, W, 3) uint8
    time: np.ndarray                               # (T,) seconds
    instruction: str = ""
    agent: str = "panda"                           # panda | hand | none
    depth: np.ndarray | None = None                # (T, H, W) metres, 0 = no return
    K: np.ndarray | None = None                    # (3, 3)
    T_world_cam: np.ndarray | None = None          # (4, 4) OpenCV optical frame
    q: np.ndarray | None = None                    # (T, 7) arm joints
    tcp: np.ndarray | None = None                  # (T, 4, 4)
    gripper: np.ndarray | None = None              # (T,) measured opening, 1 open .. 0 closed
    gripper_cmd: np.ndarray | None = None          # (T,) commanded, when known
    base: np.ndarray | None = None                 # (4, 4) robot base in the world
    meta: dict[str, Any] = field(default_factory=dict)
    gt: dict[str, Any] | None = None               # objects {name: label, kind, half_size, poses (T, 4, 4)}, seg, success

    @property
    def T(self) -> int:
        return int(len(self.rgb))

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps") or (1.0 / np.median(np.diff(self.time)) if self.T > 1 else 10.0))

    def subsample(self, step: int, max_frames: int | None = None) -> "Recording":
        """Every `step`-th frame (and at most max_frames): the copy shares no arrays with self."""
        idx = np.arange(0, self.T, max(1, int(step)))
        if max_frames:
            idx = idx[:max_frames]

        def pick(a: Any) -> Any:
            return None if a is None else np.asarray(a)[idx].copy()

        gt = None
        if self.gt is not None:
            gt = dict(self.gt)
            gt["objects"] = {k: {**v, "poses": np.asarray(v["poses"])[idx]} for k, v in self.gt["objects"].items()}
            if gt.get("seg") is not None:
                gt["seg"] = np.asarray(gt["seg"])[idx]
        meta = {**self.meta, "fps": self.fps / max(1, int(step)), "subsample": int(step)}
        return Recording(self.name, pick(self.rgb), pick(self.time), self.instruction, self.agent, pick(self.depth),
                         self.K, self.T_world_cam, pick(self.q), pick(self.tcp), pick(self.gripper),
                         pick(self.gripper_cmd), self.base, meta, gt)

    # -- persistence --------------------------------------------------------------------------------------------
    def save(self, out: str | Path) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"rgb": self.rgb, "time": self.time}
        for k in ("depth", "K", "T_world_cam", "q", "tcp", "gripper", "gripper_cmd", "base"):
            v = getattr(self, k)
            if v is not None:
                arrays[k] = np.asarray(v, np.float32 if k == "depth" else np.float64)
        gt_meta = None
        if self.gt is not None:
            gt_meta = {"objects": {}, "success": self.gt.get("success"), "segments": self.gt.get("segments", {})}
            for name, o in self.gt["objects"].items():
                arrays[f"gt_pose_{name}"] = np.asarray(o["poses"], np.float64)
                gt_meta["objects"][name] = {k: v for k, v in o.items() if k != "poses"}
            if self.gt.get("seg") is not None:
                arrays["gt_seg"] = np.asarray(self.gt["seg"], np.int16)
        np.savez_compressed(out / "recording.npz", **arrays)
        (out / "recording.json").write_text(json.dumps({
            "name": self.name, "instruction": self.instruction, "agent": self.agent, "meta": self.meta,
            "gt": gt_meta}, indent=2, default=_json))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        path = Path(path)
        head = json.loads((path / "recording.json").read_text())
        with np.load(path / "recording.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        gt = None
        if head.get("gt"):
            g = head["gt"]
            gt = {"objects": {n: {**o, "poses": a[f"gt_pose_{n}"]} for n, o in g["objects"].items()},
                  "success": g.get("success"), "segments": g.get("segments", {}), "seg": a.get("gt_seg")}
        return cls(head["name"], a["rgb"], a["time"], head.get("instruction", ""), head.get("agent", "panda"),
                   a.get("depth"), a.get("K"), a.get("T_world_cam"), a.get("q"), a.get("tcp"), a.get("gripper"),
                   a.get("gripper_cmd"), a.get("base"), head.get("meta", {}), gt)


def _json(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


# ---------------------------------------------------------------------------------------------------- rotations
def axis_angle_to_R(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def euler_xyz_to_R(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic X, then Y, then Z (R = Rz Ry Rx), as DROID stores end-effector orientation."""
    r, p, y = (float(x) for x in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose(R: np.ndarray | None = None, p: Any = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


# ---------------------------------------------------------------------------------------------------- profiles
def _libero(state: np.ndarray) -> dict[str, np.ndarray]:
    """robosuite Panda: eef pos (3), axis-angle (3), finger joint positions (2: +0.04 / -0.04 open)."""
    tcp = np.stack([pose(axis_angle_to_R(s[3:6]) @ EE_Z_APPROACH, s[:3]) for s in state])
    opening = np.clip((state[:, 6] - state[:, 7]) / 0.08, 0.0, 1.0)
    return {"tcp": tcp, "gripper": opening}


def _droid(state: np.ndarray) -> dict[str, np.ndarray]:
    """Franka (DROID): end-effector position (3) and XYZ Euler angles (3) in the base frame, gripper 0 open .. 1."""
    tcp = np.stack([pose(euler_xyz_to_R(s[3:6]) @ EE_Z_APPROACH, s[:3]) for s in state])
    return {"tcp": tcp, "gripper": np.clip(1.0 - state[:, 6], 0.0, 1.0)}


PROFILES: dict[str, dict[str, Any]] = {
    "libero": {"convert": _libero, "agent": "panda", "rotate180": False, "frame": "robot_world",
               "base": pose(None, [-0.66, 0.0, 0.912]), "max_opening": 0.08,
               "notes": "LIBERO (robosuite) Panda; eef pose in the LIBERO world frame; base pose fitted by IK; images "
                        "of physical-intelligence/libero are already upright (pass rotate180 for raw LIBERO renders)"},
    "droid": {"convert": _droid, "agent": "panda", "rotate180": False, "frame": "robot_base", "base": np.eye(4),
              "max_opening": 0.085, "notes": "DROID Franka with a Robotiq 2F-85; end-effector pose in the base frame"},
    "video": {"convert": None, "agent": "hand", "rotate180": False, "frame": "camera", "base": None,
              "max_opening": 0.08, "notes": "camera images only; a human hand is tracked and retargeted"},
}


# ---------------------------------------------------------------------------------------------------- sources
def fetch_lerobot(repo: str, episode: int, camera: str, out: str | Path, *, rotate180: bool = False,
                  fps: float | None = None, max_frames: int | None = None, timeout: float = 3600) -> Path:
    """Run replay_fetch.py in its own uv environment; returns the folder with recording.npz and source.json."""
    out = Path(out)
    if (out / "source.json").is_file() and (out / "recording.npz").is_file():
        return out
    uv = os.environ.get("HYLO_TOOLS_UV") or shutil.which("uv") or "uv"
    cmd = [uv, "run", "--quiet", "--script", str(FETCH_SCRIPT), repo, str(episode), "--camera", camera, "--out", str(out)]
    if rotate180:
        cmd.append("--rotate180")
    if fps:
        cmd += ["--fps", str(fps)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"fetching {repo} episode {episode} failed:\n{proc.stderr[-2000:]}")
    return out


def from_fetched(folder: str | Path, profile: str, name: str | None = None) -> Recording:
    folder = Path(folder)
    src = json.loads((folder / "source.json").read_text())
    with np.load(folder / "recording.npz") as z:
        a = {k: z[k] for k in z.files}
    prof = PROFILES[profile]
    conv = prof["convert"](a["state"]) if prof["convert"] is not None and "state" in a else {}
    meta = {"source": f"hf:{src['repo']}:{src['episode']}", "repo": src["repo"], "revision": src.get("revision"),
            "episode": src["episode"], "camera": src["camera"], "licence": src.get("licence"), "fps": src["fps"],
            "profile": profile, "frame": prof["frame"], "notes": prof["notes"], "max_opening": prof["max_opening"]}
    return Recording(name or f"{profile}-{src['episode']}", a["rgb"], a["time"], src.get("task", ""), prof["agent"],
                     tcp=conv.get("tcp"), gripper=conv.get("gripper"),
                     base=None if prof["base"] is None else np.asarray(prof["base"], float), meta=meta)


def from_video(path: str | Path, *, fps: float | None = None, max_frames: int | None = None, size: int | None = 480,
               instruction: str = "", licence: str = "unknown") -> Recording:
    """A video file or a folder of images (a human demonstration)."""
    from PIL import Image

    path = Path(path)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
        src_fps = fps or 15.0
    else:
        import imageio.v3 as iio

        meta = iio.immeta(path)
        src_fps = float(meta.get("fps") or 30.0)
        frames = [np.asarray(f)[..., :3] for f in iio.imiter(path)]
    step = max(1, int(round(src_fps / fps))) if fps else 1
    frames = frames[::step][:max_frames]
    if size and max(frames[0].shape[:2]) > size:
        s = size / max(frames[0].shape[:2])
        wh = (int(round(frames[0].shape[1] * s)), int(round(frames[0].shape[0] * s)))
        frames = [np.asarray(Image.fromarray(f).resize(wh, Image.BILINEAR)) for f in frames]
    rgb = np.stack(frames)
    t = np.arange(len(rgb)) * step / src_fps
    return Recording(path.stem, rgb, t, instruction, "hand",
                     meta={"source": str(path), "fps": src_fps / step, "licence": licence, "profile": "video",
                           "frame": "camera", "max_opening": 0.08})


# ---------------------------------------------------------------------------------------------------- our episodes
def record_panda(task: str, seed: int, *, width: int = 400, height: int = 300, camera: str = "front", every: int = 2,
                 max_steps: int | None = None, extra_steps: int = 10) -> Recording:
    """Record one `mujoco.panda_table` episode solved by `oracle.panda_table` (privileged): the camera's RGB-D and
    segmentation, joint positions, measured and commanded hand, TCP, and ground-truth object poses every `every`
    control steps. Needs an OpenGL backend for the camera (MUJOCO_GL=osmesa on a CPU)."""
    from hylo.core.agent import EpisodeContext
    from hylo.core.registry import registry
    from hylo.core.task import TaskInstance, TaskSpec
    from hylo.core.types import Action

    env = registry.build("envs", {"type": "mujoco.panda_table", "params": {
        "cameras": [camera], "width": width, "height": height, "segmentation": True, "privileged_obs": True}})
    agent = registry.build("agents", {"type": "oracle.panda_table", "params": {}})
    spec = env.spec()
    steps = int(max_steps or 900)
    ts = TaskSpec(id=f"mujoco/manip_{task}", version="1", instruction=task, params={"task": task}, max_steps=steps)
    inst = TaskInstance(spec=ts, seed=int(seed))
    obs = env.reset(inst)
    agent.reset(EpisodeContext(episode_id=f"replay-{task}-{seed}", task=inst, env_spec=spec, emit=lambda *a: None,
                               seed=int(seed)))
    rows: list[dict[str, Any]] = []
    t, success, done_at = 0, False, None
    cmd = np.concatenate([obs.state["q.arm"], [1.0]])

    def snap(o: Any, command: np.ndarray) -> None:
        p = o.extra["privileged"]
        rows.append({"rgb": o.images[camera], "depth": o.images[f"{camera}.depth"], "seg": o.images[f"{camera}.seg"],
                     "q": np.asarray(o.state["q.arm"], float), "hand": float(o.state["hand.arm"][0]),
                     "cmd": float(command[7]), "tcp": np.asarray(p["tcp"], float), "objects": p["objects"],
                     "time": float(o.time)})

    snap(obs, cmd)
    while t < steps and (done_at is None or t < done_at + extra_steps):
        chunk = agent.act(obs)
        for a in np.asarray(chunk.actions):
            res = env.step(Action(data=a))
            obs, t, cmd = res.observation, t + 1, a
            if t % every == 0:
                snap(obs, cmd)
            if res.success and done_at is None:
                success, done_at = True, t
            if (res.terminated or res.truncated) and not res.success:
                done_at = done_at or t
            if t >= steps or (done_at is not None and t >= done_at + extra_steps):
                break
    info = {c: env.camera_info(c) for c in [camera]}[camera]
    priv = obs.extra["privileged"]
    segments = {int(k): v for k, v in obs.extra.get("segments", {}).items()}
    objects = {}
    for name, o in rows[0]["objects"].items():
        poses = []
        for r in rows:
            ob = r["objects"][name]
            poses.append(pose(_quat_R(ob["quat"]), ob["pos"]))
        objects[name] = {"label": o["label"], "kind": o["kind"], "half_size": o["half_size"],
                         "instance": o["instance"], "poses": np.stack(poses)}
    robot = spec.robot
    arm = robot.arms[0]
    env.close()
    return Recording(
        f"panda-{task}-{seed}", np.stack([r["rgb"] for r in rows]), np.array([r["time"] for r in rows]),
        priv.get("instruction") or obs.instruction or task, "panda",
        depth=np.stack([r["depth"] for r in rows]).astype(np.float32), K=np.asarray(info["K"], float),
        T_world_cam=np.asarray(info["T_world_cam"], float), q=np.stack([r["q"] for r in rows]),
        tcp=np.stack([r["tcp"] for r in rows]), gripper=np.array([r["hand"] for r in rows]),
        gripper_cmd=np.array([r["cmd"] for r in rows]), base=np.asarray(arm.base_in_world, float),
        meta={"source": f"panda:{task}:{seed}", "env": "mujoco.panda_table", "agent": "oracle.panda_table",
              "privileged_oracle": True, "fps": spec.control_hz / every, "camera": camera, "licence": "ours (Apache-2.0)",
              "task": task, "seed": int(seed), "success": bool(success), "steps": int(t), "frame": "world",
              "max_opening": float(arm.hand.max_width), "table_height": float(robot.table_height)},
        gt={"objects": objects, "seg": np.stack([r["seg"] for r in rows]).astype(np.int16),
            "segments": {str(k): v for k, v in segments.items()}, "success": bool(success)})


def _quat_R(q: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_source(source: str, *, work: str | Path, profile: str | None = None, camera: str | None = None,
                fps: float | None = None, max_frames: int | None = None) -> Recording:
    """A Recording from a source string (see the module doc)."""
    if source.startswith("panda:"):
        _, task, seed = source.split(":")
        cached = Path(work) / "recording"
        if (cached / "recording.npz").is_file():
            return Recording.load(cached)
        rec = record_panda(task, int(seed))
        rec.save(cached)
        return rec
    if source.startswith("hf:"):
        _, rest = source.split(":", 1)
        repo, _, ep = rest.rpartition(":")
        prof = profile or ("libero" if "libero" in repo.lower() else "droid" if "droid" in repo.lower() else "video")
        if not camera:
            raise ValueError("hf: sources need --camera (the dataset's image key)")
        folder = fetch_lerobot(repo, int(ep), camera, Path(work) / "fetched", rotate180=PROFILES[prof]["rotate180"],
                               fps=fps, max_frames=max_frames)
        return from_fetched(folder, prof)
    p = Path(source)
    if p.is_dir() and (p / "recording.json").is_file():
        return Recording.load(p)
    if p.is_dir() and (p / "source.json").is_file():
        return from_fetched(p, profile or "video")
    if p.exists():
        return from_video(p, fps=fps, max_frames=max_frames)
    raise FileNotFoundError(f"replay source {source!r}: not a recording, panda:TASK:SEED, hf:REPO:EP or a video")


def main(argv: list[str] | None = None) -> int:
    """python -m hylo.codeagents.replay_data record TASK SEED OUT  (records one of our Panda episodes)"""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4 or args[0] != "record":
        print(main.__doc__, file=sys.stderr)
        return 2
    rec = record_panda(args[1], int(args[2]))
    rec.save(args[3])
    print(json.dumps({"out": args[3], "frames": rec.T, "success": rec.meta["success"], "steps": rec.meta["steps"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EE_Z_APPROACH", "PROFILES", "Recording", "axis_angle_to_R", "euler_xyz_to_R", "fetch_lerobot",
           "from_fetched", "from_video", "load_source", "pose", "record_panda"]
