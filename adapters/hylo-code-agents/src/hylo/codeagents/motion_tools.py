"""The motion studio's tool layer (RFC-0006 part 8), shared by the guided studio (motion.py) and free mode
(motion_free.py): the model-facing tools, the backend that runs them, and the final report of a run.

- RobotMotion: a motion retargeted to one character (robot qpos, or a rigged character's posed keypoints).
- ToolBackend: the real tools, each in its own environment: hylo-tools workers (text_motion kept loaded, human,
  render and the perception tools), the MuJoCo worker (motion_sim: retarget, check, validate) and the media worker
  (motion_media: clip, compose, rig). Tests pass a fake with the same methods.
- StudioTools: what the model calls: text_motion, video_motion, pose_from_image, retarget (any character of
  motion_characters), check, render (robots through the hylo-tools render tool, rigged characters through
  Blender), lineup (the same motion on several characters, one labelled video grid), look (frames of a video for
  the model), finish. Every call is logged and timed; slow calls pause the sandbox's block clock.
- finalize: the final motion -> motion.npz, check.json, a report per character (characters.json) and the videos.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import motion_characters as mc
from . import motion_ops as ops

DEFAULT_MODEL: dict[str, Any] = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True,
                                 "reasoning_effort": "none", "max_tokens": 3000, "timeout_s": 900}
SIM_WORKER = Path(__file__).with_name("motion_sim.py")
MEDIA_WORKER = Path(__file__).with_name("motion_media.py")
WORKER_OPS = {"retarget": SIM_WORKER, "check": SIM_WORKER, "validate": SIM_WORKER, "clip": MEDIA_WORKER,
              "compose": MEDIA_WORKER, "rig": MEDIA_WORKER}
RENDER_VIEW = {"azimuth": 150.0, "elevation": -12.0}     # the camera of every single-character render
ANKLE_GAIN = 1.0          # the check's second rollout: PD tracking plus an ankle strategy (see motion_sim.rollout)


def robot_spec(robot: str, work: Path) -> dict[str, Any]:
    """What motion_sim needs to know about an MJCF character (see its docstring and motion_characters)."""
    if robot not in mc.names("mjcf"):
        raise ValueError(f"unknown robot {robot!r} ({', '.join(mc.names('mjcf'))}; rigged characters: "
                         f"{', '.join(mc.names('rig'))})")
    return mc.mjcf_spec(robot, work)


@dataclass
class RobotMotion:
    """A motion retargeted to a robot: qpos (T, nq) for its scene (scene.xml), at fps."""
    qpos: np.ndarray
    fps: float
    robot: str
    scene: str
    source: ops.Motion
    info: dict[str, Any] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=list)
    kind: str = "mjcf"         # "rig": qpos holds the character's posed keypoints (T, 17, 3), scene is unused

    @property
    def seconds(self) -> float:
        return round(len(self.qpos) / self.fps, 3)

    def summary(self) -> dict[str, Any]:
        """The source motion's summary plus the retargeting error."""
        return {**self.source.summary(), "robot": self.robot,
                "mean_keypoint_err_m": self.info.get("mean_keypoint_err_m"),
                "limb_angle_err_deg": self.info.get("limb_angle_err_deg"),
                "limit_hit_frac": self.info.get("limit_hit_frac")}

    def __repr__(self) -> str:
        return (f"RobotMotion({self.robot}: {self.seconds} s, keypoint error "
                f"{self.info.get('mean_keypoint_err_m')} m, from {self.source.source})")


# ------------------------------------------------------------------------------------------------- backends
class ToolBackend:
    """The real tools: hylo-tools workers (text_motion kept loaded in a ToolServer, human, render, perception
    tools) and the studio's workers (motion_sim, motion_media), each in its own uv environment."""

    def __init__(self, threads: int = 64, steps: int = 100) -> None:
        self.threads, self.steps = threads, steps
        self._server: Any = None

    def text_motion(self, prompt: Any, seconds: Any, seed: int, out: Path) -> tuple[np.ndarray, list[str], float, dict]:
        from hylo.tools.runner import ToolServer
        from hylo.tools.text_motion import text_motion
        if self._server is None:
            self._server = ToolServer("text_motion", threads=self.threads, workdir=out.parent / "text_motion_worker")
            self._server.start()
        mo = text_motion(prompt, duration=seconds, seed=seed, steps=self.steps, retarget=False, render=False,
                         server=self._server)
        return mo.joints, mo.names, mo.fps, {"timing": mo.result.timing, "model": mo.result.provenance.get("model")}

    def human(self, frames: Path, fps: float, out: Path) -> tuple[np.ndarray, np.ndarray, float, dict, str | None]:
        from hylo.tools.human import human
        hm = human(str(frames), fps=fps, workdir=out, threads=self.threads)
        return hm.joints_world, hm.valid, hm.fps, {"timing": hm.result.timing, "backend": hm.backend,
                                                   "fallback": hm.fallback_reason}, \
            str(hm.result.files.get("overlay")) if hm.result.files.get("overlay") else None

    def render(self, scene: str, qpos: np.ndarray, fps: float, out: Path, *, camera: str, style: str,
               distance: float) -> dict:
        from hylo.tools.render import render
        r = render(scene, {"qpos": qpos, "fps": fps}, backend=style, camera=camera, distance=distance,
                   elevation=RENDER_VIEW["elevation"], azimuth=RENDER_VIEW["azimuth"], width=640, height=360,
                   samples=16, fps=fps,
                   max_frames=max(300, len(qpos)), workdir=out, threads=self.threads)
        return {"video": str(r.video), "still": str(r.still), "backend": r.backend, "fallback": r.fallback_reason,
                "timing": r.result.timing}

    def tool(self, name: str, kwargs: dict, out: Path) -> dict:
        """Another hylo tool (hands, depth, segment, ...) through its Python API; returns its outputs and files."""
        import importlib
        from hylo.tools import TOOLS
        if name not in TOOLS:
            raise ValueError(f"unknown tool {name!r}")
        mod = importlib.import_module(TOOLS[name])
        fn = getattr(mod, name)
        res = fn(**kwargs, workdir=out, threads=self.threads)
        tr = getattr(res, "result", res)
        return {"outputs": getattr(tr, "outputs", {}), "files": {k: str(v) for k, v in getattr(tr, "files", {}).items()},
                "timing": getattr(tr, "timing", {})}

    def sim(self, op: str, params: dict, inputs: dict, out: Path) -> dict:
        """One op of the studio's workers (motion_sim: retarget, check, validate; motion_media: clip, compose,
        rig), each run with `uv run --script` in its own environment."""
        if op == "rig":
            params = {"render_tool": str(render_tool_dir()), "threads": self.threads, **params}
        out.mkdir(parents=True, exist_ok=True)
        req = out / "request.json"
        req.write_text(json.dumps({"op": op, "params": params, "inputs": inputs, "out_dir": str(out)}))
        uv = os.environ.get("HYLO_TOOLS_UV") or shutil.which("uv") or "uv"
        from hylo.tools.runner import worker_env
        proc = subprocess.run([uv, "run", "--quiet", "--script", str(WORKER_OPS[op]), str(req)], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=worker_env(self.threads))
        res_path = out / "result.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"ok": False, "error": proc.stdout[-1500:]}
        if not res.get("ok"):
            raise RuntimeError(f"{WORKER_OPS[op].stem} {op}: {res.get('error')}")
        return res

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None


def render_tool_dir() -> Path:
    """The hylo-tools render tool's worker scripts (its Blender locator, video writer and Blender look)."""
    import hylo.tools.render as render_tool
    return Path(render_tool.__file__).resolve().parent / "workers"


def contact_sheet(path: Path, n: int = 4, tile: tuple[int, int] = (480, 270)) -> np.ndarray:
    """An RGB image of n evenly spaced frames of a video (time stamps drawn), or the image at `path` itself."""
    from PIL import Image, ImageDraw
    n = int(np.clip(n, 1, 8))
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        frames, times = [np.asarray(Image.open(path).convert("RGB"))], [None]
    else:
        import imageio.v2 as imageio
        r = imageio.get_reader(str(path))
        fps = float(r.get_meta_data().get("fps", 30) or 30)
        allf = [f for f in r]
        r.close()
        if not allf:
            raise ValueError(f"{path.name} has no frames")
        ids = np.linspace(0, len(allf) - 1, n).round().astype(int)
        frames, times = [allf[i][..., :3] for i in ids], [i / fps for i in ids]
    tiles = []
    for f, tm in zip(frames, times):
        im = Image.fromarray(f)
        im.thumbnail(tile)
        if tm is not None:
            ImageDraw.Draw(im).text((6, 4), f"t={tm:.1f}s", fill=(255, 40, 40))
        tiles.append(im)
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % cols) * w, (i // cols) * h))
    return np.asarray(sheet)


# ------------------------------------------------------------------------------------------------- tools
class StudioTools:
    """The model's tools for one run; every call is logged and timed; slow calls pause the block clock."""

    def __init__(self, backend: Any, out: Path, robot: str, inputs: dict[str, str], *, style: str = "auto",
                 log: Callable[[dict], None] = lambda r: None, characters: list[str] | None = None,
                 samples: int = 16) -> None:
        self.backend, self.out, self.robot, self.inputs, self.style, self.log = backend, out, robot, inputs, style, log
        self.characters = list(characters or [])
        self.samples = samples
        self.clips: list[dict[str, Any]] = []
        self.sandbox: Any = None            # the code-agents Sandbox while a block runs (its clock pauses in tools)
        self.loop: Any = None               # the CodeLoop (images for the next feedback)
        self.final: Any = None
        self.note = ""
        self.timing: dict[str, dict[str, float]] = {}
        self.renders: dict[int, dict] = {}
        self._retargets: dict[tuple[int, str], RobotMotion] = {}
        self._specs: dict[str, dict] = {}
        self._texts: dict[str, ops.Motion] = {}
        self.lineups: list[dict] = []
        self._n = 0
        self.made = 0

    # -- plumbing
    def _dir(self, name: str) -> Path:
        self._n += 1
        p = self.out / "tools" / f"{self._n:03d}-{name}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _call(self, name: str, fn: Callable[[], Any], args: dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"event": "tool", "tool": name, "args": {k: repr(v)[:200] for k, v in args.items()}}
        ctx = self.sandbox.paused() if self.sandbox is not None else None
        try:
            if ctx is not None:
                with ctx:
                    value = fn()
            else:
                value = fn()
            rec["result"] = repr(value)[:400]
            return value
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:800]
            raise
        finally:
            dt = time.perf_counter() - t0
            rec["seconds"] = round(dt, 3)
            st = self.timing.setdefault(name, {"calls": 0, "seconds": 0.0})
            st["calls"] += 1
            st["seconds"] = round(st["seconds"] + dt, 3)
            self.log(rec)

    def _spec(self, robot: str) -> dict:
        if robot not in self._specs:
            self._specs[robot] = (robot_spec(robot, self.out) if mc.kind(robot) == "mjcf" else mc.rig_spec(robot))
        return self._specs[robot]

    def _media(self, path: str) -> str:
        names = {Path(v).name: v for v in self.inputs.values()}          # the key, the file name or the path
        p = self.inputs.get(path) or names.get(Path(str(path)).name if str(path) in names else "", path)
        if p not in self.inputs.values():
            raise ValueError(f"only the provided inputs can be read: {sorted(self.inputs)}")
        return p

    def _show(self, image_path: str | None, note: str) -> None:
        if image_path and self.loop is not None and Path(image_path).exists():
            from PIL import Image
            self.loop.show(np.asarray(Image.open(image_path).convert("RGB")), note)

    def _made(self, m: ops.Motion) -> ops.Motion:
        self.made += 1
        return m

    # -- sources
    def text_motion(self, prompt: Any, seconds: Any = 4.0, seed: int = 0) -> ops.Motion:
        """Text -> motion (NVIDIA Kimodo). prompt: one text, or a list of texts for consecutive segments (then
        seconds is a list too). 2-10 s per segment."""
        key = json.dumps([prompt, seconds, int(seed)])

        def run():
            if key not in self._texts:                      # the same call again: the same motion
                out = self._dir("text_motion")
                J, names, fps, info = self.backend.text_motion(prompt, seconds, int(seed), out)
                m = ops.from_soma(J, names, fps, f"text:{prompt!r}"[:120])
                m.meta.update(info)
                self._texts[key] = m
            return self._made(self._texts[key])
        return self._call("text_motion", run, {"prompt": prompt, "seconds": seconds, "seed": seed})

    def _human(self, media: str, start: float, seconds: float | None, kind: str) -> tuple[ops.Motion, str | None]:
        out = self._dir(kind)
        clip = self.backend.sim("clip", {"start": start, "seconds": seconds}, {"media": media}, out / "clip")
        o = clip["outputs"]
        self.clips.append({"frames": clip["files"]["frames"], "fps": float(o["fps"]), "media": media, "kind": kind})
        W, valid, fps, info, overlay = self.backend.human(Path(clip["files"]["frames"]), float(o["fps"]), out / "human")
        m = ops.from_mediapipe(W, valid, fps, f"{kind}:{Path(media).name}")
        m.meta.update(info, frames_with_person=int(np.sum(valid)), frames=int(len(valid)))
        return m, overlay

    def video_motion(self, path: str = "video", start: float = 0.0, seconds: float | None = None) -> ops.Motion:
        """Reference video -> human motion (the `human` tool: MediaPipe Pose 3D keypoints; approximate world
        grounding, static camera). start / seconds select a window."""
        def run():
            m, overlay = self._human(self._media(path), float(start), seconds, "video_motion")
            self._show(overlay, "human tool overlay (middle frame)")
            return self._made(m)
        return self._call("video_motion", run, {"path": path, "start": start, "seconds": seconds})

    def pose_from_image(self, path: str = "image") -> ops.Motion:
        """A still image of a person -> a one-frame pose (use it with keyframe(...) or hold(...))."""
        def run():
            m, overlay = self._human(self._media(path), 0.0, None, "pose_from_image")
            self._show(overlay, "detected pose")
            return self._made(ops.Motion(m.joints[-1:], m.fps, m.source, m.meta))
        return self._call("pose_from_image", run, {"path": path})

    def motion_from_array(self, joints: Any, fps: float = 30.0) -> ops.Motion:
        """Your own motion: joints (T, 17, 3) in NAMES order, metres, z up."""
        J = np.asarray(joints, float)
        if J.ndim != 3 or J.shape[1:] != (len(ops.NAMES), 3):
            raise ValueError(f"joints must be (T, {len(ops.NAMES)}, 3)")
        return self._made(ops.Motion(J, float(fps), "array"))

    # -- robot side
    def retarget(self, motion: Any, robot: str | None = None) -> RobotMotion:
        """Skeleton motion -> character motion: robots by keypoint IK (MuJoCo), rigged characters by bone driving
        (Blender); see motion_characters for the characters."""
        if isinstance(motion, RobotMotion):
            if robot is None or motion.robot == robot:     # a character's motion stays on its character
                return motion
            motion = motion.source
        robot = robot or self.robot
        key = (id(motion), robot)
        if key in self._retargets:
            return self._retargets[key]

        def run():
            out = self._dir(f"retarget-{robot}")
            np.save(out / "keypoints.npy", motion.joints)
            if mc.kind(robot) == "rig":
                r = self.backend.sim("rig", self._rig_params(robot, out / "keypoints.npy", motion.fps, False), {}, out)
                rm = RobotMotion(np.load(r["files"]["joints"]), motion.fps, robot, "", motion,
                                 dict(r["outputs"]["metrics"]), list(ops.NAMES), kind="rig")
            else:
                smooth = 0.1 if motion.source.startswith(("video", "pose")) else 0.0
                r = self.backend.sim("retarget", {"robot": self._spec(robot), "names": ops.NAMES, "fps": motion.fps,
                                                  "smooth_s": smooth}, {"keypoints": str(out / "keypoints.npy")}, out)
                rm = RobotMotion(np.load(r["files"]["qpos"]), motion.fps, robot, r["files"]["scene"], motion,
                                 {k: v for k, v in r["outputs"].items() if k != "joint_names"},
                                 list(r["outputs"].get("joint_names", [])))
            self._retargets[key] = rm
            return rm
        return self._call("retarget", run, {"motion": motion, "robot": robot})

    def _rig_params(self, name: str, keypoints: Path, fps: float, render: bool) -> dict:
        return {"rig": self._spec(name), "keypoints": str(keypoints), "names": list(ops.NAMES), "fps": fps,
                "render": render, "samples": self.samples, "width": 640, "height": 360,
                "camera": {"distance": mc.camera_distance(name), **RENDER_VIEW}}

    def report(self, motion: Any, robot: str | None = None, physics: bool = True) -> dict[str, Any]:
        """The full plausibility report (check.json): retargeting, kinematic metrics, physics rollouts (robots with
        actuators); for a rigged character its retargeting and contact metrics from Blender."""
        rm = self.retarget(motion, robot)
        if rm.kind == "rig":
            return {"robot": rm.robot, "seconds": rm.seconds, "physics": "n/a: a rigged character (kinematic only)",
                    **rm.info}

        def run():
            out = self._dir(f"check-{rm.robot}")
            np.save(out / "qpos.npy", rm.qpos)
            r = self.backend.sim("check", {"robot": self._spec(rm.robot), "fps": rm.fps, "physics": physics,
                                           "ankle_gain": ANKLE_GAIN},
                                 {"qpos": str(out / "qpos.npy")}, out)
            rep = {"robot": rm.robot, "seconds": rm.seconds, **rm.info, **r["outputs"]}
            if "sim_qpos" in r.get("files", {}):
                rm.info["sim_qpos"] = r["files"]["sim_qpos"]
            return rep
        return self._call("check", run, {"motion": rm, "physics": physics})

    def check(self, motion: Any, robot: str | None = None, physics: bool = False) -> dict[str, Any]:
        """Plausibility of the robot motion for the model: a list of real problems (floor penetration, sliding
        feet, self-collision, joint-limit violations, poor retargeting) and the kinematic metrics. The physics
        rollout (physics=True) is information only; the final report always has it. With a local model it is
        off by default: shown the falls, the model kept trading the request away for a motion that stays up."""
        rep = self.report(motion, robot, physics)
        problems = []
        if rep.get("ground_penetration_max_m", 0) > 0.02:
            problems.append(f"feet or body {rep['ground_penetration_max_m']:.3f} m below the floor")
        if rep.get("foot_skate_mean_speed_mps", 0) > 0.15:
            problems.append(f"planted feet slide at {rep['foot_skate_mean_speed_mps']:.2f} m/s on average")
        if rep.get("self_collision_frames", 0):
            problems.append(f"self-collision in {rep['self_collision_frames']} frames: {rep.get('self_collision_pairs')}")
        if rep.get("joint_limit_violation_frames", 0):
            problems.append(f"joint limits exceeded in {rep['joint_limit_violation_frames']} frames")
        if rep.get("limit_hit_frac", 0) > 0.3:
            problems.append(f"the character hits joint limits in {100 * rep['limit_hit_frac']:.0f}% of the frames "
                            f"({rep.get('limit_hits')}): it cannot follow that part of the motion")
        if rep.get("mean_keypoint_err_m", 0) > 0.1:
            problems.append(f"the robot follows the skeleton poorly ({rep['mean_keypoint_err_m']:.3f} m)")
        phys = ("upright_s", "clip_s", "upright_frac", "fell", "fall_reason", "joint_track_err_rad", "root_drift_m",
                "with_ankle_strategy", "physics")
        out = {"problems": problems or ["none found"], "kinematics": {k: v for k, v in rep.items() if k not in phys}}
        if physics:
            out["physics_info"] = {k: rep[k] for k in phys if k in rep}
        return out

    def render(self, motion: Any, camera: str = "track", style: str | None = None, robot: str | None = None,
               physics: bool = False) -> str:
        """Render to MP4 (style "blender": Cycles, slower, nicer; "mujoco": fast; "auto"); camera "track",
        "orbit" or "fixed". physics=True renders the physics rollout of the last check instead. Returns the path;
        a still frame is attached to your next feedback. Rigged characters render in Blender (track camera)."""
        rm = self.retarget(motion, robot)

        def run():
            out = self._dir(f"render-{rm.robot}")
            if rm.kind == "rig":
                kp = out / "keypoints.npy"
                np.save(kp, rm.source.joints)
                res = self.backend.sim("rig", self._rig_params(rm.robot, kp, rm.fps, True), {}, out)
                r = {"video": res["files"]["video"], "still": res["files"].get("still"), "backend": "blender"}
            else:
                qpos = rm.qpos
                if physics:
                    if "sim_qpos" not in rm.info:
                        raise ValueError("run check(motion) first: physics=True renders its rollout")
                    qpos = np.load(rm.info["sim_qpos"])
                r = self.backend.render(rm.scene, qpos, rm.fps, out, camera=camera, style=style or self.style,
                                        distance=mc.camera_distance(rm.robot))
            self.renders[id(rm)] = {**r, "physics": physics}
            self._show(r.get("still"), f"render ({r.get('backend')}) of {rm!r}")
            return r["video"]
        return self._call("render", run, {"motion": rm, "camera": camera, "style": style, "physics": physics})

    def lineup(self, motion: Any, characters: list[str] | None = None, cols: int | None = None) -> str:
        """The same motion on several characters: one render per character (each with its own proportions and
        the same camera angle), composed into one labelled video grid. Characters default to the run's list (or
        all). Returns the MP4 path; a still of each render is shown."""
        src = motion.source if isinstance(motion, RobotMotion) else motion
        chars = list(characters or self.characters or mc.names())
        bad = [c for c in chars if c not in mc.names()]
        if bad:
            raise ValueError(f"unknown characters {bad}; characters: {mc.names()}")
        videos = []
        for c in chars:
            rm = self.retarget(src, c)
            rend = self.renders.get(id(rm))
            videos.append(rend["video"] if rend and not rend.get("physics") else self.render(rm))

        def run():
            out = self._dir("lineup")
            n = len(chars)
            r = self.backend.sim("compose", {"videos": videos, "labels": [mc.title(c) for c in chars], "fps": src.fps,
                                             "height": 360, "cols": cols or (n if n <= 4 else (n + 1) // 2)}, {}, out)
            self.lineups.append({"video": r["files"]["video"], "characters": chars})
            return r["files"]["video"]
        return self._call("lineup", run, {"motion": src, "characters": chars})

    def look(self, path: str, n: int = 4, note: str = "") -> str:
        """Frames of a video (n, evenly spaced) or an image, as one contact sheet attached to the next feedback."""
        sheet = contact_sheet(Path(path), n)
        if self.loop is not None:
            self.loop.show(sheet, f"{Path(path).name}: {note}"[:200])
        return f"{n} frames of {Path(path).name} will be shown in your next feedback"

    def finish(self, motion: Any, note: str = "") -> str:
        """The final motion (skeleton or robot motion); ends the session."""
        if not isinstance(motion, (ops.Motion, RobotMotion)):
            raise TypeError("finish(motion): pass a motion")
        self.final, self.note = motion, str(note)[:500]
        if self.loop is not None:
            self.loop.done = "finished"
        return "finished"

    def namespace(self) -> dict[str, Any]:
        names = {n: getattr(self, n) for n in ("text_motion", "video_motion", "pose_from_image", "motion_from_array",
                                               "retarget", "check", "render", "lineup", "look", "finish")}
        names.update(ops.EDITS)
        names["NAMES"] = list(ops.NAMES)
        names["CHARACTERS"] = mc.names()
        return names

    def status(self) -> dict[str, Any]:
        return {"robot": self.robot, "motions_made": self.made, "finished": self.final is not None}




# ------------------------------------------------------------------------------------------------- run report
def clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    if out.get("images"):
        out["images"] = [{"note": i.get("note"), "shape": list(np.asarray(i["image"]).shape)} for i in out["images"]]
    return out


def parse_characters(spec: Any) -> list[str]:
    """--characters: None, "all", "robots", "humans" or a comma-separated list (or a list)."""
    if not spec:
        return []
    if isinstance(spec, str):
        everyone = [*mc.ROBOTS_DEFAULT, *[c for c in mc.names() if c not in mc.ROBOTS_DEFAULT]]  # robots first
        spec = {"all": everyone, "robots": list(mc.ROBOTS_DEFAULT), "humans": mc.names("rig")}.get(spec, spec)
        if isinstance(spec, str):
            spec = [c.strip() for c in spec.split(",") if c.strip()]
    bad = [c for c in spec if c not in mc.names()]
    if bad:
        raise ValueError(f"unknown characters {bad}; characters: {', '.join(mc.names())}")
    return list(dict.fromkeys(spec))


def character_table(tools: StudioTools, src: ops.Motion, characters: list[str]) -> dict[str, dict[str, Any]]:
    """Retargeting error, joint-limit hits and plausibility per character (characters.json)."""
    keys = ("scale", "mean_keypoint_err_m", "max_keypoint_err_m", "keypoint_err_pct_height", "limb_angle_err_deg",
            "limit_hit_frac", "limit_hits", "limits", "joint_limit_violation_frames", "ground_penetration_max_m",
            "frames_floating_5cm", "foot_skate_mean_speed_mps", "self_collision_frames", "flight_frames",
            "com_outside_support_frac", "upright_s", "clip_s", "fell", "fall_reason", "with_ankle_strategy", "physics")
    table = {}
    for c in characters:
        rep = tools.report(src, c, physics=True)
        table[c] = {"title": mc.title(c), "kind": mc.kind(c), **{k: rep[k] for k in keys if k in rep}}
    return table


def finalize(tools: StudioTools, out: Path, *, side_by_side: bool = False,
             characters: list[str] | None = None) -> dict[str, Any]:
    """Final motion -> retarget, check, render (unless the model rendered exactly it), files in the run dir; with
    characters, also a report per character and the lineup video (next to the source video with side_by_side)."""
    final = tools.final
    if final is None:
        return {"final": None, "error": "the model did not finish with a motion"}
    tools.sandbox = None                                  # no block is running any more
    rm = tools.retarget(final)
    report = tools.report(rm, physics=True)
    (out / "check.json").write_text(json.dumps(report, indent=2, default=str))
    src = rm.source
    np.savez_compressed(out / "motion.npz", joints=src.joints.astype(np.float32), names=np.array(ops.NAMES),
                        fps=np.float32(src.fps), qpos=rm.qpos.astype(np.float32), robot=np.array(rm.robot),
                        joint_names=np.array(rm.joint_names), source=np.array(src.source))
    files = {"motion": "motion.npz", "check": "check.json"}
    result: dict[str, Any] = {"final": repr(final)}
    if characters:
        table = character_table(tools, src, characters)
        (out / "characters.json").write_text(json.dumps(table, indent=2, default=str))
        files["characters"] = "characters.json"
        result["per_character"] = table
        video = tools.lineup(src, characters)
        shutil.copy(video, out / "lineup.mp4")
        files["lineup"] = "lineup.mp4"
        rend = {"backend": "blender"}
        if side_by_side and tools.clips:                  # the source next to each character
            videos = [tools.clips[-1]["frames"]] + [tools.renders[id(tools.retarget(src, c))]["video"]
                                                    for c in characters]
            r = tools._call("compose", lambda: tools.backend.sim(
                "compose", {"videos": videos, "labels": ["source video"] + [mc.title(c) for c in characters],
                            "fps": src.fps, "height": 360}, {}, out / "compose"), {})
            shutil.copy(r["files"]["video"], out / "split.mp4")
            files["split"] = "split.mp4"
    else:
        rend = tools.renders.get(id(rm))
        if rend is None or rend.get("physics"):
            tools.render(rm)
            rend = tools.renders[id(rm)]
        shutil.copy(rend["video"], out / "motion.mp4")
        files["video"] = "motion.mp4"
        if side_by_side:
            other = "human" if rm.robot == "g1" else "g1"
            tools.render(src, robot=other)
            vids = {rm.robot: rend["video"], other: tools.renders[id(tools.retarget(src, other))]["video"]}
            order = [rm.robot, other]
            r = tools._call("compose", lambda: tools.backend.sim(
                "compose", {"videos": [vids[k] for k in order], "labels": [mc.title(k) for k in order]}, {},
                out / "compose"), {})
            shutil.copy(r["files"]["video"], out / "split.mp4")
            files["split"] = "split.mp4"
    sizes = {k: (out / v).stat().st_size for k, v in files.items() if v.endswith(".mp4")}
    return {**result, "files": files, "video_bytes": sizes, "render_backend": rend.get("backend"), "check": report}




__all__ = ["ANKLE_GAIN", "DEFAULT_MODEL", "MEDIA_WORKER", "RobotMotion", "SIM_WORKER", "StudioTools", "ToolBackend", "character_table",
           "clean_record", "contact_sheet", "finalize", "parse_characters", "render_tool_dir", "robot_spec"]
