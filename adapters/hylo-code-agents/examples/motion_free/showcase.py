"""The motion-studio showcase: many target characters and free mode, one command per request.

    uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py list
    uv run python .../showcase.py run text_all            # one request (Ollama qwen3.5:35b-a3b, Kimodo, Blender)
    uv run python .../showcase.py run all                 # every request, one after another
    uv run python .../showcase.py lineup text_all         # that run's final motion on all characters (no model)
    uv run python .../showcase.py collect                 # results.json from the run directories
    uv run python .../showcase.py smoke --out DIR         # every request with a scripted model, synthetic motion
                                                          # sources, MuJoCo renders and no Blender (minutes)

Real runs write into this directory (one folder per request); inputs are fetched from their original pages into
the asset cache and verified by sha256 (sources.json lists them with their licences). Requests run one at a time:
the Ollama server is shared. Rendering needs OSMesa for MuJoCo (export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu) and Blender ($HYLO_BLENDER, default
~/.local/opt/blender/blender) for the Cycles renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from hylo.codeagents import motion_characters as mc
from hylo.codeagents import motion_ops as ops
from hylo.codeagents.motion import run_studio
from hylo.codeagents.motion_free import run_free
from hylo.codeagents.motion_tools import StudioTools, ToolBackend

HERE = Path(__file__).resolve().parent
UA = "hylo-motion-showcase/0.1"
INPUTS = {
    "jumping_jacks": {
        "file": "Jumping_jacks_and_burpees.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/57/Jumping_jacks_and_burpees.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm",
        "sha256": "1c8370a29c52b5b1538f03f8c63be6cdc9333dea3cbd77cc8f2af0dbe496d5b4",
        "author": "Taco Fleur (Cavemantraining), own work, 2017-11-15",
        "licence": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
        "used": "10.6 s to 13.4 s (jumping jacks)",
        "note": "videos derived from this clip (a motion estimated from it) are shared under CC BY-SA 4.0 with this "
                "attribution"},
    "warrior_ii": {
        "file": "Warrior_II.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/06/Warrior_II.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Warrior_II.jpg",
        "sha256": "bf888f44625f7bc50766dbb8dd851e4abc85e120feb4e1491d16f8b2be519367",
        "author": "lululemon athletica (Flickr: Wunder Groove Crop, Warrior II)",
        "licence": "CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/)",
        "used": "the whole photo: a woman in the Warrior II yoga pose"},
}
ROBOTS3 = ["g1", "h1", "t1"]
REQUESTS: dict[str, dict[str, Any]] = {
    "text_all": {"mode": "guided", "robot": "g1", "characters": "all", "budget": "turns=8",
                 "request": "wave with the right hand, then squat down and stand up again",
                 "what": "a text request rendered on every character"},
    "video_jacks": {"mode": "guided", "robot": "g1", "characters": ["g1", "h1", "kenney_survivor"], "budget": "turns=8",
                    "video": "jumping_jacks", "side_by_side": True,
                    "request": "copy the jumping jacks the person does in the video between 10.6 s and 13.4 s",
                    "what": "a sports clip on two robots and one human character, next to the source"},
    "image_warrior": {"mode": "guided", "robot": "g1", "characters": ["g1", "apollo", "quaternius_casual"],
                      "budget": "turns=8", "image": "warrior_ii",
                      "request": "from standing, move into the pose of the person in the photo, hold it for two "
                                 "seconds, then return to standing",
                      "what": "an image request"},
    "free_loop": {"mode": "free", "budget": "turns=12", "video": "jumping_jacks", "max_tool_calls": 30,
                  "request": "Turn the jumping jacks in the video (10.6 s to 13.4 s) into a routine of about 12 "
                             "seconds: loop the movement, mirror every other repeat, make the joins smooth, and show "
                             "it on two robots and one human character",
                  "what": "free mode: a motion from a video, looped and mirrored into a longer routine"},
    "free_keyframe": {"mode": "free", "budget": "turns=12", "image": "warrior_ii", "max_tool_calls": 30,
                      "request": "A person walks forward four steps and ends in the pose of the person in the photo, "
                                 "holding it for two seconds. Show it on three robots and both human characters",
                      "what": "free mode: text plus a pose image as a keyframe"},
}


# ------------------------------------------------------------------------------------------------- inputs
def fetch(name: str) -> Path:
    """An input from its original page, verified by sha256, cached under $HYLO_ASSETS/inputs."""
    spec = INPUTS[name]
    path = mc.asset_root() / "inputs" / spec["file"]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]:
        return path
    req = urllib.request.Request(spec["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise RuntimeError(f"digest mismatch for {spec['url']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sources() -> dict[str, Any]:
    """Every input, character asset and model the showcase uses, with its licence (sources.json)."""
    return {
        "note": "Inputs and assets are fetched from their original pages into the asset cache and verified by sha256; "
                "none is stored in the repository. Only the outputs of the runs are stored here.",
        "inputs": INPUTS,
        "characters": mc.credits(mc.names()),
        "models_and_programs": {
            "text_motion": "NVIDIA Kimodo (nvidia/Kimodo-SOMA-RP-v1.2), its own model licence, run on the CPU",
            "human": "MediaPipe Pose (Apache-2.0), the hylo-tools human fallback",
            "physics_and_retargeting": "MuJoCo (Apache-2.0)",
            "render": "Blender 5.2.3 Cycles on the CPU (GPL-3.0-or-later, an external program)",
            "agent": "qwen3.5:35b-a3b (vision) through Ollama, the model's own licence",
        },
    }


# ------------------------------------------------------------------------------------------------- runs
def run(name: str, out_root: Path, *, model: Any = None, backend: Any = None, style: str = "auto",
        samples: int = 16, log=print) -> dict[str, Any]:
    """One showcase request; the run directory is out_root/name."""
    spec = REQUESTS[name]
    out = out_root / name
    if out.exists():
        shutil.rmtree(out)
    media = {k: str(fetch(spec[k])) for k in ("video", "image") if spec.get(k)}
    if spec["mode"] == "guided":
        return run_studio(spec["request"], out, robot=spec["robot"], model=model, budget=spec["budget"],
                          backend=backend or ToolBackend(), style=style, side_by_side=spec.get("side_by_side", False),
                          characters=spec["characters"], samples=samples, log=log, **media)
    return run_free(spec["request"], out, model=model, budget=spec["budget"], backend=backend or ToolBackend(),
                    max_tool_calls=spec["max_tool_calls"], samples=samples, style=style, log=log, **media)


def lineup(name: str, out_root: Path, *, backend: Any = None, characters: list[str] | None = None,
           style: str = "auto") -> str:
    """A finished run's final motion on all characters (or `characters`): out_root/name/lineup_all.mp4."""
    run_dir = out_root / name
    z = np.load(run_dir / "motion.npz")
    m = ops.Motion(np.asarray(z["joints"], float), float(z["fps"]), str(z["source"]))
    tools = StudioTools(backend or ToolBackend(), run_dir / "lineup_all", "g1", {}, style=style)
    video = tools.lineup(m, characters or mc.names())
    shutil.copy(video, run_dir / "lineup_all.mp4")
    return str(run_dir / "lineup_all.mp4")


def collect(out_root: Path) -> dict[str, Any]:
    """results.json from the run directories: turns, wall time, time per tool, per-character retargeting error and
    plausibility, video paths. Requests without a finished run are listed as not run, with their command."""
    base = "uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py"
    res: dict[str, Any] = {
        "note": "Each request lists its status; a request that has not been run has no numbers. Model, tools and "
                "renders are those of sources.json; every number below comes from that request's run directory.",
        "commands": {"one request": f"{base} run NAME", "all requests": f"{base} run all",
                     "all characters": f"{base} lineup text_all", "check the plumbing": f"{base} smoke --out DIR"},
        "requests": {}}
    for name, spec in REQUESTS.items():
        d = out_root / name
        cmd = f"{base} run {name}"
        entry: dict[str, Any] = {"what": spec["what"], "mode": spec["mode"], "request": spec["request"],
                                 "inputs": [spec[k] for k in ("video", "image") if spec.get(k)], "reproduce": cmd}
        if not (d / "summary.json").exists():
            res["requests"][name] = {**entry, "status": "not run yet"}
            continue
        s = json.loads((d / "summary.json").read_text())
        videos = {k: f"{name}/{v}" for k, v in (s.get("files") or {}).items() if str(v).endswith(".mp4")}
        if (d / "lineup_all.mp4").exists():
            videos["lineup_all"] = f"{name}/lineup_all.mp4"
        res["requests"][name] = {
            **entry, "status": "finished" if s.get("finished_by_model") else f"stopped: {s.get('stopped')}",
            "model": s.get("model"), "turns": s.get("turns"), "wall_s": s.get("wall_s"),
            "tool_seconds": s.get("tool_seconds"), "tool_calls": s.get("tool_calls"), "final": s.get("final"),
            "characters": s.get("per_character"), "videos": videos}
    lu = out_root / "text_all" / "lineup_all.mp4"
    res["lineup_all_characters"] = {"what": "the final motion of text_all on every character, one video grid",
                                    "reproduce": f"{base} run text_all && {base} lineup text_all",
                                    "status": "finished" if lu.exists() else "not run yet",
                                    **({"video": "text_all/lineup_all.mp4"} if lu.exists() else {})}
    return res


# ------------------------------------------------------------------------------------------------- smoke
def _block(code: str) -> str:
    return f"Smoke test.\n```python\n{code}\n```"


SMOKE_REPLIES = {
    "text_all": [_block("m = text_motion(['a person waves with the right hand', 'a person squats'], seconds=[0.5, 0.5])"
                        "\nprint(m)\nprint(check(m)['problems'])"), _block("finish(m, 'smoke')")],
    "video_jacks": [_block("m = video_motion('video', start=10.6, seconds=1.0)\nprint(m)\nfinish(m, 'smoke')")],
    "image_warrior": [_block("p = pose_from_image('image')\nm = concat(hold(p, 0.5), hold(p, 0.5))\nprint(m)\n"
                             "finish(m, 'smoke')")],
    "free_loop": [_block("from studio import *\nm = video_motion('inputs/Jumping_jacks_and_burpees.webm', 10.6, 1.0)\n"
                         "r = loop(concat(m, mirror(m)), 2)\nsave_motion(r, 'work/routine.npz')\nprint(r)\n"
                         "print(retarget(r, 'h1')['mean_keypoint_err_m'])"),
                  _block("from studio import *\nm = load_motion('work/routine.npz')\n"
                         "finish(m, ['g1', 'h1', 'kenney_survivor'], 'smoke')")],
    "free_keyframe": [_block("from studio import *\nw = text_motion('a person walks forward four steps', 1.0)\n"
                             "p = pose_from_image('inputs/Warrior_II.jpg')\nm = concat(w, hold(p, 0.5))\n"
                             "m = keyframe(m, p, at=m.seconds - 0.1)\nprint(m)\n"
                             "finish(m, ['g1', 'h1', 't1', 'kenney_survivor', 'quaternius_casual'], 'smoke')")],
}


class SmokeBackend(ToolBackend):
    """The real MuJoCo and media workers and the render tool (MuJoCo backend), with synthetic motion sources
    (no Kimodo, no pose model) and rigged characters stood in by their keypoints (no Blender)."""

    REST = {"Hips": (0, 0, 0.95), "Chest": (0, 0, 1.15), "Head": (0, 0, 1.55), "LeftLeg": (0, 0.09, 0.88),
            "LeftShin": (0.01, 0.1, 0.48), "LeftFoot": (0, 0.1, 0.08), "LeftToeBase": (0.13, 0.11, 0.02),
            "LeftArm": (0, 0.18, 1.38), "LeftForeArm": (0, 0.22, 1.1), "LeftHand": (0.02, 0.24, 0.84)}
    MEDIAPIPE = ops.MEDIAPIPE

    def _skeleton(self, seconds: float, fps: float = 30.0) -> np.ndarray:
        rest = dict(self.REST)
        rest.update({k.replace("Left", "Right"): (x, -y, z) for k, (x, y, z) in self.REST.items() if "Left" in k})
        T = max(2, int(seconds * fps))
        J = np.repeat(np.array([rest[n] for n in ops.NAMES], float)[None], T, 0)
        t = np.arange(T) / fps
        J[:, ops.IDX["RightHand"], 2] += 0.4 * np.sin(np.pi * t / max(seconds, 1e-3))
        return J

    def text_motion(self, prompt, seconds, seed, out):
        secs = float(sum(seconds)) if isinstance(seconds, list) else float(seconds)
        return self._skeleton(secs), list(ops.NAMES), 30.0, {"model": "smoke: synthetic"}

    def human(self, frames, fps, out):
        n = len(np.load(frames))
        J = self._skeleton(n / fps, fps)[:n]
        W = np.zeros((len(J), 33, 3))
        for name, ids in self.MEDIAPIPE.items():
            for k in ids:
                W[:, k] = J[:, ops.IDX[name]]
        W[:, [11, 12]] = J[:, [ops.IDX["LeftArm"], ops.IDX["RightArm"]]]
        return W, np.ones(len(J), bool), fps, {"backend": "smoke: synthetic"}, None

    def sim(self, op, params, inputs, out):
        if op != "rig":
            return super().sim(op, params, inputs, out)
        import imageio.v2 as imageio
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "joints.npy", np.load(params["keypoints"]))
        files = {"joints": str(out / "joints.npy")}
        if params.get("render"):
            T = len(np.load(params["keypoints"]))
            with imageio.get_writer(str(out / "video.mp4"), fps=params.get("fps", 30), codec="libx264",
                                    macro_block_size=8, ffmpeg_log_level="error") as w:
                for _ in range(T):
                    w.append_data(np.full((360, 640, 3), 200, np.uint8))
            files["video"] = str(out / "video.mp4")
        return {"ok": True, "outputs": {"metrics": {"smoke": "rig stood in by its keypoints (no Blender)",
                                                    "mean_keypoint_err_m": 0.0, "limit_hit_frac": 0.0}},
                "files": files}


def smoke(out_root: Path, names: list[str]) -> dict[str, Any]:
    from hylo.reasoners.llm.chat import ScriptedChat
    out: dict[str, Any] = {}
    for n in names:
        s = run(n, out_root, model=ScriptedChat(SMOKE_REPLIES[n]), backend=SmokeBackend(threads=16), style="mujoco",
                log=lambda m: None)
        out[n] = {"stopped": s.get("stopped"), "files": s.get("files"), "error": s.get("error")}
        print(n, json.dumps(out[n]), flush=True)
    lineup(names[0], out_root, backend=SmokeBackend(threads=16), characters=["g1", "human", "kenney_survivor"],
           style="mujoco")
    return out


# ------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("name", choices=[*REQUESTS, "all"])
    r.add_argument("--samples", type=int, default=16)
    lu = sub.add_parser("lineup")
    lu.add_argument("name", choices=list(REQUESTS))
    sub.add_parser("collect")
    sub.add_parser("sources")
    sm = sub.add_parser("smoke")
    sm.add_argument("--out", required=True)
    sm.add_argument("names", nargs="*", default=list(REQUESTS))
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n, s in REQUESTS.items():
            print(f"{n:15s} {s['mode']:7s} {s['what']}")
    elif a.cmd == "run":
        for n in (list(REQUESTS) if a.name == "all" else [a.name]):
            s = run(n, HERE, samples=a.samples, log=lambda m: print(m, file=sys.stderr))
            print(n, json.dumps({k: s.get(k) for k in ("stopped", "turns", "wall_s", "files")}))
        (HERE / "results.json").write_text(json.dumps(collect(HERE), indent=1, default=str) + "\n")
    elif a.cmd == "lineup":
        print(lineup(a.name, HERE))
    elif a.cmd == "collect":
        (HERE / "results.json").write_text(json.dumps(collect(HERE), indent=1, default=str) + "\n")
    elif a.cmd == "sources":
        (HERE / "sources.json").write_text(json.dumps(sources(), indent=1) + "\n")
    else:
        smoke(Path(a.out).resolve(), a.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
