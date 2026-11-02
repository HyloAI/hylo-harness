"""Free mode of the motion studio: the model composes its own pipeline as scripts in its run directory.

    hylo motion --free "REQUEST" --out DIR [--video V] [--image I] [--model SPEC] [--budget turns=12]
                [--max-tool-calls 40] [--script-timeout-s 900] [--max-wall-s S]

Each turn the model writes one Python script; motion_scripts checks it, saves it as scripts/turnNN.py and runs it
in a guarded process in the run directory (motion_freerun). Scripts import the kit (`from studio import *`,
motion_freekit) and may do their own processing with numpy and scipy, writing any files inside the run directory.
Every tool call is a request answered here by FreeTools (an allowed tool, every path inside the run directory, the
tool-call budget), run with the studio's tools (motion_tools.StudioTools) and logged in log.jsonl with every
script. finish(motion, characters) ends the session; the studio then reports each character (retargeting error,
joint-limit hits, plausibility) and renders the lineup (motion_tools.finalize). Inputs are copied into inputs/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hylo.reasoners.llm.chat import build_chat, describe

from . import motion_characters as mc
from . import motion_ops as ops
from .motion_scripts import ScriptRunner, inside, plain
from .motion_tools import DEFAULT_MODEL, StudioTools, ToolBackend, clean_record, finalize
from .turns import Budget, CodeLoop

TOOLS = ("text_motion", "video_motion", "pose_from_image", "tool", "characters", "retarget", "check", "render",
         "lineup", "look", "finish")
FREE_OPS = {"characters", "look"}                                    # not counted against the budget
OTHER_TOOLS = {"hands", "depth", "segment", "human", "gravity"}       # through tool(name, ...)
PATH_KEYS = {"path", "video", "image", "motion", "mask", "file", "frames"}


class FreeTools:
    """Answers the kit's requests (motion_freekit) with StudioTools, inside the run directory and the budget."""

    def __init__(self, root: Path, tools: StudioTools, max_tool_calls: int = 40) -> None:
        self.root, self.tools, self.max_tool_calls = root.resolve(), tools, max_tool_calls
        self.calls = 0
        self.final: ops.Motion | None = None
        self.characters: list[str] = []
        self.note = ""

    def _motion(self, path: Any) -> ops.Motion:
        z = np.load(inside(self.root, path))
        J = np.asarray(z["joints"], float)
        if J.ndim != 3 or J.shape[1:] != (len(ops.NAMES), 3) or not np.all(np.isfinite(J)):
            raise ValueError(f"{path}: joints must be finite, shape (T, {len(ops.NAMES)}, 3)")
        return ops.Motion(J, float(z["fps"]), str(z["source"]) if "source" in z else Path(str(path)).stem)

    def _save(self, m: ops.Motion, name: str) -> dict[str, Any]:
        d = self.root / "work" / "studio"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.calls:03d}-{name}.npz"
        np.savez(p, joints=m.joints.astype(np.float32), fps=np.float32(m.fps), source=np.array(m.source))
        return {"motion": str(p.relative_to(self.root)), "summary": m.summary()}

    def _local(self, path: Any) -> str:
        """A tool's output file as a path in the run directory (copied in when it lives elsewhere)."""
        p = Path(str(path)).resolve()
        if self.root in p.parents:
            return str(p.relative_to(self.root))
        d = self.root / "tools" / "files"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{self.calls:03d}-{p.name}"
        shutil.copy(p, dst)
        return str(dst.relative_to(self.root))

    @staticmethod
    def _character(c: Any) -> str:
        if c not in mc.names():
            raise ValueError(f"unknown character {c!r}; characters: {', '.join(mc.names())}")
        return c

    def __call__(self, op: str, args: dict[str, Any]) -> Any:
        if op not in TOOLS:
            raise PermissionError(f"refused: {op!r} is not a studio tool (tools: {', '.join(TOOLS)})")
        if op not in FREE_OPS:
            if self.calls >= self.max_tool_calls:
                raise PermissionError(f"refused: the tool-call budget ({self.max_tool_calls}) is used up; finish now")
            self.calls += 1
        for k, v in args.items():
            if k in PATH_KEYS and v is not None:
                inside(self.root, v)
        return getattr(self, f"op_{op}")(**args)

    # -- the requests (one method per kit function)
    def op_characters(self) -> list[dict]:
        return [{"name": c, "title": mc.title(c), "kind": mc.kind(c), "height_m": mc.load(c).get("height_m")}
                for c in mc.names()]

    def op_text_motion(self, prompt: Any, seconds: Any = 4.0, seed: int = 0) -> dict:
        return self._save(self.tools.text_motion(prompt, seconds, int(seed)), "text_motion")

    def _register(self, path: str) -> str:
        key = str(inside(self.root, path))
        self.tools.inputs = {**self.tools.inputs, Path(key).name: key}
        return key

    def op_video_motion(self, path: str, start: float = 0.0, seconds: float | None = None) -> dict:
        return self._save(self.tools.video_motion(self._register(path), float(start), seconds), "video_motion")

    def op_pose_from_image(self, path: str) -> dict:
        return self._save(self.tools.pose_from_image(self._register(path)), "pose_from_image")

    def op_tool(self, name: str, kwargs: dict | None = None) -> dict:
        if name not in OTHER_TOOLS:
            raise PermissionError(f"refused: tool {name!r} is not allowed here (allowed: {sorted(OTHER_TOOLS)}; "
                                  f"text_motion, video_motion, pose_from_image and render have their own functions)")
        kw = dict(kwargs or {})
        for k in list(kw):
            if k in ("workdir", "out", "threads"):
                raise PermissionError(f"refused: {k} is set by the studio")
            if k in PATH_KEYS and kw[k] is not None:
                kw[k] = str(inside(self.root, kw[k]))
        t = self.tools
        out = t._dir(f"tool-{name}")
        r = t._call(name, lambda: t.backend.tool(name, kw, out), kw)
        return {"outputs": json.loads(json.dumps(r.get("outputs", {}), default=plain)),
                "files": {k: self._local(v) for k, v in r.get("files", {}).items() if v and Path(v).exists()}}

    def op_retarget(self, motion: str, character: str) -> dict:
        rm = self.tools.retarget(self._motion(motion), self._character(character))
        return {k: v for k, v in rm.info.items() if k != "sim_qpos"}

    def op_check(self, motion: str, character: str) -> dict:
        return json.loads(json.dumps(self.tools.check(self._motion(motion), self._character(character)), default=plain))

    def op_render(self, motion: str, character: str = "g1", camera: str = "track") -> str:
        return self._local(self.tools.render(self._motion(motion), camera, robot=self._character(character)))

    def op_lineup(self, motion: str, characters: list[str] | None = None) -> str:
        cs = [self._character(c) for c in (characters or mc.names())]
        return self._local(self.tools.lineup(self._motion(motion), cs))

    def op_look(self, path: str, n: int = 4, note: str = "") -> str:
        return self.tools.look(str(inside(self.root, path)), int(n), str(note))

    def op_finish(self, motion: str, characters: list[str] | None = None, note: str = "") -> str:
        m = self._motion(motion)
        self.characters = [self._character(c) for c in (characters or ["g1"])]
        self.final, self.note = m, str(note)[:500]
        self.tools.final, self.tools.note = m, self.note
        if self.tools.loop is not None:
            self.tools.loop.done = "finished"
        return "finished: the studio will check and render it"


# ------------------------------------------------------------------------------------------------- prompt
SYSTEM = """You are a motion director with a studio of tools. You turn a request into a humanoid motion shown on
robots and human characters, by writing Python scripts. You decide the pipeline: which tools, which processing,
which characters.

Each turn: a short thought, then exactly ONE fenced ```python``` block. The studio saves it as scripts/turnNN.py
and runs it as a separate process in your run directory (the current directory). You see what it printed, any
error, and images you asked to look at. Scripts do not share variables: save what you need to files (save_motion,
np.save, json) and load it in the next script.

Start every script with `from studio import *`. The studio kit:
- Sources (each returns a Motion): text_motion(prompt, seconds=4.0, seed=0) (NVIDIA Kimodo; a list of prompts
  and a list of seconds makes one continuous motion; several minutes per call), video_motion(path, start=0.0,
  seconds=None) (pose estimation from a video of one person), pose_from_image(path) (a one-frame Motion).
- Other tools: hands(video), depth(image), segment(image, prompts), human(video) -> {"outputs", "files"}.
- Motion: .joints (T, 17, 3) metres, z up, ground z = 0, facing +x at the start; .fps; .frames; .seconds;
  .summary(); .joint(name); .pose(t). NAMES lists the 17 keypoints. motion_from_array(joints, fps) wraps your
  own array. save_motion(m, path=None) -> path, load_motion(path).
- Edits (return new motions): concat(m1, m2, ..., blend=0.3), blend(a, b, seconds=0.5), loop(m, times=2),
  mirror(m), retime(m, speed=None, seconds=None), trim(m, start, end), hold(pose, seconds), keyframe(m, pose, at,
  width=0.8). You can also write your own processing with numpy and scipy (filters, blending, stylising, looping)
  on m.joints; keep bone lengths plausible and the feet on the ground (z >= 0).
- characters() lists the characters; retarget(m, character) -> metrics (keypoint error, limb angle error,
  joint-limit hits); check(m, character) -> {"problems", "kinematics"}.
- render(m, character, camera="track") -> MP4 path (Blender, minutes); lineup(m, characters) -> one MP4 with
  the motion on several characters side by side (minutes per character). look(path, n=4) shows you frames of a
  video or an image in your next feedback: look at your renders and fix what is wrong.
- finish(m, characters=[...], note="") ends the session; the studio then checks the motion on each of those
  characters and renders them together. A ToolError means a request was refused or failed.

Rules: work only inside the run directory (relative paths; inputs are in inputs/). Imports allowed: numpy, scipy,
math, json, statistics, itertools, functools, collections, random, re, copy, pathlib, studio, and your own
modules in lib/ (write them with open("lib/name.py", "w")). No processes, network or eval. Each script has a
time limit (tool time excluded) and the session a tool-call budget. print() what you need to know.
Be economical: plan in few scripts, generate once, process, check on the characters, render once and look, then
finish. Never drop part of the request to make the motion easier."""


def first_message(request: str, inputs: dict[str, str], budget: Budget, max_tool_calls: int) -> str:
    lines = [f"Request: {request}", *[f"Input {k}: {v}" for k, v in inputs.items()]]
    chars = "; ".join(f"{c} ({mc.title(c)}, {mc.kind(c)}, {mc.load(c).get('height_m')} m)" for c in mc.names())
    lines += [f"Characters: {chars}", f"Budget: {budget.turns} turns, {max_tool_calls} tool calls",
              "Write your first script."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- run
def run_free(request: str, out: str | Path, *, image: str | None = None, video: str | None = None, model: Any = None,
             budget: Budget | str | None = None, backend: Any = None, max_tool_calls: int = 40,
             script_timeout_s: float = 900.0, max_wall_s: float | None = None, samples: int = 16,
             style: str = "auto", log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """One free-mode session; returns summary.json's content."""
    t0 = time.perf_counter()
    out = Path(out).resolve()
    for d in ("inputs", "lib", "work"):
        (out / d).mkdir(parents=True, exist_ok=True)
    budget = budget if isinstance(budget, Budget) else Budget.parse(budget or "turns=12")
    chat = build_chat(model or DEFAULT_MODEL)
    backend = backend or ToolBackend()
    inputs = {}
    for k, v in (("image", image), ("video", video)):
        if v:
            dst = out / "inputs" / Path(v).name
            if Path(v).resolve() != dst:
                shutil.copy(v, dst)
            inputs[k] = str(dst.relative_to(out))
    logf = open(out / "log.jsonl", "w")

    def emit(rec: dict[str, Any]) -> None:
        logf.write(json.dumps(clean_record(rec), default=str) + "\n")
        logf.flush()
        if "turn" in rec:
            log(f"turn {rec['turn']}: {(rec.get('error') or '')[:160]} {(rec.get('output') or '')[:240]}")
        elif rec.get("event") == "request":
            log(f"  {rec.get('op')} {rec.get('seconds')} s {rec.get('refused') or rec.get('error') or ''}"[:240])

    tools = StudioTools(backend, out, "g1", {k: str(out / v) for k, v in inputs.items()}, log=emit, samples=samples,
                        style=style)
    free = FreeTools(out, tools, max_tool_calls)
    runner = ScriptRunner(out, free, emit=emit, timeout_s=script_timeout_s)

    def status() -> dict[str, Any]:
        return {"tool_calls": free.calls, "tool_calls_left": max_tool_calls - free.calls, "scripts": runner.scripts,
                "refused_requests": runner.refusals, "finished": free.final is not None}
    loop = CodeLoop(chat, runner, system=SYSTEM, first=first_message(request, inputs, budget, max_tool_calls),
                    emit=emit, status=status)
    tools.loop = loop
    emit({"event": "start", "mode": "free", "request": request, "inputs": inputs, "model": describe(chat),
          "budget": budget.model_dump(), "max_tool_calls": max_tool_calls, "script_timeout_s": script_timeout_s})
    stopped = None
    try:
        while loop.done is None:
            stopped = loop.exceeded(budget)
            if stopped is None and max_wall_s is not None and time.perf_counter() - t0 > max_wall_s:
                stopped = f"budget: wall time {max_wall_s:g} s"
            if stopped:
                break
            loop.step({"turns": budget.turns - loop.turn, "tool_calls": max_tool_calls - free.calls})
        stopped = loop.done or stopped
        if free.final is not None:
            tools.robot = free.characters[0]
            result = finalize(tools, out, characters=free.characters)
        else:
            result = {"final": None, "error": "the model did not finish with a motion"}
    finally:
        backend.close()
    summary = {"mode": "free", "request": request, "inputs": inputs, "model": describe(chat), "stopped": stopped,
               "finished_by_model": free.final is not None, "note": free.note, "characters": free.characters,
               "turns": loop.turn, "scripts": runner.scripts, "tool_calls": free.calls,
               "refused_requests": runner.refusals, "usage": loop.usage.as_dict(),
               "cost_usd": loop.usage.as_dict()["cost_usd"], "wall_s": round(time.perf_counter() - t0, 1),
               "tool_seconds": tools.timing, **result}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    emit({"event": "end", **{k: summary[k] for k in ("stopped", "turns", "wall_s")}})
    logf.close()
    return summary


def free_command(argv: list[str]) -> int:
    """hylo motion --free: the model composes its own pipeline as scripts in its run directory"""
    ap = argparse.ArgumentParser(prog="hylo motion --free", description=__doc__.split("\n")[0])
    ap.add_argument("request")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="chat model spec (JSON or shorthand); default Ollama qwen3.5:35b-a3b")
    ap.add_argument("--budget", default="turns=12", help="turns=,tokens=,cost=")
    ap.add_argument("--max-tool-calls", type=int, default=40)
    ap.add_argument("--script-timeout-s", type=float, default=900.0, help="per script, tool time excluded")
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--samples", type=int, default=16, help="Cycles samples per pixel")
    ap.add_argument("--style", default="auto", choices=["auto", "blender", "mujoco"], help="robot renders")
    ap.add_argument("--steps", type=int, default=100, help="Kimodo denoising steps")
    ap.add_argument("--threads", type=int, default=64, help="CPU threads per tool worker")
    a = ap.parse_args(argv)
    model = (json.loads(a.model) if a.model.lstrip().startswith("{") else a.model) if a.model else None
    s = run_free(a.request, a.out, image=a.image, video=a.video, model=model, budget=a.budget,
                 backend=ToolBackend(threads=a.threads, steps=a.steps), max_tool_calls=a.max_tool_calls,
                 script_timeout_s=a.script_timeout_s, max_wall_s=a.max_wall_s, samples=a.samples, style=a.style,
                 log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({k: s.get(k) for k in ("stopped", "turns", "tool_calls", "wall_s", "files", "final")}, indent=2))
    return 0 if s.get("files") else 1


__all__ = ["FreeTools", "free_command", "run_free"]
