"""The motion studio (RFC-0006 part 8): a request (text, an image of a pose, or a reference video) -> a humanoid or
robot motion, checked and rendered, by a chat model writing Python against motion tools.

    hylo motion "wave with the right hand, then bow" --robot g1 --out runs/motion/wave
    hylo motion "do this pose" --image pose.jpg --robot g1 --out DIR [--model SPEC] [--budget turns=12,cost=1]
    hylo motion "copy the person" --video clip.mp4 --robot human --out DIR [--side-by-side] [--style mujoco]
    hylo motion "jump and wave" --characters all --out DIR          the final motion on every character, one video
    hylo motion --free "a looping dance from the clip" --video clip.mp4 --out DIR    (free mode: motion_free)

One agent loop (the code-agents sandbox and CodeLoop): each turn the model writes one Python block; the tools in
its namespace are text_motion (NVIDIA Kimodo), video_motion and pose_from_image (the `human` tool), the motion edits
of motion_ops, retarget (any character of motion_characters: humanoid robots from MuJoCo Menagerie, Hylo's capsule
figure, rigged human characters driven in Blender), check (plausibility and a physics rollout), render (Blender
Cycles on the CPU, MuJoCo fallback), lineup (one render per character, composed into a labelled grid) and look;
finish(motion) ends it. Slow tool calls do not count against the block time limit. The run directory gets log.jsonl
(turns and tool calls), motion.npz, check.json, motion.mp4 (+ split.mp4 with --side-by-side) and summary.json
(turns, cost, wall time, time per tool). With --characters the final motion is also retargeted to, checked on and
rendered with each of those characters: characters.json and lineup.mp4 (with --video and --side-by-side, split.mp4
puts the source clip next to each character) instead of motion.mp4.

This module is the agent loop and the CLI; the tools, their backend and the final report are in motion_tools, the
characters in motion_characters (mapping files) and the retargeting in motion_retarget / motion_sim / motion_blender.
Tests pass a fake backend. SPEC is a chat-model spec (JSON or shorthand, hylo.reasoners.llm.chat); default Ollama
qwen3.5:35b-a3b with vision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hylo.reasoners.llm.chat import ChatModel, build_chat, describe

from . import motion_characters as mc
from .motion_tools import DEFAULT_MODEL, StudioTools, ToolBackend, clean_record, finalize, parse_characters
from .sandbox import Sandbox
from .turns import Budget, CodeLoop


# ------------------------------------------------------------------------------------------------- prompt
SYSTEM = """You are a motion director. You turn a request into a humanoid motion for a robot by writing Python that
calls motion tools, then check it and render it.

Tools (already in your namespace; motions are objects, print them to see a summary):
- text_motion(prompt, seconds=4.0, seed=0) -> Motion. A text-to-motion model (NVIDIA Kimodo, ~1 min per call).
  Describe one person's movement plainly ("a person waves with the right hand"). For a sequence, pass a list of
  prompts and a list of seconds (one per segment): it makes one continuous motion.
- video_motion(path="video", start=0.0, seconds=None) -> Motion from the reference video (pose estimation).
- pose_from_image(path="image") -> a one-frame Motion: the pose of the person in the image.
- Edits (return new motions): concat(m1, m2, ..., blend=0.3), blend(a, b, seconds=0.5), loop(m, times=2),
  mirror(m) (left <-> right), retime(m, speed=None, seconds=None), trim(m, start=0.0, end=None),
  hold(pose, seconds=1.0) (stand still in the last frame of pose), keyframe(m, pose, at, width=0.8) (make m pass
  through pose at time `at` seconds), motion_from_array(joints, fps) (your own (T, 17, 3) keypoints, names NAMES).
- A Motion has .seconds, .frames, .fps, .joints (T, 17, 3 metres, z up, ground z = 0, facing +x at the start),
  .summary() (path length, turn, pelvis and hand heights), .pose(t), .joint(name).
- retarget(m, robot=None) -> RobotMotion for a character (CHARACTERS lists them: robots g1, h1, t1, apollo, n1,
  "human" (a capsule figure) and rigged human characters); the run's robot by default.
- check(m) -> {"problems": [...], "kinematics": {...}}. problems lists real defects (floor penetration,
  sliding feet, self-collision, joint limits, poor retargeting); kinematics has the metrics
  (com_outside_support_frac is static balance and is high for any dynamic motion: not a defect). The studio
  adds a physics rollout to the final report; you do not need it. Never change or drop part of the request to
  make the motion calmer or more stable.
- render(m, camera="track", style=None) -> path of an MP4 (minutes; optional). A still frame is shown to you
  if you can see images.
- lineup(m, characters=None) -> path of an MP4 with the motion on several characters side by side (minutes).
- look(path, n=4) shows you n frames of a video in your next feedback.
- finish(m, note="") -> the final motion. The studio then checks and renders it (you do not have to).

How to work:
- Reply with a short thought, then exactly ONE fenced block ```python ... ```. It runs at once; you see what it
  printed and any error, then you write the next block. Variables persist between blocks.
- Only math and numpy can be imported; no files, no classes, no names starting with an underscore.
- print() what you need to see. Look at the summary: does the motion do what was asked (duration, path length,
  turn angle, hand heights)? If not, try a clearer prompt, another seed, or edit it.
- Be economical: on this busy CPU every text_motion call takes 5-20 minutes and a render a few minutes. A good
  plan uses 2-3 blocks: generate (and edit), check, finish.
Ending: call finish(motion) as soon as the motion does everything the request asks."""

EXAMPLE = """Example (another request: "step to the left, then clap twice"):
```python
m = text_motion(["a person steps to the left", "a person claps twice"], seconds=[2.5, 3.0])
print(m, m.summary())
print(check(m))
```
then, if it looks right: ```python
finish(m, "steps left, then claps")
```"""


def first_message(request: str, robot: str, inputs: dict[str, str], budget: Budget,
                  characters: list[str] | None = None) -> str:
    lines = [f"Request: {request}", f"Robot: {robot}"]
    if characters:
        lines.append(f"The studio will also show the final motion on: {', '.join(characters)}")
    for k, v in inputs.items():
        lines.append(f"Input {k!r}: {Path(v).name} (pass {k!r} as the path)")
    lines.append(f"Budget: {budget.turns} turns" + (f", ${budget.cost_usd}" if budget.cost_usd is not None else ""))
    lines.append("Write your first block.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- run
def run_studio(request: str, out: str | Path, *, robot: str = "g1", image: str | None = None,
               video: str | None = None, model: Any = None, budget: Budget | str | None = None,
               backend: Any = None, style: str = "auto", side_by_side: bool = False, max_wall_s: float | None = None,
               block_timeout_s: float = 60.0, characters: list[str] | None = None, samples: int = 16,
               log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """One motion-studio run; returns summary.json's content."""
    t0 = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    budget = budget if isinstance(budget, Budget) else Budget.parse(budget or "turns=8")
    chat: ChatModel = build_chat(model or DEFAULT_MODEL)
    backend = backend or ToolBackend()
    inputs = {k: str(Path(v).resolve()) for k, v in (("image", image), ("video", video)) if v}
    logf = open(out / "log.jsonl", "w")

    def emit(rec: dict[str, Any]) -> None:
        logf.write(json.dumps(clean_record(rec), default=str) + "\n")
        logf.flush()
        if "turn" in rec:
            log(f"turn {rec['turn']}: {(rec.get('error') or '')[:120]} {(rec.get('output') or '')[:200]}")

    characters = parse_characters(characters)
    tools = StudioTools(backend, out, robot, inputs, style=style, log=emit, characters=characters, samples=samples)
    sandbox = Sandbox(tools.namespace(), block_timeout_s=block_timeout_s)
    tools.sandbox = sandbox
    loop = CodeLoop(chat, sandbox, system=SYSTEM + "\n\n" + EXAMPLE,
                    first=first_message(request, robot, inputs, budget, characters), emit=emit, status=tools.status)
    tools.loop = loop
    emit({"event": "start", "request": request, "robot": robot, "inputs": inputs, "model": describe(chat),
          "budget": budget.model_dump(), "characters": characters})
    stopped = None
    try:
        while loop.done is None:
            stopped = loop.exceeded(budget)
            if stopped is None and max_wall_s is not None and time.perf_counter() - t0 > max_wall_s:
                stopped = f"budget: wall time {max_wall_s:g} s"
            if stopped:
                break
            remaining = {"turns": budget.turns - loop.turn}
            loop.step(remaining)
        stopped = loop.done or stopped
        result = finalize(tools, out, side_by_side=side_by_side, characters=characters)
    finally:
        backend.close()
    wall = time.perf_counter() - t0
    summary = {"request": request, "robot": robot, "characters": characters, "inputs": inputs, "model": describe(chat),
               "stopped": stopped,
               "finished_by_model": tools.final is not None, "note": tools.note, "turns": loop.turn,
               "usage": loop.usage.as_dict(), "cost_usd": loop.usage.as_dict()["cost_usd"],
               "wall_s": round(wall, 1), "tool_seconds": tools.timing, **result}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    emit({"event": "end", **{k: summary[k] for k in ("stopped", "turns", "wall_s")}})
    logf.close()
    return summary


# ------------------------------------------------------------------------------------------------- CLI
def motion_command(argv: list[str]) -> int:
    """motion studio: text, an image of a pose or a reference video -> a motion on robots and human characters"""
    if "--free" in argv:
        from .motion_free import free_command
        return free_command([a for a in argv if a != "--free"])
    ap = argparse.ArgumentParser(prog="hylo motion", description=__doc__.split("\n")[0])
    ap.add_argument("request")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--robot", default="g1", choices=mc.names(), help="the character the agent works with")
    ap.add_argument("--characters", help="also show the final motion on these: all, robots, humans or a,b,c")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="chat model spec (JSON or shorthand); default Ollama qwen3.5:35b-a3b")
    ap.add_argument("--budget", default="turns=8", help="turns=,tokens=,cost=")
    ap.add_argument("--style", default="auto", choices=["auto", "blender", "mujoco"])
    ap.add_argument("--side-by-side", action="store_true",
                    help="with --characters and --video: the lineup next to the source; else the G1 next to the human")
    ap.add_argument("--samples", type=int, default=16, help="Cycles samples per pixel for Blender renders")
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--steps", type=int, default=100, help="Kimodo denoising steps")
    ap.add_argument("--threads", type=int, default=64, help="CPU threads per tool worker")
    a = ap.parse_args(argv)
    model = (json.loads(a.model) if a.model.lstrip().startswith("{") else a.model) if a.model else None
    s = run_studio(a.request, a.out, robot=a.robot, image=a.image, video=a.video, model=model, budget=a.budget,
                   backend=ToolBackend(threads=a.threads, steps=a.steps), style=a.style, side_by_side=a.side_by_side,
                   max_wall_s=a.max_wall_s, characters=parse_characters(a.characters), samples=a.samples,
                   log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({k: s.get(k) for k in ("stopped", "turns", "wall_s", "cost_usd", "files", "final")}, indent=2))
    return 0 if s.get("files") else 1


__all__ = ["DEFAULT_MODEL", "motion_command", "run_studio"]
