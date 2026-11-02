"""The scene builder workflow (RFC-0006 part 8): task text -> scene file -> checked -> solved with privileged state.

    hylo scene build "put the ball in the cup" --out runs/scenes/ball-cup [--model SPEC] [--attempts 6]
    hylo scene solve runs/scenes/ball-cup [--model SPEC] [--turns 30] [--seed 0] [--no-video]
    hylo scene check runs/scenes/ball-cup/scene.json

build: a chat model writes a `hylo.scene/v0.1` file (hylo.core.scene) from the task text; the scene checker
(scene_check.py: schema, static, physics, reach) answers with a numbered list of problems, which go back to the
model until the scene passes or the attempt budget is spent. Writes DIR/scene.json (the last attempt),
DIR/attempts/NN.json and DIR/build.json (per attempt: problems, tokens, latency; totals; built or not).

solve: `code.llm` on `mujoco.panda_scene` with the scene, with PRIVILEGED state: the env runs with
`privileged_obs: true` (part of the env params, so of any digest) and the toolkit is PrivilegedSceneTools, which adds
`robot.scene()` (simulator object poses, extents, joint states), `robot.object_points(name)` and
`robot.scene_table()`. The trace names the toolkit and every turn's code. Writes DIR/solve/trace (the episode
trace), DIR/solve/turns.jsonl, DIR/solve.json (outcome, turns, usage, wall time) and DIR/front.mp4 (front camera,
when imageio-ffmpeg is installed).

Free mode (scene_free.py): `hylo scene build --free TASK --out DIR` has the model write a Python scene program
(`hylo.freescene/v0.1`: any robot of panda / aloha / g1, any MuJoCo feature, its own success function) instead of a
scene file; `check` and `solve` recognise a directory holding scene.py (or a .py path) and use the free checker
(scene_free_check.py) and the free-scene privileged toolkit.

SPEC is a chat-model spec (JSON or shorthand, see hylo.reasoners.llm.chat); default: Ollama qwen3.5:35b-a3b.
Task-agnostic: nothing here knows a task; the prompt teaches the format with two examples of its own.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hylo.core.registry import registry
from hylo.core.scene import Problem, Scene
from hylo.core.task import TaskSpec
from hylo.manip.tools import Table, Tools
from hylo.reasoners.llm.chat import ChatError, ChatModel, Usage, build_chat, describe

from .scene_check import DEFAULT_ENV, CheckReport, SceneChecker, parse_scene
from .scene_episode import FrameRecorder, run_recorded_episode, write_mp4

DEFAULT_MODEL: dict[str, Any] = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True,
                                 "reasoning_effort": "none", "max_tokens": 3500, "timeout_s": 3600,
                                 "retries": 1}
PRIVILEGED_TOOLS = "hylo.codeagents.scenes:PrivilegedSceneTools"

# ---------------------------------------------------------------------------------------------------------- prompt
SYSTEM = """You write simulation scenes for a robot manipulation benchmark. A scene is one JSON document in the
format hylo.scene/v0.1 (its JSON Schema is below). The simulator turns it into a MuJoCo world: a Franka Panda arm
with a two-finger gripper at one end of a table, and your objects on the table. A checker loads your scene, lets it
settle, checks that the robot can reach everything and that the goal is not already met; then a separate agent
tries to solve the task with the robot.

The world
- Units: metres and kilograms; ANGLES IN DEGREES. +z is up; the table top is z = 0.
{robot}
- Keep every object the robot has to touch, and every place it has to put something, inside the area the arm
  serves best (above). Do not put objects in a straight line between the arm base and the area.
- The gripper opens {opening:.2f} m: whatever must be picked up needs a horizontal side of at most 0.06 m (blocks of
  0.04 m, balls and cylinders of radius 0.02-0.03 m), mass at most 0.3 kg.
- Typical sizes: block 0.04 m cube, 0.05 kg; ball radius 0.02 m; tray [0.14, 0.14, 0.04]; cup [0.045, 0.08].

Objects
- Every object has a unique `name` (identifier) and a unique `label`: the words the instruction uses for it.
- Give `shape` and `size`, or `parts` (primitives placed in the object's own frame) for a built thing (a cabinet).
  `tray` and `cup` are ready-made open containers: use them for a box, bin, bowl, basket or cup to put things in.
- Start pose: `pose` with pos [x, y] rests the object on the table, or on the object named in `on`. Use `region`
  (x and y ranges about 0.08-0.15 m wide, yaw range) for objects the robot moves, so every episode starts a bit
  differently. A region must leave room for the object next to everything else.
- `fixed: true` for things that must not move (cabinet, button housing, a bin fixed to the table).
- Moving parts (drawer, door, lid, button) are separate objects with `parent` (what they are attached to) and a
  `joint` (slide: metres along `axis`; hinge: degrees about `axis` through `anchor`). Their `pose.pos` is
  [x, y, z] in the parent's frame. A part does not collide with its parent: its joint range is what stops it.
  The parent of a drawer or door is usually fixed.
- Objects must not overlap at the start; leave at least 0.03 m between them.
- Make it physically what the task says: something to put things in is hollow (tray, cup, or walls and a floor
  from parts), a drawer is a hollow box on a slide joint, a sliding lid uses a slide joint, a door or flap a hinge.

Success predicate (`success`, field `op`)
- on(a, b): a rests on top of b (b may be "table"). inside(a, b): a's centre inside container b (or `region`).
  near(a, b or point, max_dist). above(a, b). upright(a, axis). lifted(a, height). row(objects, axis): in a line,
  in the listed order of increasing x or y. joint_at_least / joint_at_most(object, value or fraction of range).
  not(term). all_of(terms). any_of(terms). sequence(terms): each becomes true after the previous ones.
- Relations about an object need it at rest and released by the gripper (unless released: false).
- The predicate must be FALSE at the start. Use only object names that exist.
- The instruction names every object the robot must act on by its label.

Reply with one or two sentences of plan, then the complete scene in ONE ```json block (no comments in the JSON).

JSON Schema of the format:
{schema}

Example 1: "put the green cylinder in the tray"
```json
{example1}
```

Example 2: "press the button, then put the blue block on the yellow block"
```json
{example2}
```"""

EXAMPLE_1 = {
    "name": "cylinder_to_tray", "instruction": "put the green cylinder in the tray", "max_steps": 600,
    "objects": [
        {"name": "tray", "label": "tray", "shape": "tray", "size": [0.15, 0.15, 0.04], "colour": "grey",
         "fixed": True, "pose": {"pos": [0.05, 0.18]}},
        {"name": "green_cylinder", "label": "green cylinder", "shape": "cylinder", "size": [0.02, 0.06],
         "colour": "green", "mass": 0.05, "region": {"x": [-0.12, 0.0], "y": [-0.2, -0.08], "yaw_deg": [0, 0]}},
        {"name": "red_block", "label": "red block", "shape": "box", "size": [0.04, 0.04, 0.04], "colour": "red",
         "mass": 0.05, "region": {"x": [0.06, 0.14], "y": [-0.12, -0.02], "yaw_deg": [-30, 30]}}],
    "success": {"op": "inside", "a": "green_cylinder", "b": "tray"}}

EXAMPLE_2 = {
    "name": "button_then_stack", "instruction": "press the button, then put the blue block on the yellow block",
    "max_steps": 900,
    "objects": [
        {"name": "housing", "label": "button housing", "shape": "box", "size": [0.08, 0.08, 0.02], "colour": "black",
         "fixed": True, "pose": {"pos": [0.08, 0.2]}},
        {"name": "button", "label": "button", "parent": "housing", "shape": "cylinder", "size": [0.025, 0.012],
         "colour": "red", "mass": 0.02, "pose": {"pos": [0, 0, 0.016]},
         "joint": {"type": "slide", "axis": [0, 0, 1], "range": [-0.008, 0], "spring": 300, "damping": 3}},
        {"name": "yellow_block", "label": "yellow block", "shape": "box", "size": [0.04, 0.04, 0.04],
         "colour": "yellow", "mass": 0.05, "region": {"x": [-0.05, 0.05], "y": [-0.22, -0.12], "yaw_deg": [-20, 20]}},
        {"name": "blue_block", "label": "blue block", "shape": "box", "size": [0.04, 0.04, 0.04], "colour": "blue",
         "mass": 0.05, "region": {"x": [-0.2, -0.12], "y": [0.0, 0.12], "yaw_deg": [-20, 20]}}],
    "success": {"op": "sequence", "terms": [{"op": "joint_at_most", "object": "button", "value": -0.005},
                                            {"op": "on", "a": "blue_block", "b": "yellow_block"}]}}


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def scene_schema() -> str:
    """The format's JSON Schema, compact (titles dropped) for a prompt."""
    return json.dumps(_strip_titles(Scene.model_json_schema()), separators=(",", ":"))


def builder_system(robot_notes: str, opening: float) -> str:
    bullets = "\n".join(f"- {s.strip()}." for s in robot_notes.split(". ") if s.strip() and "Action:" not in s
                        and "hand.arm" not in s and "TCP" not in s and "success needs" not in s.lower())
    return SYSTEM.format(robot=bullets.replace("..", "."), opening=opening, schema=scene_schema(),
                         example1=json.dumps(EXAMPLE_1), example2=json.dumps(EXAMPLE_2))


FENCE_JSON = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_json(text: str) -> str | None:
    """The JSON document of a reply: the first fenced block that parses as an object, else the outermost {...}."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for block in FENCE_JSON.findall(text):
        if block.strip().startswith("{"):
            return block.strip()
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if 0 <= i < j else None


def feedback_text(report: CheckReport | None, problems: Sequence[Problem], attempt: int, left: int) -> str:
    listed = report.text() if report is not None else "\n".join(f"{i + 1}. [{p.code}] {p.message}"
                                                                for i, p in enumerate(problems))
    where = ""
    if report is not None and report.settled:
        where = "\nWhere the objects ended up after settling (centre, extent):\n" + "\n".join(
            f"- {n}: {v['pos']} {v['extent']}" for n, v in report.settled.items())
    return (f"The checker rejected attempt {attempt}:\n{listed}{where}\n\nFix every problem ({left} attempt(s) left). "
            "Reply with the complete corrected scene in one ```json block.")


# --------------------------------------------------------------------------------------------------------- builder
class SceneBuilder:
    """A chat model writes a scene; the checker's problems go back to it until it passes (attempt budget)."""

    def __init__(self, model: str | Mapping[str, Any] | ChatModel | None = None, *, attempts: int = 6,
                 env_type: str = DEFAULT_ENV, seeds: Sequence[int] = (0, 1), reply_tokens: int | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.model = build_chat(model or DEFAULT_MODEL)
        self.attempts, self.env_type, self.seeds, self.reply_tokens = attempts, env_type, tuple(seeds), reply_tokens
        self.say = log or (lambda msg: None)

    def system(self, checker: SceneChecker) -> str:
        robot = checker.env().spec().robot
        opening = robot.arms[0].hand.max_width if robot is not None and robot.arms else 0.08
        return builder_system(robot.notes if robot is not None else "", opening)

    def build(self, task: str, out_dir: str | Path) -> dict[str, Any]:
        out = Path(out_dir)
        (out / "attempts").mkdir(parents=True, exist_ok=True)
        checker = SceneChecker(self.env_type, self.seeds)
        usage, records, t0 = Usage(), [], time.monotonic()
        system = self.system(checker)
        first = {"role": "user", "content": f"Task: {task.strip()}\n\nWrite the scene."}
        history: list[dict[str, Any]] = []
        built, history_note = False, ""
        try:
            for k in range(1, self.attempts + 1):
                msgs = [{"role": "system", "content": system}, first, *history]
                rec: dict[str, Any] = {"attempt": k}
                try:
                    reply = self.model.chat(msgs, max_tokens=self.reply_tokens)
                except ChatError as exc:
                    rec.update(model_error=str(exc)[:300], ok=False, problems=[])
                    records.append(rec)
                    self.say(f"attempt {k}: model error {exc}")
                    continue
                usage.add(reply)
                rec.update(reply.accounting())
                doc = extract_json(reply.text)
                report: CheckReport | None = None
                if doc is None:
                    problems = [Problem(code="format", message="no ```json block with the scene in the reply")]
                else:
                    (out / "attempts" / f"{k:02d}.json").write_text(doc)
                    path = out / "scene.json"
                    path.write_text(doc)
                    scene, problems = parse_scene(doc)
                    if scene is not None:
                        path.write_text(scene.model_dump_json(indent=1, exclude_defaults=True))
                        report = checker.check(scene, path)
                        problems = list(report.problems)
                rec.update(ok=not problems, problems=[p.model_dump(mode="json") for p in problems],
                           warnings=[p.model_dump(mode="json") for p in (report.warnings if report else ())],
                           stages=list(report.stages) if report else ["schema"], reply_chars=len(reply.text))
                records.append(rec)
                self.say(f"attempt {k}: {'passed' if not problems else f'{len(problems)} problem(s)'} "
                         f"({reply.latency_s:.0f} s, {reply.tokens_out} tokens out)")
                if not problems:
                    built = True
                    break
                history_note += f"\nattempt {k}: " + "; ".join(f"[{p.code}] {p.message[:120]}" for p in problems[:4])
                history = [{"role": "assistant", "content": f"```json\n{doc}\n```" if doc else reply.text[:2000]},
                           {"role": "user", "content": feedback_text(report, problems, k, self.attempts - k)
                            + (f"\n\n(Problems of your earlier attempts, for reference:{history_note})" if k > 1 else "")}]
        finally:
            checker.close()
        result = {"task": task, "model": describe(self.model), "built": built,
                  "attempts": len(records), "records": records, "usage": usage.as_dict(),
                  "wall_s": round(time.monotonic() - t0, 1), "scene": str(out / "scene.json")}
        (out / "build.json").write_text(json.dumps(result, indent=1, default=str))
        return result


# ------------------------------------------------------------------------------------------- privileged toolkit
class PrivilegedSceneTools(Tools):
    """The hylo-manip toolkit plus PRIVILEGED simulator state (the env must run with privileged_obs: true)."""

    privileged = True

    def _priv(self) -> dict[str, Any]:
        obs = self.executor.observation()
        priv = (obs.extra or {}).get("privileged")
        if not priv:
            raise RuntimeError("no privileged state in the observation: run the env with privileged_obs: true")
        return priv

    def scene(self) -> dict[str, dict[str, Any]]:
        """PRIVILEGED: every object of the scene now, from the simulator: {name: {"label", "shape", "pos" (3,)
        centre of its body frame, "quat" (w, x, y, z), "yaw" (rad), "size" (the scene file's size, if a primitive),
        "extent" (3,) full extents in its own frame, "bounds" ((3,), (3,)) world bounding box min / max, "top"
        (z of its highest point), "movable", "fixed", "parent", "joint" {"type", "value", "range", "units"} for
        drawers / doors / lids / buttons, "interior" {"centre", "half"} for trays and cups}}."""
        out = {}
        for name, o in self._priv()["objects"].items():
            lo, hi = np.asarray(o["aabb"][0]), np.asarray(o["aabb"][1])
            rec = {"label": o["label"], "shape": o["shape"], "pos": np.asarray(o["pos"]), "quat": np.asarray(o["quat"]),
                   "yaw": float(o["yaw"]), "extent": np.asarray(o["extent"]), "bounds": (lo, hi), "top": float(hi[2]),
                   "movable": o["movable"], "fixed": o["fixed"], "parent": o["parent"]}
            for key in ("size", "joint", "interior", "colour"):
                if key in o:
                    rec[key] = o[key]
            out[name] = rec
        return out

    def object_points(self, name: str, spacing: float = 0.006) -> np.ndarray:
        """PRIVILEGED: (N, 3) world points on the surface of an object's oriented bounding box, spaced about
        `spacing` m: use them where the toolkit wants observed object points (robot.box, robot.grasps,
        robot.place_at's obj_points after moving them into the object frame)."""
        o = self._priv()["objects"][name]
        lo, hi = np.asarray(o["local_box"][0]), np.asarray(o["local_box"][1])
        w, x, y, z = o["quat"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        pts = []
        for ax in range(3):
            u, v = [k for k in range(3) if k != ax]
            gu = np.linspace(lo[u], hi[u], max(2, int(np.ceil((hi[u] - lo[u]) / spacing)) + 1))
            gv = np.linspace(lo[v], hi[v], max(2, int(np.ceil((hi[v] - lo[v]) / spacing)) + 1))
            uu, vv = np.meshgrid(gu, gv)
            for side in (lo[ax], hi[ax]):
                p = np.zeros((uu.size, 3))
                p[:, u], p[:, v], p[:, ax] = uu.ravel(), vv.ravel(), side
                pts.append(p)
        return np.concatenate(pts) @ R.T + np.asarray(o["pos"])

    def scene_table(self, height: float | None = None) -> Table:
        """PRIVILEGED: the table as a Table (height from the simulator), or a Table at `height` (e.g. the top of an
        object to place on: `robot.place_at(..., table=robot.scene_table(scene["box"]["top"]))`)."""
        t = self._priv()["table"]
        return Table(height=float(t["height"] if height is None else height), normal=np.array([0.0, 0.0, 1.0]),
                     extent=np.array([t["x"], t["y"]], float))

    def extra_reference(self) -> str:
        return PRIVILEGED_REFERENCE


PRIVILEGED_REFERENCE = """## Privileged simulator state (this run is flagged privileged)
This run gives you the simulator's ground truth: start with robot.scene() and use its exact poses and sizes
instead of look / find / objects (perception is still there if you want to look).
robot.scene() -> dict
    {name: {"label", "shape", "pos", "quat", "yaw", "size", "extent", "bounds": (min, max), "top", "movable",
    "fixed", "parent", "joint": {"type", "value", "range", "units"}, "interior": {"centre", "half"}}}: exact object
    poses and sizes from the simulator, fresh at every call. Prefer it to perception.
robot.object_points(name, spacing=0.006) -> (N, 3) points on the object's oriented bounding box (world), usable
    wherever the toolkit takes observed object points.
robot.scene_table(height=None) -> Table: the table, or a support plane at `height` for place_at.

Example (pick an object and put it on another one):
    s = robot.scene(); table = robot.scene_table()
    pts = robot.object_points("red_block"); box = robot.box(pts, table)
    plan = robot.plan(robot.grasps("arm", pts, table), table)
    out = robot.run(plan); print(out["tcp_error"], robot.holding("arm"))
    up = out["tcp"]["arm"].copy(); up[2, 3] += 0.12
    robot.run(robot.plan(robot.Goal("lift", "arm", tcp=up), table))
    obj_in_tcp = np.linalg.inv(plan.goal.tcp) @ box.pose
    local = (pts - box.pose[:3, 3]) @ box.pose[:3, :3]
    goal = robot.place_at("arm", obj_in_tcp, local, xy=s["plate"]["pos"][:2], table=robot.scene_table(s["plate"]["top"]))
    robot.run(robot.plan(goal, table))
    retreat = robot.status()["tcp"]["arm"].copy(); retreat[2, 3] += 0.1
    robot.run(robot.plan(robot.Goal("retreat", "arm", tcp=retreat), table))
For a drawer, door or lid: grasp its handle with robot.grasp_at (a TCP pose you build from scene() numbers) and move
along the joint with robot.Goal("push", ...) (a straight line from the current TCP)."""


# ----------------------------------------------------------------------------------------------------------- solve
def solve_scene(scene_dir: str | Path, model: str | Mapping[str, Any] | ChatModel | None = None, *, turns: int = 30,
                seed: int = 0, block_timeout_s: float = 90.0, reply_tokens: int | None = None,
                cameras: Sequence[str] = ("front", "side", "wrist"), video: bool = True, env_type: str = DEFAULT_ENV,
                max_cost_usd: float | None = None, act_wait_s: float = 4 * 3600.0,
                log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run code.llm with privileged state on DIR/scene.json; returns (and writes DIR/solve.json) the outcome.
    `act_wait_s`: how long the kernel waits for the model's next motion (a shared local model may queue calls)."""
    from .agents import CodeLLMAgent

    say = log or (lambda msg: None)
    d = Path(scene_dir)
    path = (d / "scene.json") if d.is_dir() else d
    d = path.parent
    scene = Scene.model_validate_json(path.read_text())
    cams = list(cameras)
    if video and "front" not in cams:
        cams.append("front")
    env_params = {"scene": str(path.resolve()), "cameras": cams, "segmentation": True, "privileged_obs": True}
    env = FrameRecorder(registry.build("envs", {"type": env_type, "params": env_params}), "front")
    tools = {"type": PRIVILEGED_TOOLS}
    agent = CodeLLMAgent(model=model or DEFAULT_MODEL, tools=tools, max_turns=turns, block_timeout_s=block_timeout_s,
                         reply_tokens=reply_tokens, max_cost_usd=max_cost_usd, act_wait_s=act_wait_s)
    task = TaskSpec(id=f"scene/{scene.name}", version="1", instruction=scene.instruction,
                    params={"scene": str(path.resolve())}, max_steps=scene.max_steps, tags=("scene", "privileged"))
    say(f"solving {scene.name!r}: {scene.instruction}")
    run = run_recorded_episode(env, agent, task, d / "solve", seed=seed, video=d / "front.mp4" if video else None)
    result = {"scene": str(path), "instruction": scene.instruction, "model": describe(agent.model), "seed": seed,
              "privileged": True, "env": {"type": env_type, "params": env_params}, "tools": PRIVILEGED_TOOLS, **run}
    (d / "solve.json").write_text(json.dumps(result, indent=1, default=str))
    say(f"{run['status']} in {run['turns']} turns, {run['steps']} steps, {run['wall_s']:.0f} s ({run['reason']})")
    return result


# ------------------------------------------------------------------------------------------------------------- CLI
def _model_arg(spec: str | None) -> Any:
    if not spec:
        return None
    return json.loads(spec) if spec.lstrip().startswith("{") else spec


def scene_command(argv: list[str]) -> int:
    """scene builder: write a scene from task text (build), check it (check), solve it with privileged state (solve)"""
    ap = argparse.ArgumentParser(prog="hylo scene", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="a chat model writes a scene (--free: a scene program) for TASK, checked until it passes")
    b.add_argument("task")
    b.add_argument("--out", required=True)
    b.add_argument("--model", help="chat model spec (JSON or shorthand); default Ollama qwen3.5:35b-a3b")
    b.add_argument("--attempts", type=int, default=6)
    b.add_argument("--seeds", help="seeds the checker settles the scene with (default 0,1; free mode 0,1,2)")
    b.add_argument("--free", action="store_true", help="free mode: the model writes a Python scene program")
    c = sub.add_parser("check", help="check a scene file or a free scene program")
    c.add_argument("scene", help="a scene file, or a scene program (.py, or a directory holding scene.py)")
    c.add_argument("--seeds", help="default 0,1 (free scenes 0,1,2)")
    s = sub.add_parser("solve", help="code.llm solves DIR/scene.json (or DIR/scene.py) with privileged state")
    s.add_argument("dir")
    s.add_argument("--model")
    s.add_argument("--turns", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--block-timeout", type=float, default=90.0)
    s.add_argument("--reply-tokens", type=int, help="cap on each reply's tokens")
    s.add_argument("--act-wait", type=float, default=4 * 3600.0, help="seconds the kernel waits for the next motion")
    s.add_argument("--cameras", help="default front,side,wrist (free scenes: the robot's cameras)")
    s.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def seeds(default: str) -> list[int]:
        return [int(x) for x in (a.seeds or default).split(",") if x]

    from .scene_free import FreeSceneBuilder, is_free_scene, solve_free_scene

    if a.cmd == "build":
        if a.free:
            res = FreeSceneBuilder(_model_arg(a.model), attempts=a.attempts, seeds=seeds("0,1,2"),
                                   log=log).build(a.task, a.out)
        else:
            res = SceneBuilder(_model_arg(a.model), attempts=a.attempts, seeds=seeds("0,1"), log=log).build(a.task, a.out)
        print(json.dumps({k: res[k] for k in ("built", "attempts", "usage", "wall_s", "scene")}, indent=1))
        return 0 if res["built"] else 1
    if a.cmd == "check":
        if is_free_scene(a.scene):
            from .scene_free_check import FreeSceneChecker

            checker: Any = FreeSceneChecker(seeds=seeds("0,1,2"))
        else:
            checker = SceneChecker(seeds=seeds("0,1"))
        try:
            rep = checker.check(a.scene)
        finally:
            checker.close()
        print(rep.text())
        return 0 if rep.ok else 1
    cams = [c for c in a.cameras.split(",") if c] if a.cameras else None
    if is_free_scene(a.dir):
        res = solve_free_scene(a.dir, _model_arg(a.model), turns=a.turns, seed=a.seed, block_timeout_s=a.block_timeout,
                               reply_tokens=a.reply_tokens, act_wait_s=a.act_wait, cameras=cams,
                               video=not a.no_video, log=log)
    else:
        res = solve_scene(a.dir, _model_arg(a.model), turns=a.turns, seed=a.seed, block_timeout_s=a.block_timeout,
                          reply_tokens=a.reply_tokens, act_wait_s=a.act_wait,
                          cameras=cams or ["front", "side", "wrist"], video=not a.no_video, log=log)
    print(json.dumps({k: res.get(k) for k in ("status", "success", "steps", "turns", "wall_s", "reason", "video")},
                     indent=1, default=str))
    return 0 if res["success"] else 1


__all__ = ["DEFAULT_MODEL", "EXAMPLE_1", "EXAMPLE_2", "FrameRecorder", "PRIVILEGED_TOOLS", "PrivilegedSceneTools",
           "SceneBuilder", "builder_system", "extract_json", "scene_command", "scene_schema", "solve_scene",
           "write_mp4"]
