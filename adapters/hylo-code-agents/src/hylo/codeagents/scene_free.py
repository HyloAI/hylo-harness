"""Free-form scene building (RFC-0006 part 8, free mode): task text -> a scene program -> checked -> solved.

    hylo scene build --free "lift the cable by both ends" --out runs/scenes_free/cable [--model SPEC] [--attempts 6]
    hylo scene check runs/scenes_free/cable            (a directory with scene.py, or the .py itself)
    hylo scene solve runs/scenes_free/cable [--model SPEC] [--turns 30]

The fixed format (`hylo.scene/v0.1`, scenes.py) describes a scene as data: primitives on a Panda's table and a
predicate from a small language. Free mode lets the model write the task as a Python program instead
(`hylo.freescene/v0.1`, run by the env `mujoco.free_scene`): it chooses the robot (Panda, the two ALOHA arms or the
G1 humanoid), builds the world with MuJoCo's model editor or MJCF (cables, cloth, tendons, equality constraints,
hinged and sliding parts: anything MuJoCo compiles, except actuators and keyframes), and writes success(sim) over
the simulation state. Optional demo(sim) proves the task can be done with invisible hands; the checker runs it.

build: the code agents' turn loop (turns.CodeLoop) with a scene-program runner in place of the robot sandbox. Each
turn the model replies with one ```python block holding the whole program; the runner validates it with a wider
allow-list than robot programs (scene programs may import mujoco and use the model editor; still no files, no
underscores, no exec), saves it, and runs the free checker (scene_free_check.py) under a time limit. The checker's
numbered problems are what the model reads next, until the program passes or the attempt budget is spent. Writes
DIR/scene.py (the last attempt), DIR/attempts/NN.py and DIR/build.json (per attempt: problems, tokens, latency).

solve: `code.llm` on `mujoco.free_scene` with privileged state: PrivilegedFreeTools adds robot.scene() (rigid bodies,
cable nodes, flex vertices, joint values and sites from the simulator), robot.points(name), robot.top_down(xyz) and
robot.scene_table(). Two-arm robots use the toolkit's multi-arm verbs (arm=..., robot.run({arm: plan}, sync=True)).
Writes DIR/solve/trace, DIR/solve/turns.jsonl, DIR/solve.json and DIR/front.mp4, like the fixed-format solve
(scene_episode.run_recorded_episode). `agent=` runs another agent instead, e.g. code.program with a scripted
solution (written to DIR/<out_name>.json and DIR/<out_name>.mp4).

Modules: scene_free_check (the program allow-list and the checker), scene_free_prompt (what the model is told),
scene_free_tools (the privileged toolkit); this file only orchestrates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hylo.core.registry import registry
from hylo.core.task import TaskSpec
from hylo.reasoners.llm.chat import ChatModel, build_chat, describe

from .sandbox import BlockResult, CodeRejected
from .scene_check import problem
from .scene_episode import FrameRecorder, run_recorded_episode
from .scene_free_check import FREE_ENV, FreeSceneChecker, scene_program_path, validate_scene_program
from .scene_free_prompt import feedback_text, free_builder_system
from .scene_free_tools import PRIVILEGED_FREE_TOOLS
from .scenes import DEFAULT_MODEL
from .turns import CodeLoop


class SceneProgramRunner:
    """What CodeLoop runs each turn in free mode: validate the block as a scene program, save it, check it.
    Duck-types the Sandbox's run(); the last CheckReport stays in `report`."""

    def __init__(self, out: Path, checker: FreeSceneChecker) -> None:
        self.out, self.checker = out, checker
        self.k = 0
        self.report: Any = None
        self.problems: list[Any] = []

    def run(self, code: str) -> BlockResult:
        self.k += 1
        t0 = time.monotonic()
        self.report, self.problems = None, []
        try:
            validate_scene_program(code)
        except CodeRejected as exc:
            self.problems = [problem("program", str(exc))]
            return BlockResult(error=str(exc), rejected=True, seconds=round(time.monotonic() - t0, 2))
        (self.out / "attempts" / f"{self.k:02d}.py").write_text(code.rstrip() + "\n")
        path = self.out / "scene.py"
        path.write_text(code.rstrip() + "\n")
        self.report = self.checker.check(path)
        self.problems = list(self.report.problems)
        return BlockResult(output=feedback_text(self.report), seconds=round(time.monotonic() - t0, 2))



class FreeSceneBuilder:
    """A chat model writes a scene program; the free checker's problems go back to it until it passes."""

    def __init__(self, model: str | Mapping[str, Any] | ChatModel | None = None, *, attempts: int = 6,
                 seeds: Sequence[int] = (0, 1, 2), reply_tokens: int | None = None, timeout_s: float = 600.0,
                 log: Callable[[str], None] | None = None) -> None:
        model_spec = model or {**DEFAULT_MODEL, "max_tokens": 6000}
        self.model = build_chat(model_spec)
        self.attempts, self.seeds, self.reply_tokens, self.timeout_s = attempts, tuple(seeds), reply_tokens, timeout_s
        self.say = log or (lambda msg: None)

    def build(self, task: str, out_dir: str | Path) -> dict[str, Any]:
        out = Path(out_dir)
        (out / "attempts").mkdir(parents=True, exist_ok=True)
        checker = FreeSceneChecker(seeds=self.seeds, timeout_s=self.timeout_s)
        runner = SceneProgramRunner(out, checker)
        records: list[dict[str, Any]] = []
        t0 = time.monotonic()
        loop = CodeLoop(self.model, runner, system=free_builder_system(),       # type: ignore[arg-type]
                        first=f"Task: {task.strip()}\n\nWrite the scene program.", emit=records.append,
                        status=lambda: None, reply_tokens=self.reply_tokens, max_history_turns=3,
                        max_feedback=6000, max_context_chars=60_000)
        built = False
        attempts: list[dict[str, Any]] = []
        try:
            while loop.turn < self.attempts:
                before = runner.k
                rec = loop.step({"attempts": self.attempts - loop.turn - 1})
                a: dict[str, Any] = {"attempt": loop.turn, **{k: rec.get(k) for k in ("model", "tokens_in", "tokens_out",
                                                                                      "latency_s", "cost_usd")}}
                if rec.get("model_error"):
                    a.update(ok=False, model_error=rec["model_error"], problems=[{"code": "model_error"}])
                elif rec.get("code") is None:
                    a.update(ok=False, problems=[{"code": "format", "message": "no ```python block in the reply"}])
                else:
                    rep = runner.report if runner.k > before else None
                    a.update(ok=bool(rep is not None and rep.ok),
                             problems=[p.model_dump(mode="json") for p in runner.problems],
                             warnings=[p.model_dump(mode="json") for p in (rep.warnings if rep else ())],
                             stages=list(rep.stages) if rep else ["program"], check_s=rec.get("exec_s"),
                             file=f"attempts/{runner.k:02d}.py" if rep is not None else None)
                attempts.append(a)
                verdict = "passed" if a["ok"] else f"{len(a['problems'])} problem(s)"
                self.say(f"attempt {loop.turn}: {verdict} ({float(rec.get('latency_s') or 0):.0f} s model)")
                if a["ok"]:
                    built = True
                    break
                if loop.done:
                    break
        finally:
            checker.close()
        with open(out / "turns.jsonl", "w") as fh:
            for r in records:
                fh.write(json.dumps({k: v for k, v in r.items() if k not in ("images",)}, default=str) + "\n")
        result = {"task": task, "mode": "free", "model": describe(self.model), "built": built,
                  "attempts": len(attempts), "records": attempts, "usage": loop.usage.as_dict(),
                  "wall_s": round(time.monotonic() - t0, 1), "scene": str(out / "scene.py")}
        (out / "build.json").write_text(json.dumps(result, indent=1, default=str))
        return result




def solve_free_scene(scene_dir: str | Path, model: str | Mapping[str, Any] | ChatModel | None = None, *,
                     turns: int = 30, seed: int = 0, block_timeout_s: float = 120.0, reply_tokens: int | None = None,
                     cameras: Sequence[str] | None = None, video: bool = True, max_cost_usd: float | None = None,
                     act_wait_s: float = 4 * 3600.0, agent: Any = None, out_name: str = "solve",
                     log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """code.llm with privileged state on DIR/scene.py (or `agent`, e.g. code.program with a scripted solution);
    returns, and writes to DIR/<out_name>.json, the outcome. cameras: None for the robot's own cameras."""
    from .agents import CodeLLMAgent

    say = log or (lambda msg: None)
    path = scene_program_path(scene_dir).resolve()
    d = path.parent
    params: dict[str, Any] = {"scene": str(path), "segmentation": True, "privileged_obs": True}
    if cameras is not None:
        params["cameras"] = list(cameras) + (["front"] if video and "front" not in cameras else [])
    env = FrameRecorder(registry.build("envs", {"type": FREE_ENV, "params": params}), "front")
    program = env.program
    if agent is None:
        agent = CodeLLMAgent(model=model or DEFAULT_MODEL, tools={"type": PRIVILEGED_FREE_TOOLS}, max_turns=turns,
                             block_timeout_s=block_timeout_s, reply_tokens=reply_tokens, max_cost_usd=max_cost_usd,
                             act_wait_s=act_wait_s)
    task = TaskSpec(id=f"free/{d.name}", version="1", instruction=program.instruction, params={"scene": str(path)},
                    max_steps=program.max_steps, tags=("scene", "free", "privileged"))
    say(f"solving {d.name!r}: {program.instruction}")
    clip = d / ("front.mp4" if out_name == "solve" else f"{out_name}.mp4")
    run = run_recorded_episode(env, agent, task, d / out_name, seed=seed, video=clip if video else None)
    result = {"scene": str(path), "instruction": program.instruction, "robot": program.robot,
              "agent": getattr(agent, "name", type(agent).__name__),
              "model": describe(agent.model) if hasattr(agent, "model") else None, "seed": seed, "privileged": True,
              "env": {"type": FREE_ENV, "params": params}, "tools": PRIVILEGED_FREE_TOOLS, **run}
    (d / f"{out_name}.json").write_text(json.dumps(result, indent=1, default=str))
    say(f"{run['status']} in {run['turns']} turns, {run['steps']} steps, {run['wall_s']:.0f} s ({run['reason']})")
    return result


def is_free_scene(path: str | Path) -> bool:
    """A scene program (scene.py, or a directory holding one and no scene.json)."""
    p = Path(path)
    if p.is_dir():
        return (p / "scene.py").is_file() and not (p / "scene.json").is_file()
    return p.suffix == ".py"


__all__ = ["FreeSceneBuilder", "SceneProgramRunner", "is_free_scene", "solve_free_scene"]
