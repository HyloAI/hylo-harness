"""Build, check, solve and record the free-mode showcase tasks with one command.

    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu
    uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py              # every task, every stage
    uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py --tasks cable_lift --stages build
    uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py --smoke      # scripted model, no
                                                                                                  # Ollama, no rendering

Tasks come from tasks.json (beside this file); every task gets a directory under --out (default: this directory).
Stages, in order, each skipped when its output already exists (--force redoes it):

- build     the model writes a scene program from the task text (hylo scene build --free): scene.py, build.json
- solve     code.llm solves it with privileged state (hylo scene solve): solve.json, solve/, front.mp4
- scripted  if DIR/scripted.py exists (a `solve(robot)` program written for the built scene, run by code.program):
            scripted.json, scripted/, scripted.mp4. Labelled as a scripted solution, never as the model's.
- newton    for tasks marked deformable: the demo and success function replayed in Newton on the CPU
            (newton_check.py, run through `uv run --with newton --with warp-lang`): newton_check.json
- blender   for the task marked "blender": the successful episode (model's, else scripted) re-simulated from its
            recorded actions and rendered by `hylo tool render` (Blender): blender.mp4
- collect   results.json in --out: every task's numbers, or "not run yet" where a stage has no output

One chat-model session at a time: the stages run one after another. Model: --model SPEC (default: the scene
builder's, Ollama qwen3.5:35b-a3b).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NOT_RUN = "not run yet"
STAGES = ("build", "solve", "scripted", "newton", "blender", "collect")


def reproduce(task: str, stage: str | None = None) -> str:
    cmd = f"uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py --tasks {task}"
    return cmd + (f" --stages {stage}" if stage else "")


# ---------------------------------------------------------------------------------------------------- stages
def stage_build(task: dict, d: Path, model: Any, attempts: int) -> dict:
    from hylo.codeagents.scene_free import FreeSceneBuilder

    return FreeSceneBuilder(model, attempts=attempts, log=print).build(task["text"], d)


def stage_solve(task: dict, d: Path, model: Any, video: bool, cameras: list[str] | None) -> dict:
    from hylo.codeagents.scene_free import solve_free_scene

    return solve_free_scene(d, model, turns=task.get("turns", 15), video=video, cameras=cameras, log=print)


def stage_scripted(task: dict, d: Path, video: bool, cameras: list[str] | None) -> dict:
    from hylo.codeagents.agents import CodeProgramAgent
    from hylo.codeagents.scene_free import solve_free_scene
    from hylo.codeagents.scene_free_tools import PRIVILEGED_FREE_TOOLS

    agent = CodeProgramAgent(source=(d / "scripted.py").read_text(), tools={"type": PRIVILEGED_FREE_TOOLS},
                             timeout_s=900)
    res = solve_free_scene(d, agent=agent, out_name="scripted", video=video, cameras=cameras, log=print)
    res["label"] = "scripted solution (privileged, hand-written for the built scene), not the model's"
    (d / "scripted.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def stage_newton(d: Path) -> None:
    cmd = ["uv", "run", "--with", "newton", "--with", "warp-lang", "python", str(HERE / "newton_check.py"), str(d)]
    subprocess.run(cmd, cwd=ROOT, check=False)


def replay_qpos(d: Path, run: dict) -> list:
    """qpos after reset and after every recorded action of a finished episode (re-simulated)."""
    import numpy as np

    from hylo.core import codec
    from hylo.core.registry import registry
    from hylo.core.task import TaskInstance, TaskSpec
    from hylo.core.trace import TraceStore
    from hylo.core.types import Action

    tr = TraceStore(run["trace"]).read(run["episode_id"])
    actions = [codec.decode(e["payload"]["data"]) for e in tr.of_type("action")]
    env = registry.build("envs", {"type": "mujoco.free_scene", "params": {"scene": str(d), "cameras": []}})
    task = TaskSpec(id="replay", version="1", instruction="replay", params={}, max_steps=len(actions) + 1)
    env.reset(TaskInstance(spec=task, seed=int(run.get("seed", 0))))
    qpos = [env.data.qpos.copy()]
    for a in actions:
        r = env.step(Action(data=np.asarray(a, float)))
        qpos.append(env.data.qpos.copy())
        if r.terminated or r.truncated:
            break
    (d / "world.xml").write_text(env.model_xml())
    env.close()
    return qpos


def stage_blender(d: Path, samples: int) -> dict:
    import numpy as np

    run = next((json.loads((d / f).read_text()) for f in ("solve.json", "scripted.json")
                if (d / f).is_file() and json.loads((d / f).read_text()).get("success")), None)
    if run is None:
        return {"written": False, "reason": "no successful episode to render"}
    qpos = replay_qpos(d, run)
    np.savez(d / "trajectory.npz", qpos=np.asarray(qpos), fps=20.0)
    out = d / "blender"
    cmd = ["uv", "run", "hylo", "tool", "render", str(d / "world.xml"), "--trajectory", str(d / "trajectory.npz"),
           "--backend", "blender", "--camera", "front", "--samples", str(samples), "--out", str(out)]
    done = subprocess.run(cmd, cwd=ROOT, check=False)
    return {"written": done.returncode == 0, "dir": str(out), "episode": run.get("agent"), "steps": len(qpos) - 1}


# --------------------------------------------------------------------------------------------------- collect
def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def collect(tasks: list[dict], out: Path) -> list[dict]:
    rows = []
    for t in tasks:
        d = out / t["name"]
        b, s, sc, nw = (_load(d / f) for f in ("build.json", "solve.json", "scripted.json", "newton_check.json"))
        row: dict[str, Any] = {"task": t["name"], "text": t["text"], "deformable": t.get("deformable", False),
                               "reproduce": reproduce(t["name"])}
        row["build"] = NOT_RUN if b is None else {
            "built": b["built"], "attempts": b["attempts"], "wall_s": b["wall_s"], "usage": b["usage"],
            "problems_per_attempt": [[p.get("code") for p in r.get("problems", [])] for r in b["records"]]}
        row["robot"] = s["robot"] if s else (NOT_RUN if b is None else None)
        for key, rec in (("solve", s), ("scripted", sc)):
            row[key] = NOT_RUN if rec is None else {
                k: rec.get(k) for k in ("agent", "model", "status", "success", "turns", "steps", "wall_s", "model_s",
                                        "reason", "usage", "video", "label") if k in rec}
        row["newton"] = (NOT_RUN if nw is None else nw) if t.get("deformable") else "not applicable"
        row["blender"] = ("written" if (d / "blender").is_dir() else NOT_RUN) if t.get("blender") else "not applicable"
        rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


# ------------------------------------------------------------------------------------------------------ smoke
SMOKE_SCENE = '''ROBOT = "aloha"
INSTRUCTION = "lift the cable by both ends"
MAX_STEPS = 300

def build(scene):
    t = scene.table_top
    scene.add_mjcf(f"""<mujoco><extension><plugin plugin="mujoco.elasticity.cable"/></extension><worldbody>
      <composite prefix="cable" type="cable" curve="s" count="17 1 1" size="0.44" offset="-0.22 0.05 {t + 0.012}"
                 initial="free">
        <plugin plugin="mujoco.elasticity.cable"><config key="twist" value="1e6"/><config key="bend" value="1e6"/>
          <config key="vmax" value="0"/></plugin>
        <joint kind="main" damping="0.02"/><geom type="capsule" size="0.012" density="300" condim="4"/>
      </composite></worldbody></mujoco>""")

def success(sim):
    n = sim.nodes("cable")
    return bool(min(n[0][2], n[-1][2]) > sim.table_top + 0.1)

def demo(sim):
    a, b = sim.pos("cableB_first"), sim.pos("cableB_last")
    sim.grab("cableB_first", hand=0)
    sim.grab("cableB_last", hand=1)
    sim.move_together({0: a + [0.03, 0, 0.18], 1: b + [-0.03, 0, 0.18]}, seconds=1.5)
'''

SMOKE_SOLUTION = '''s = robot.scene(); table = robot.scene_table()
n = s["cables"]["cable"]["nodes"]
plans = {}
for arm, i, j in (("left", 0, 1), ("right", -1, -2)):
    d = n[i] - n[j]; yaw = math.atan2(d[1], d[0]) + math.pi / 2
    p = n[i].copy(); p[2] = max(p[2], table.height + 0.017)
    plans[arm] = robot.plan(robot.grasp_at(robot.top_down(p, yaw), arm=arm), table)
robot.run(plans)
st = robot.status(); ups = {}
for arm in ("left", "right"):
    t = st["tcp"][arm].copy(); t[2, 3] += 0.15
    ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t), table)
robot.run(ups, sync=True)
print(robot.scene()["cables"]["cable"]["nodes"][[0, -1]])
'''


def smoke(out: Path) -> int:
    """The real build / check / solve / scripted / collect path with scripted chat models: no Ollama, no cameras."""
    from hylo.reasoners.llm.chat import ScriptedChat

    task = {"name": "smoke_cable", "text": "lift the cable by both ends", "turns": 3, "deformable": True}
    d = out / task["name"]
    builder = ScriptedChat([f"Two arms, one cable.\n```python\n{SMOKE_SCENE}\n```"])
    b = stage_build(task, d, builder, attempts=2)
    solver = ScriptedChat([f"Both ends.\n```python\n{SMOKE_SOLUTION}\n```"], then="```python\nfinish('done')\n```")
    s = stage_solve(task, d, solver, video=False, cameras=[])
    (d / "scripted.py").write_text("def solve(robot):\n" + "".join("    " + ln + "\n" for ln in
                                                                     SMOKE_SOLUTION.splitlines()))
    sc = stage_scripted(task, d, video=False, cameras=[])
    rows = collect([task], out)
    ok = b["built"] and s["success"] and sc["success"] and rows[0]["newton"] == NOT_RUN
    print(json.dumps({"smoke": "passed" if ok else "FAILED", "build_attempts": b["attempts"],
                      "solve": s["status"], "scripted": sc["status"], "out": str(out)}, indent=1))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", help="comma-separated task names (default: all in tasks.json)")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated, of {', '.join(STAGES)}")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--model", help="chat-model spec (JSON or shorthand); default Ollama qwen3.5:35b-a3b")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32, help="Blender samples per pixel")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo stages whose outputs exist")
    ap.add_argument("--smoke", action="store_true", help="scripted models only (no Ollama, no rendering)")
    a = ap.parse_args(argv)
    if a.smoke:
        out = Path(tempfile.mkdtemp(prefix="scenes_free_smoke_")) if a.out == str(HERE) else Path(a.out)
        return smoke(out)
    spec = json.loads((HERE / "tasks.json").read_text())["tasks"]
    names = a.tasks.split(",") if a.tasks else [t["name"] for t in spec]
    unknown = sorted(set(names) - {t["name"] for t in spec})
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}; tasks.json has {[t['name'] for t in spec]}")
    tasks = [t for t in spec if t["name"] in names]
    stages = a.stages.split(",")
    out = Path(a.out)
    model = (json.loads(a.model) if a.model and a.model.lstrip().startswith("{") else a.model) or None
    for t in tasks:
        d = out / t["name"]
        print(f"=== {t['name']}: {t['text']}", flush=True)
        if "build" in stages and (a.force or not (d / "build.json").is_file()):
            if a.force and d.is_dir():
                shutil.rmtree(d)
            stage_build(t, d, model, a.attempts)
        built = (_load(d / "build.json") or {}).get("built", False)
        if "solve" in stages and built and (a.force or not (d / "solve.json").is_file()):
            stage_solve(t, d, model, not a.no_video, None)
        if "scripted" in stages and (d / "scripted.py").is_file() and (a.force or not (d / "scripted.json").is_file()):
            stage_scripted(t, d, not a.no_video, None)
        if "newton" in stages and built and t.get("deformable") and (a.force or not (d / "newton_check.json").is_file()):
            stage_newton(d)
        if "blender" in stages and built and t.get("blender") and (a.force or not (d / "blender").is_dir()):
            print(stage_blender(d, a.samples))
    if "collect" in stages:
        collect(spec, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
