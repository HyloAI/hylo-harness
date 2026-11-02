"""Evaluate the reference programs through `hylo run` (agent code.program).

Suites: `manip` (default; mujoco/manip-v0 on mujoco.panda_table, six tasks), `bimanual` (mujoco/bimanual-v0 on
mujoco.aloha_table) and `humanoid` (mujoco/humanoid-v0 on mujoco.g1_table). One experiment per task (the program's
source in agent.params.source, every camera of the environment with its segmentation, no privileged state), run as
separate `hylo run` processes so the episode workers of different tasks do not share one interpreter. Writes
results.json and results.md (success per task, steps and wall time per episode) next to the runs.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/programs/run_programs.py --out runs/manip-programs

Options: --suite, --tasks (default: all of the suite), --split (validation), --workers (episode workers in total,
default 16, shared out over the tasks running at once), --parallel (tasks at once, default 6), --image-every
(default 0: no images in the traces).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TASKS = ("pick_place", "stack", "put_in_bowl", "push_to_zone", "press_by_number", "sort_by_colour")
SUITES = {
    "manip": {"benchmark": "mujoco/manip-v0", "env": "mujoco.panda_table", "cameras": ["front", "side", "wrist"],
              "tasks": TASKS, "steps": "hylo.envs.mujoco.manip_tasks"},
    "bimanual": {"benchmark": "mujoco/bimanual-v0", "env": "mujoco.aloha_table",
                 "cameras": ["overhead", "front", "left_wrist", "right_wrist"], "tasks": ("handover", "lift_box", "two_bins"),
                 "steps": "hylo.envs.mujoco.bimanual_tasks"},
    "humanoid": {"benchmark": "mujoco/humanoid-v0", "env": "mujoco.g1_table", "cameras": ["head", "front"],
                 "tasks": ("pick_place", "two_hands", "press_button"), "steps": "hylo.envs.mujoco.humanoid_tasks",
                 "file_prefix": "g1_"},
}


def config(task: str, split: str, out: Path, image_every: int, suite: str = "manip") -> dict:
    su = SUITES[suite]
    prefix = "manip" if suite == "manip" else suite
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"{prefix}-program-{task.replace('_', '-')}",
        "benchmark": {"type": su["benchmark"], "params": {"tasks": [task]}, "split": split},
        "env": {"type": su["env"], "params": {"cameras": list(su["cameras"]), "segmentation": True}},
        "agent": {"type": "code.program",
                  "params": {"source": (HERE / f"{su.get('file_prefix', '')}{task}.py").read_text(), "timeout_s": 600}},
        "trace": {"record_privileged": False, "image_every": image_every},
        "output_dir": str(out),
        "tags": ["reference-program", task],
    }


def episode_stats(run_dir: Path) -> list[dict]:
    """Per episode: seed, success, steps, wall seconds, program seconds and the program's last printed lines."""
    log = json.loads((run_dir / "evallog.json").read_text())
    out = []
    for s in log["samples"]:
        start = end = None
        prog: dict = {}
        with open(run_dir / s["trace"]) as f:
            for line in f:
                if '"episode.start"' in line[:40] or '"episode.end"' in line[:40] or '"agent.code.program"' in line[:40]:
                    ev = json.loads(line)
                    if ev["type"] == "episode.start":
                        start = ev["wall_time"]
                    elif ev["type"] == "episode.end":
                        end = ev["wall_time"]
                    else:
                        prog = ev["payload"]
        out.append({"seed": s["seed"], "success": s["status"] == "success", "steps": s["steps"],
                    "wall_s": round(end - start, 2) if start and end else None,
                    "program_s": prog.get("seconds"), "error": (prog.get("error") or "")[-300:] or None,
                    "output_tail": "\n".join((prog.get("output") or "").strip().splitlines()[-4:])})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suite", choices=sorted(SUITES), default="manip")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--split", default="validation")
    p.add_argument("--out", default="runs/manip-programs")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--image-every", type=int, default=0)
    a = p.parse_args()
    a.tasks = a.tasks or list(SUITES[a.suite]["tasks"])
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    # episode workers in proportion to the tasks' step budgets (long tasks get more), at least one each
    import importlib

    MAX_STEPS = importlib.import_module(SUITES[a.suite]["steps"]).MAX_STEPS

    budget = {t: MAX_STEPS.get(t, 500) for t in a.tasks}
    share = max(a.workers, len(a.tasks)) * min(1.0, a.parallel / len(a.tasks))
    workers = {t: max(1, round(share * budget[t] / sum(budget.values()))) for t in a.tasks}
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    pending, running, results = list(a.tasks), {}, {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.parallel:
            task = pending.pop(0)
            cfg_path = out / "configs" / f"{task}.yaml"
            cfg_path.write_text(yaml.safe_dump(config(task, a.split, out, a.image_every, a.suite), sort_keys=False))
            log = open(out / "configs" / f"{task}.log", "w")
            cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(cfg_path), "--set", f"workers={workers[task]}"]
            running[task] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env),
                             time.time(), log)
            print(f"started {task} ({workers[task]} workers)", flush=True)
        for task, (proc, started, log) in list(running.items()):
            if proc.poll() is None:
                continue
            stdout = proc.stdout.read() if proc.stdout else ""
            log.close()
            del running[task]
            try:
                info = json.loads(stdout[stdout.index("{"):])
            except ValueError:
                print(f"{task}: hylo run failed (exit {proc.returncode}); see configs/{task}.log", flush=True)
                results[task] = {"error": f"exit {proc.returncode}"}
                continue
            eps = episode_stats(Path(info["run_dir"]))
            walls = [e["wall_s"] for e in eps if e["wall_s"]]
            results[task] = {"run_dir": info["run_dir"], "success": info.get("success_mean"),
                             "n": len(eps), "steps_mean": info.get("steps_mean"),
                             "wall_s_mean": round(sum(walls) / len(walls), 1) if walls else None,
                             "task_wall_s": round(time.time() - started, 1), "episodes": eps}
            print(f"{task}: success {info.get('success_mean'):.2f} over {len(eps)} episodes "
                  f"({results[task]['wall_s_mean']} s per episode)", flush=True)
        time.sleep(2)
    (out / "results.json").write_text(json.dumps({"suite": a.suite, "split": a.split, "workers": a.workers,
                                                  "total_wall_s": round(time.time() - t0, 1), "tasks": results},
                                                 indent=1))
    lines = [f"# Reference programs on {SUITES[a.suite]['benchmark']} ({a.split})", "",
             "| task | success | episodes | steps (mean) | wall s / episode | failed seeds |", "|---|---|---|---|---|---|"]
    for task in a.tasks:
        r = results.get(task, {})
        if "episodes" not in r:
            lines.append(f"| {task} | error | | | | |")
            continue
        failed = ", ".join(str(e["seed"]) for e in r["episodes"] if not e["success"]) or "-"
        lines.append(f"| {task} | {r['success']:.2f} | {r['n']} | {r['steps_mean']:.0f} | {r['wall_s_mean']} | {failed} |")
    lines += ["", f"Total wall time {time.time() - t0:.0f} s with {a.workers} workers."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all("episodes" in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
