"""Evaluate the `tools.llm` agent (RFC-0007) with a real chat model and summarise the traces.

Suites:
  manip    mujoco/manip-v0 on mujoco.panda_table (cameras front / side / wrist, the environment's segmentation)
  libero   LIBERO in its tool mode (joint-position control, RGB-D, segmentation) in the LIBERO simulator venv
           (runtime libero-sim); `--perturb position language ...` adds perturbed variants of every task

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/tools_llm/run_tools_llm.py run --out runs/tools-llm \\
        [--suite manip] [--tasks pick_place stack] [--episodes 3] [--parallel 2] [--model JSON]
    uv run python .../run_tools_llm.py run --suite libero --tasks libero_goal:8 libero_spatial:0 --perturb position \\
        --episodes 2 --out runs/tools-llm-libero
    uv run python .../run_tools_llm.py summarize runs/tools-llm       # results.json + results.md

One `hylo run` per task (and perturbation), each its own process; at most --parallel model sessions run at once
(keep it at 1-2 for a local model server).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

TASKS = ("pick_place", "stack", "put_in_bowl", "push_to_zone", "press_by_number", "sort_by_colour")
QWEN = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True, "reasoning_effort": "none",
        "max_tokens": 4096, "timeout_s": 2400, "retries": 1}


def src_dir(module: str) -> str:
    spec = importlib.util.find_spec(module)
    return str(Path(list(spec.submodule_search_locations)[0]).parents[len(module.split(".")) - 1])


def config(job: dict, a: argparse.Namespace, out: Path) -> dict:
    model = json.loads(a.model) if a.model else QWEN
    agent = {"type": "tools.llm", "params": {"model": model, "max_decisions": a.max_decisions, "max_calls": a.max_calls,
                                              "observe_cameras": job["cameras"][:2], "act_wait_s": 7200}}
    if job["suite"] == "manip":
        return {"schema_version": "hylo.experiment/v0.1", "name": f"tools-llm-{a.tag}-{job['task'].replace('_', '-')}",
                "benchmark": {"type": "mujoco/manip-v0", "params": {"tasks": [job["task"]], "validation": a.episodes},
                              "split": "validation"},
                "env": {"type": "mujoco.panda_table", "params": {"cameras": job["cameras"], "segmentation": True}},
                "agent": agent, "trace": {"record_privileged": False, "image_every": 0}, "output_dir": str(out),
                "tags": ["tools-llm", job["task"]]}
    suite, idx = job["task"].split(":")
    bench = {"suite": suite, "task_indices": [int(idx)], "n_validation": a.episodes, "n_heldout": 0}
    if job.get("perturb"):
        bench["perturb"] = [job["perturb"]]
    return {"schema_version": "hylo.experiment/v0.1",
            "name": f"tools-llm-{a.tag}-{suite.removeprefix('libero_')}-{idx}-{job.get('perturb') or 'base'}",
            "benchmark": {"type": "libero", "split": "validation", "params": bench},
            "env": {"type": "libero", "params": {"control": "joint_position", "tool_obs": True, "segmentation": True},
                    "remote": {"runtime": "libero-sim",
                               "env": {"PYTHONPATH": os.pathsep.join([src_dir("hylo.core"), src_dir("hylo.envs.libero")])}}},
            "agent": agent, "trace": {"record_privileged": False, "image_every": 0}, "workers": 1,
            "output_dir": str(out), "tags": ["tools-llm", "libero", suite, job.get("perturb") or "base"]}


def jobs(a: argparse.Namespace) -> list[dict]:
    if a.suite == "manip":
        return [{"suite": "manip", "task": t, "cameras": ["front", "side", "wrist"]} for t in (a.tasks or TASKS)]
    out = []
    for t in a.tasks:
        for p in [None, *(a.perturb or [])] if not a.only_perturbed else (a.perturb or []):
            out.append({"suite": "libero", "task": t, "perturb": p, "cameras": ["agentview", "wrist"]})
    return out


def run(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    pending, running = jobs(a), []
    while pending or running:
        while pending and len(running) < a.parallel:
            job = pending.pop(0)
            cfg = config(job, a, out)
            path = out / "configs" / f"{cfg['name']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            log = open(out / f"{cfg['name']}.log", "w")
            running.append((cfg["name"], subprocess.Popen([sys.executable, "-m", "hylo.core.cli", "run", str(path)],
                                                          stdout=log, stderr=subprocess.STDOUT), time.time()))
            print(f"started {cfg['name']}", flush=True)
        time.sleep(5)
        for item in list(running):
            name, proc, t0 = item
            if proc.poll() is not None:
                running.remove(item)
                print(f"finished {name} (exit {proc.returncode}, {time.time() - t0:.0f} s)", flush=True)
    return summarize(a)


def summarize(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    rows = []
    for log in sorted(out.rglob("evallog.json")):
        run_dir = log.parent
        ev = json.loads(log.read_text())
        name = run_dir.name
        for s in ev["samples"]:
            tr = run_dir / s["trace"]
            summ: dict = {}
            start: dict = {}
            with open(tr) as f:
                for line in f:
                    if '"agent.tool.summary"' in line[:40]:
                        summ = json.loads(line)["payload"]
                    elif '"episode.start"' in line[:40]:
                        start = json.loads(line)["payload"]
            u = summ.get("usage") or {}
            rows.append({"run": name, "task": start.get("task_id"), "seed": s["seed"], "success": s["status"] == "success",
                         "steps": s["steps"], "decisions": summ.get("decisions"), "calls": summ.get("calls"),
                         "tokens_in": u.get("tokens_in"), "tokens_out": u.get("tokens_out"),
                         "model_s": u.get("latency_s"), "wall_s": summ.get("wall_s"), "stopped": summ.get("stopped"),
                         "outcome": (summ.get("outcome") or {}).get("outcome")})
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    lines = ["| task | success | decisions (mean) | model calls (mean) | tokens in / out (mean) | wall s (mean) |",
             "|---|---|---|---|---|---|"]
    for task, rs in sorted(by.items()):
        n = len(rs)
        mean = lambda k: sum((r[k] or 0) for r in rs) / n          # noqa: E731
        lines.append(f"| {task} | {sum(r['success'] for r in rs)}/{n} | {mean('decisions'):.1f} | {mean('calls'):.1f} | "
                     f"{mean('tokens_in'):.0f} / {mean('tokens_out'):.0f} | {mean('wall_s'):.0f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--suite", default="manip", choices=["manip", "libero"])
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--perturb", nargs="*")
    r.add_argument("--only-perturbed", action="store_true")
    r.add_argument("--episodes", type=int, default=3)
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--max-decisions", type=int, default=20)
    r.add_argument("--max-calls", type=int, default=60)
    r.add_argument("--model")
    r.add_argument("--tag", default="qwen")
    s = sub.add_parser("summarize")
    s.add_argument("out")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    sys.exit(main())
