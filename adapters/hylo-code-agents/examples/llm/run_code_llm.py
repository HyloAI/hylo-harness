"""Run the `code.llm` agent with a real chat model on mujoco/manip-v0 and summarise the traces.

One `hylo run` per (task, mode), cameras front / side / wrist with the environment's segmentation, validation split.
Modes: `zero` (no demonstration) and `one` (one demonstration: the reference program of a *different* task family,
see DEMO_FROM, to test transfer). Runs at most --parallel model sessions at once (a local model server queues the
rest anyway).

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/llm/run_code_llm.py run --out runs/code-llm \\
        --modes zero one --episodes 5 [--tasks stack ...] [--model JSON] [--parallel 3] [--max-turns 16]
    uv run python .../run_code_llm.py summarize runs/code-llm        # results.json + results.md from the traces
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROGRAMS = HERE.parent / "programs"
TASKS = ("pick_place", "stack", "put_in_bowl", "push_to_zone", "press_by_number", "sort_by_colour")
# one-shot: the demonstration comes from another task family
DEMO_FROM = {"pick_place": "put_in_bowl", "put_in_bowl": "pick_place", "stack": "pick_place",
             "sort_by_colour": "pick_place", "push_to_zone": "press_by_number", "press_by_number": "push_to_zone"}
QWEN = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True, "reasoning_effort": "none",
        "max_tokens": 3000, "timeout_s": 2400, "retries": 1}


def config(task: str, mode: str, a: argparse.Namespace, out: Path) -> dict:
    model = json.loads(a.model) if a.model else QWEN
    params = {"model": model, "max_turns": a.max_turns, "block_timeout_s": 60, "act_wait_s": 7200}
    if mode == "one":
        params["demonstration"] = (PROGRAMS / f"{DEMO_FROM[task]}.py").read_text()
    if a.skills:
        params["skills"] = a.skills
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"code-llm-{a.tag}-{mode}-{task.replace('_', '-')}",
        "benchmark": {"type": "mujoco/manip-v0", "params": {"tasks": [task], "validation": a.episodes},
                      "split": "validation"},
        "env": {"type": "mujoco.panda_table", "params": {"cameras": ["front", "side", "wrist"], "segmentation": True}},
        "agent": {"type": "code.llm", "params": params},
        "trace": {"record_privileged": False, "image_every": 0},
        "output_dir": str(out),
        "tags": ["code-llm", mode, task],
    }


def run(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    jobs = [(t, m) for m in a.modes for t in a.tasks]
    running: dict = {}
    while jobs or running:
        while jobs and len(running) < a.parallel:
            task, mode = jobs.pop(0)
            cfg = config(task, mode, a, out)
            path = out / "configs" / f"{mode}-{task}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            log = open(out / "configs" / f"{mode}-{task}.log", "w")
            cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(path)]
            running[(task, mode)] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env), log)
            print(f"{time.strftime('%H:%M:%S')} started {mode} {task}", flush=True)
        for key, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            stdout = proc.stdout.read() if proc.stdout else ""
            log.close()
            del running[key]
            try:
                info = json.loads(stdout[stdout.index("{"):])
                print(f"{time.strftime('%H:%M:%S')} {key[1]} {key[0]}: success {info.get('success_mean')} "
                      f"({info.get('run_dir')})", flush=True)
            except ValueError:
                print(f"{key}: hylo run failed (exit {proc.returncode})", flush=True)
        time.sleep(5)
    return 0


# ---------------------------------------------------------------- summaries from the traces

def episode(trace: Path) -> dict:
    turns, summary, start, end, prog = [], {}, None, None, {}
    with open(trace) as f:
        for line in f:
            head = line[:60]
            if '"agent.code.turn"' in head or '"agent.code.summary"' in head or '"episode.' in head:
                ev = json.loads(line)
                t, p = ev["type"], ev["payload"]
                if t == "agent.code.turn":
                    turns.append({k: p.get(k) for k in ("turn", "latency_s", "tokens_in", "tokens_out", "error",
                                                        "timed_out", "exec_s", "model_error", "blocks", "step")}
                                 | {"code": (p.get("code") or "")[:4000], "output": (p.get("output") or "")[-1500:],
                                    "reply": (p.get("reply") or "")[:600]})
                elif t == "agent.code.summary":
                    summary = p
                elif t == "episode.start":
                    start = ev["wall_time"]
                elif t == "episode.end":
                    end, prog = ev["wall_time"], p
    lat = [t["latency_s"] for t in turns if t.get("latency_s")]
    return {"turns": len(turns), "latency_mean_s": round(statistics.mean(lat), 1) if lat else None,
            "latency_max_s": round(max(lat), 1) if lat else None,
            "tokens_in": sum(t.get("tokens_in") or 0 for t in turns),
            "tokens_out": sum(t.get("tokens_out") or 0 for t in turns),
            "errors": sum(1 for t in turns if t.get("error")), "model_errors": sum(1 for t in turns if t.get("model_error")),
            "no_code": sum(1 for t in turns if not t.get("code") and not t.get("model_error")),
            "wall_s": round(end - start, 1) if start and end else None,
            "stopped": summary.get("stopped"), "finished": summary.get("finished"), "turn_log": turns}


def summarize(a: argparse.Namespace) -> int:
    root = Path(a.out)
    rows = []
    for run_dir in sorted(p.parent for p in root.glob("*/evallog.json")):
        log = json.loads((run_dir / "evallog.json").read_text())
        man = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
        name = man.get("config", {}).get("name") or run_dir.name
        for s in log["samples"]:
            ep = episode(run_dir / s["trace"])
            rows.append({"run": name, "run_dir": str(run_dir), "seed": s["seed"], "key": s.get("key"),
                         "episode_id": s["episode_id"], "success": s["status"] == "success", "steps": s["steps"],
                         "reason": s.get("termination_reason"), **ep})
    (root / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["run"], []).append(r)
    lines = ["| run | success | turns (mean) | latency s/turn | tokens in / out per episode | wall min / episode | "
             "stopped |", "|---|---|---|---|---|---|---|"]
    for name, rs in sorted(groups.items()):
        n = len(rs)
        ok = sum(r["success"] for r in rs)
        lat = [r["latency_mean_s"] for r in rs if r["latency_mean_s"]]
        walls = [r["wall_s"] for r in rs if r["wall_s"]]
        stops = ", ".join(sorted({str(r["stopped"])[:40] for r in rs}))
        lines.append(f"| {name} | {ok}/{n} | {statistics.mean(r['turns'] for r in rs):.1f} | "
                     f"{statistics.mean(lat) if lat else 0:.0f} | {statistics.mean(r['tokens_in'] for r in rs):.0f} / "
                     f"{statistics.mean(r['tokens_out'] for r in rs):.0f} | "
                     f"{statistics.mean(walls) / 60 if walls else 0:.1f} | {stops} |")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="runs/code-llm")
    r.add_argument("--tasks", nargs="*", default=list(TASKS))
    r.add_argument("--modes", nargs="*", default=["zero"], choices=["zero", "one"])
    r.add_argument("--episodes", type=int, default=5)
    r.add_argument("--model", help="chat model spec as JSON (default: qwen3.5:35b-a3b via Ollama, vision)")
    r.add_argument("--max-turns", type=int, default=16)
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--skills", help="program skill library directory")
    r.add_argument("--tag", default="qwen")
    s = sub.add_parser("summarize")
    s.add_argument("out")
    a = p.parse_args()
    return run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    sys.exit(main())
