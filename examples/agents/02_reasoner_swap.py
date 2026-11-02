"""Reasoner swap: the same agent with a rule chooser, a local LLM chooser or a VLM chooser, one config line apart.

MuJoCo table reach (mujoco.table_reach): a planar arm must touch the pad named by a language instruction, either
direct ("touch the red pad") or indirect ("the pad the colour of a ripe tomato"); a third task adds a shove. Pads
are visible for the first 5 steps only, so the agent grounds the instruction once and then acts from spatial
memory. The agent is always `composable.object_reach` (ObjectReasoner + spatial memory + IK policy); only the
chooser that grounds the instruction changes:

    chooser: {type: chooser.rule_color}                              # colour words -> nearest RGB
    chooser: {type: chooser.llm, params: {model: qwen3.5:35b-a3b}}    # sees colour values as text
    chooser: {type: chooser.llm, params: {model: gpt-oss:20b}}
    chooser: {type: chooser.vlm, params: {model: qwen3.5:35b-a3b}}    # sees only the top-camera image (set-of-mark)

A language-free baseline (baseline.nearest_object) is the floor. Measured per chooser: success (overall and per
task), grounding accuracy (the chosen pad is the instructed one, from the trace's privileged state, which the
agent never sees) and latency per grounding decision (one call at a time, so it is latency, not queueing). LLM
responses are cached by prompt hash in OUT/llm-cache, so a rerun replays the same decisions; latencies are always
those of the original live call.

    uv run python examples/agents/02_reasoner_swap.py [--quick] [--out DIR] [--no-llm] [--no-vlm]

Runtime: about 5 min with live LLM calls, 15 s when cached (quick: 3 episodes per chooser, about 10 s once the
models are loaded).
Requires hylo-env-mujoco (bench), hylo-reasoner-llm and a local Ollama server (--base-url) with qwen3.5:35b-a3b and
gpt-oss:20b; the VLM rows need MuJoCo offscreen rendering (OSMesa in ~/.local/mesa or MUJOCO_GL=egl). Without
Ollama the LLM / VLM rows are skipped with a message.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

import numpy as np  # noqa: E402

SLUG = "reasoner_swap"
TASKS = {"mujoco/table_direct": "direct", "mujoco/table_indirect": "indirect", "mujoco/table_push": "push"}


def extra_args(ap) -> None:
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM and VLM choosers")
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM chooser (no rendering needed)")


def ollama_models(base_url: str) -> list[str] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:  # noqa: BLE001
        return None


def grounding(run_dir: Path, cache: Path) -> list[dict]:
    """Per episode: status, whether the first grounding decision picked the instructed pad, its latency."""
    from hylo.core.trace import TraceStore

    store, rows = TraceStore(run_dir), []
    for s in C.evallog(run_dir)["samples"]:
        tr = store.read(s["episode_id"])
        priv = tr.first("privileged.state")
        target = int(priv["payload"]["target"]) if priv else None
        first = next((e["payload"] for e in tr.of_type("agent.decision")
                      if (e["payload"].get("inputs") or {}).get("new_choice")), None)
        chosen = (first or {}).get("inputs", {}).get("target")
        meta = (first or {}).get("inputs", {}).get("chooser_meta") or {}
        latency = meta.get("latency_s")
        if meta.get("cached") and meta.get("prompt_hash") and (cache / f"{meta['prompt_hash']}.json").exists():
            latency = json.loads((cache / f"{meta['prompt_hash']}.json").read_text())["latency_s"]  # the live call's
        rows.append({"task": TASKS.get(s["task_id"], s["task_id"]), "status": s["status"],
                     "correct": None if chosen is None or target is None else int(chosen.split("_")[-1]) == target,
                     "latency_s": latency, "cached": bool(meta.get("cached")), "error": meta.get("error"),
                     "choosers": sum(1 for e in tr.of_type("agent.decision")
                                     if (e["payload"].get("inputs") or {}).get("new_choice"))})
    return rows


def main() -> int:
    a = C.parse_args(__doc__, SLUG, extra_args)
    t0 = time.perf_counter()
    models = None if a.no_llm else ollama_models(a.base_url)
    if not a.no_llm and models is None:
        C.log(f"Ollama is not reachable at {a.base_url}: LLM and VLM choosers skipped (rule-based rows only)")
    can_render = False
    if models is not None and not a.no_vlm:
        can_render = C.ensure_render_env()               # may re-execute this script with OSMesa set
        if not can_render:
            C.log("MuJoCo rendering unavailable (no OSMesa / MUJOCO_GL): VLM chooser skipped")
    cache = a.out / "llm-cache"
    n_seeds = 1 if a.quick else 20
    bench = {"type": "mujoco/table-v0", "params": {"validation": n_seeds}}
    vision = {"type": "mujoco/table-v0", "params": {"validation": n_seeds, "vision": True}}

    def chooser(kind: str, model: str) -> dict:
        return {"type": "composable.object_reach", "params": {"kp": 0.3, "chooser": {"type": kind, "params": {
            "model": model, "base_url": a.base_url, "cache_dir": str(cache)}}}}

    rows_spec: list[tuple[str, str, dict, dict]] = [
        ("baseline.nearest_object", "none (no language)", {"type": "baseline.nearest_object"}, bench),
        ("chooser.rule_color", "colour words", {"type": "composable.object_reach",
                                                "params": {"kp": 0.3, "chooser": {"type": "chooser.rule_color"}}}, bench)]
    for model in ("qwen3.5:35b-a3b", "gpt-oss:20b"):
        if models is not None and model in models and not (a.quick and model != "qwen3.5:35b-a3b"):
            rows_spec.append((f"chooser.llm [{model}]", "colour values as text", chooser("chooser.llm", model), bench))
        elif models is not None:
            C.log(f"skipped chooser.llm [{model}]: " + ("quick mode" if model in models else "model not pulled"))
    if can_render and "qwen3.5:35b-a3b" in (models or []):
        rows_spec.append(("chooser.vlm [qwen3.5:35b-a3b]", "top-camera image only", chooser("chooser.vlm", "qwen3.5:35b-a3b"),
                          vision))
    rows = []
    for name, sees, agent, b in rows_spec:
        t = time.perf_counter()
        llm = "llm" in name or "vlm" in name
        r = C.run(name.split(" ")[0].replace(".", "-"), agent, b, "validation", a.out / "runs",
                  env="mujoco.table_reach", workers=1 if llm else a.workers, safety={"max_delta": 0.5},
                  trace={"image_every": 1 if "vlm" in name else 0})
        eps = grounding(r.run_dir, cache)
        k, n = sum(e["status"] == "success" for e in eps), len(eps)
        lo, hi = C.wilson(k, n)
        lat = [e["latency_s"] for e in eps if e["latency_s"] is not None]
        acc = [e["correct"] for e in eps if e["correct"] is not None]
        row = {"chooser": name, "sees": sees, "episodes": n, "success_%": round(100 * k / n, 1),
               "ci95_%": f"{100 * lo:.0f}-{100 * hi:.0f}",
               **{f"{t_}_%": round(100 * np.mean([e['status'] == 'success' for e in eps if e['task'] == t_]), 1)
                  for t_ in TASKS.values()},
               "grounding_%": round(100 * float(np.mean(acc)), 1) if acc else None,
               "latency_s": round(float(np.median(lat)), 2) if lat and llm else None,
               "calls": sum(e["choosers"] for e in eps) if llm else None,
               "cached": sum(e["cached"] for e in eps) if llm else None,
               "errors": sum(1 for e in eps if e["error"]), "wall_s": round(time.perf_counter() - t, 1),
               "agent": agent, "run": str(r.run_dir)}
        rows.append(row)
        C.log(f"{name:<32} success {row['success_%']:5.1f}%  grounding {row['grounding_%']}%  "
              f"latency {row['latency_s']} s  ({row['wall_s']} s)")
    print()
    C.table(rows, ["chooser", "sees", "success_%", "ci95_%", "direct_%", "indirect_%", "push_%", "grounding_%",
                   "latency_s", "calls", "cached"],
            title=f"MuJoCo table reach, validation split ({rows[0]['episodes']} episodes per row; latency = median "
                  "per grounding call, live)")
    C.write_result(a.out, {"example": f"agents/{SLUG}", "quick": a.quick, "benchmark": bench, "split": "validation",
                           "ollama": a.base_url if models is not None else None,
                           "renderer": os.environ.get("MUJOCO_GL"), "rows": rows,
                           "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
