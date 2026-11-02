"""LLM proposer against rule-based proposers at equal budget, on MuJoCo planar reach.

What it shows
    The same parent (a flow policy trained here, run with too much sampling noise), the same gate and the same
    budget for every generator: `cycles` campaign cycles of at most K candidates each. Generators:
        rule.steering   the stock rule generator for steering (failure buckets -> sampling, voting, replanning)
        rule.gain       scales one knob (the noise scale) by fixed factors
        llm.steering    a local LLM (Ollama) reads the search-split diagnosis and the archive's verdicts and proposes
                        complete steering configs, each with a hypothesis (hylo-proposer-llm)
    For each: what it proposed, how far each candidate got in the cascade, what was promoted, the final version's
    success on 2 x 250 fresh held-out episodes, wall time and LLM calls. LLM replies are cached under <out>, so a
    rerun reproduces the same proposals without calling the model.

Command (from the harness repository root)
    uv run python examples/evolve/05_llm_vs_rule.py [--quick] [--out DIR] [--model qwen3.5:35b-a3b]

Typical runtime: about 2 min on CPU with qwen3.5:35b-a3b served by Ollama (--quick: about 1 min, one LLM call;
seconds when the replies are cached).
Requires: hylo-env-mujoco (hylo-bench), hylo-proposer-llm, an Ollama server at http://localhost:11434 with the model
pulled. Without the server the LLM arm is skipped (reported in result.json) and the rule arms still run.
Output: <out>/result.json, <out>/llm-cache, one campaign directory per generator under <out>/arms.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request

from _common import flow_agent, parse_args, pool_outcomes, reach_config, require, table, train_flow, wilson, write_result

SLUG = "llm_vs_rule"
STEERING = "agent.params.steering"


def ollama_has(base_url: str, model: str) -> str | None:
    """None when the model is served, else the reason it is not."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception as exc:  # noqa: BLE001
        return f"Ollama not reachable at {base_url} ({exc.__class__.__name__})"
    return None if model in names else f"model {model!r} not pulled on {base_url} (have: {names})"


def run_arm(name, proposer, base, out, cycles, k, log=print):
    """`cycles` campaign cycles (run_cycle, promoted candidate = next parent), at most k candidates per cycle."""
    from hylo.eval.experiment import run_experiment
    from hylo.evolve.protocol import Archive, CandidateRegistry, ProtocolConfig, run_cycle
    from hylo.evolve.steering import diagnose_traces

    root = out / "arms" / name.replace(".", "_")
    shutil.rmtree(root, ignore_errors=True)
    reg, archive = CandidateRegistry(root / "registry"), Archive(root / "archive.jsonl")
    proto = ProtocolConfig(alpha=0.05, gate_looks=2, stage1_pairs=20, max_candidates=k)
    cyc, t0, episodes = [], time.time(), [0]

    def evaluate(cfg, split, label):
        r = run_experiment(cfg, split=split, run_root=root / "runs", label=label)
        episodes[0] += r.metrics["n_episodes"]
        return r

    for c in range(1, cycles + 1):
        t = time.time()
        rep = run_cycle(base, reg, archive, proposer, evaluate, diagnose_traces, protocol=proto)
        invalid = list(getattr(proposer, "rejected", []) or [])
        entries = []
        for e in rep.entries:
            s2 = next((s for s in e.stages if s.stage == 2), None)
            g = [s for s in e.stages if s.stage == 3]
            entries.append({"id": e.candidate.id, "patch": dict(e.candidate.patch), "steering": e.candidate.patch.get(STEERING),
                            "hypothesis": e.hypothesis,
                            "verdict": e.verdict, "search_val": s2.success if s2 else None,
                            "parent_search_val": s2.parent_success if s2 else None,
                            "gate": [f"{s.wins}:{s.losses} p={s.p_value:.2g}" for s in g]})
        cyc.append({"cycle": c, "proposed": len(rep.entries), "invalid_proposals": len(invalid),
                    "invalid_reasons": [str(x.get("error", ""))[:200] for x in invalid], "entries": entries,
                    "promoted": rep.promoted, "heldout": rep.heldout, "elapsed_s": round(time.time() - t, 1)})
        log(f"  [{name}] cycle {c}: {len(rep.entries)} candidates ({len(invalid)} invalid dropped), promoted {rep.promoted} "
            f"({time.time() - t:.0f}s)")
    final = base.with_overrides(reg.effective_patch()) if reg.active else base
    return {"cycles": cyc, "lineage": reg.lineage() if reg.active else [], "final_steering": final.agent.params["steering"],
            "episodes": episodes[0], "elapsed_s": round(time.time() - t0, 1)}, final


def main() -> None:
    def extra(ap):
        ap.add_argument("--model", default="qwen3.5:35b-a3b")
        ap.add_argument("--base-url", default="http://localhost:11434")

    a = parse_args(__doc__, SLUG, extra)
    require("envs", "mujoco.planar_reach")
    from hylo.core.registry import registry
    from hylo.eval.scoring import paired_compare

    t0 = time.time()
    cycles, k = (1, 2) if a.quick else (2, 3)
    flow = train_flow(a.out, epochs=60 if a.quick else 300)
    base = reach_config(flow_agent(flow, noise_scale=3.0), "evolve-llm-vs-rule", protocol="rsi", gate_looks=2)
    arms = {
        "rule.steering": {"type": "rule.steering", "params": {"path": STEERING}},
        "rule.gain": {"type": "rule.gain", "params": {"path": f"{STEERING}.noise_scale", "factors": [0.5, 0.75, 0.9][:k]}},
        "llm.steering": {"type": "llm.steering", "params": {"model": a.model, "base_url": a.base_url, "n_candidates": k, "path": STEERING,
                                                            "cache_dir": str(a.out / "llm-cache"), "timeout_s": 600}},
    }
    skipped = {}
    try:
        registry.reference("generators", "llm.steering")
        why = ollama_has(a.base_url, a.model)
    except KeyError:
        why = "hylo-proposer-llm is not installed"
    if why:
        skipped["llm.steering"] = why
        arms.pop("llm.steering")
        print(f"skipping llm.steering: {why}")

    results, finals = {}, {}
    for name, spec in arms.items():
        proposer = registry.build("generators", spec)
        before = set((a.out / "llm-cache").glob("*.json"))
        results[name], finals[name] = run_arm(name, proposer, base, a.out, cycles, k)
        if name == "llm.steering":
            new = sorted(set((a.out / "llm-cache").glob("*.json")) - before)
            results[name]["llm"] = {"model": a.model, "requests": cycles, "calls": len(new), "cached": cycles - len(new),
                                    "latency_s": [round(json.loads(p.read_text()).get("latency_s") or 0, 1) for p in new]}

    # ---- every final version (and the parent) on the same fresh held-out episodes
    n_seeds = 30 if a.quick else 250
    seeds = range(60_000, 60_000 + n_seeds)
    parent_rows = {r["key"]: r["success"] for r in pool_outcomes(base, seeds, workers=a.workers, tmp_root=a.out / "tmp")}
    conf = {"parent": {"success": sum(v for v in parent_rows.values() if v is not None) / len(parent_rows)}}
    for name, cfg in finals.items():
        rows = {r["key"]: r["success"] for r in pool_outcomes(cfg, seeds, workers=a.workers, tmp_root=a.out / "tmp")}
        ok = [v for v in rows.values() if v is not None]
        pc = paired_compare(rows, parent_rows)
        conf[name] = {"success": sum(ok) / len(ok), "ci95": wilson(sum(ok), len(ok)),
                      "vs_parent": {"wins": pc.candidate_wins, "losses": pc.parent_wins, "p": pc.p_value}}
        results[name]["confirmation"] = conf[name]
    shutil.rmtree(a.out / "tmp", ignore_errors=True)

    print(f"\ncandidates ({cycles} cycle(s), at most {k} per cycle; search_val: candidate vs parent)")
    rows = []
    for name, r in results.items():
        for c in r["cycles"]:
            for e in c["entries"]:
                st = e["steering"] or {}
                short = {kk: st[kk] for kk in ("n_samples", "selector", "noise_scale", "execution_steps") if st.get(kk) is not None}
                if st.get("rules"):
                    short["rules"] = [x["name"] for x in st["rules"]]
                short.update({kk.split(".")[-1]: v for kk, v in e["patch"].items() if kk != STEERING})
                sv = f"{e['search_val']:.2f} vs {e['parent_search_val']:.2f}" if e["search_val"] is not None else "-"
                rows.append([name, c["cycle"], json.dumps(short)[:70], sv, " ".join(e["gate"]) or "-", e["verdict"]])
    table(rows, ["generator", "cycle", "steering", "search_val", "gate", "verdict"])
    print(f"\nfinal version on {2 * n_seeds} fresh held-out episodes (parent {conf['parent']['success']:.3f})")
    table([[n, len(r["lineage"]), f"{r['confirmation']['success']:.3f}",
            f"{r['confirmation']['vs_parent']['wins']}:{r['confirmation']['vs_parent']['losses']}",
            r["episodes"], f"{r['elapsed_s']:.0f}", r.get("llm", {}).get("requests", 0)] for n, r in results.items()],
          ["generator", "promotions", "success", "vs parent", "episodes used", "wall s", "LLM requests"])
    for n, why in skipped.items():
        print(f"  {n}: skipped ({why})")

    write_result(a.out, {"example": SLUG, "quick": a.quick, "cycles": cycles, "max_candidates": k, "arms": results,
                         "parent_confirmation": conf["parent"], "skipped": skipped, "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
