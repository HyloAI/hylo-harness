"""A full improvement campaign on MuJoCo planar reach: baseline, diagnosis, candidates, gate, promotion, per cycle.

What it shows
    1. The parent: a flow-matching policy (trained here from IK-expert demonstrations) run with too much sampling
       noise (noise_scale 3.0), so it misses about half of the targets.
    2. `run_campaign` (what `hylo evolve run` does), cycle after cycle: the parent runs on `search` and its failures
       are diagnosed; the `rule.gain` generator proposes smaller noise scales; the cascade screens them on
       `search_val`; the group-sequential gate (2 looks, McNemar) confirms the best on `gate`; the winner is reported
       on `heldout` and promoted; it becomes the next cycle's parent. The campaign stops after a cycle without a
       promotion.
    3. Confirmation outside the campaign: every version in the lineage on 2 x 250 fresh held-out episodes (seeds the
       campaign never used), paired against the base version.

Command (from the harness repository root)
    uv run python examples/evolve/01_campaign_reach.py [--quick] [--out DIR]

Typical runtime: about 1 min on CPU (--quick: about 30 s).
Requires: hylo-env-mujoco (hylo-bench); matplotlib for the figure (optional).
Output: <out>/result.json, <out>/campaign_curve.png, the campaign directory (<out>/campaign: registry, archive,
cycle reports, active.json) and every run's traces.
"""

from __future__ import annotations

import json
import shutil
import time

import yaml

from _common import (flow_agent, parse_args, pool_outcomes, reach_config, require, table, train_flow, wilson,
                     write_result)

SLUG = "campaign_reach"
PATH = "agent.params.steering.noise_scale"


def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    from hylo.eval.scoring import paired_compare
    from hylo.evolve.campaign import EvolveConfig, run_campaign, status

    t0 = time.time()
    flow = train_flow(a.out, epochs=60 if a.quick else 300)
    # rsi_states 200: every split twice the default size (search_val and each gate look: 80 episodes); with the default
    # 100 the gate cannot confirm cycle 2's smaller gain (see 03_gate_power.py)
    exp = reach_config(flow_agent(flow, noise_scale=3.0), "evolve-campaign-reach", protocol="rsi", gate_looks=2,
                       rsi_states=100 if a.quick else 200)
    (a.out / "experiment.yaml").write_text(yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False))
    camp_dir = a.out / "campaign"
    shutil.rmtree(camp_dir, ignore_errors=True)              # a campaign resumes from its state: start fresh
    cfg = EvolveConfig(name="examples-campaign-reach", experiment=str(a.out / "experiment.yaml"),
                       proposer={"type": "rule.gain", "params": {"path": PATH, "factors": [0.5, 0.75]}},
                       protocol={"alpha": 0.05, "gate_looks": 2, "stage1_pairs": 20},
                       cycles=2 if a.quick else 5, stop_after_no_progress=1, approval="auto", out_dir=str(camp_dir))
    (a.out / "campaign.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    state = run_campaign(cfg, log=lambda s: print("  " + s))
    st = status(cfg)

    # ---- per-cycle summary from the cycle reports
    cycles = []
    for k in range(1, state.cycles_done + 1):
        rep = json.loads((camp_dir / f"cycle-{k:02d}" / "cycle_report.json").read_text())
        cands = []
        for e in rep["entries"]:
            s2 = next((s for s in e["stages"] if s["stage"] == 2), None)
            looks = [s for s in e["stages"] if s["stage"] == 3]
            cands.append({"id": e["candidate"]["id"], "noise_scale": e["candidate"]["patch"][PATH], "verdict": e["verdict"],
                          "search_val": s2 and s2["success"], "parent_search_val": s2 and s2["parent_success"],
                          "gate": [{"wins": s["wins"], "losses": s["losses"], "p": s["p_value"], "notes": s["notes"]} for s in looks]})
        cycles.append({"cycle": k, "parent": rep["parent_id"], "parent_search_success": rep["diagnosis"]["success"],
                       "diagnosis": {"n_failures": rep["diagnosis"]["n_failures"], "buckets": rep["diagnosis"]["buckets"]},
                       "candidates": cands, "promoted": rep["promoted"], "heldout": rep["heldout"]})

    print("\ncampaign cycles (search_val: candidate vs parent; gate: wins:losses per look)")
    rows = []
    for c in cycles:
        for e in c["candidates"]:
            gate = " ".join(f"{g['wins']}:{g['losses']} p={g['p']:.2g}" for g in e["gate"]) or "-"
            sv = f"{e['search_val']:.2f} vs {e['parent_search_val']:.2f}" if e["search_val"] is not None else "-"
            rows.append([c["cycle"], f"{c['parent_search_success']:.2f}", e["noise_scale"], sv, gate, e["verdict"]])
    table(rows, ["cycle", "parent search", "cand noise", "search_val", "gate looks", "verdict"])
    for c in cycles:
        if c["heldout"]:
            h = c["heldout"]
            print(f"  cycle {c['cycle']} held-out report (not used for selection): {h['candidate_success']:.3f} vs "
                  f"parent {h['parent_success']:.3f}, wins {h['paired']['candidate_wins']}:{h['paired']['parent_wins']}, "
                  f"p={h['paired']['p_value']:.2g}")
    print(f"  stopped: {state.stopped}; lineage {st['lineage']}; effective patch {st['effective_patch']}")

    # ---- confirmation on fresh seeds: every version of the lineage (base first)
    from hylo.evolve.candidate import CandidateRegistry

    reg = CandidateRegistry(camp_dir / "registry")
    versions = [("base", None, 3.0)]
    for cid in reversed(reg.lineage()) if reg.active else []:
        versions.append((cid, cid, reg.get(cid)["candidate"]["patch"][PATH]))
    n_seeds = 30 if a.quick else 250
    seeds = range(50_000, 50_000 + n_seeds)                  # disjoint from every split the campaign used
    confirm, base_keys = [], None
    for name, cid, noise in versions:
        rows_ = pool_outcomes(exp.with_overrides({PATH: noise}), seeds, workers=a.workers, tmp_root=a.out / "tmp")
        keys = {r["key"]: r["success"] for r in rows_}
        ok = [v for v in keys.values() if v is not None]
        lo, hi = wilson(sum(ok), len(ok))
        row = {"version": name, "noise_scale": noise, "n": len(ok), "success": sum(ok) / len(ok), "ci95": [lo, hi]}
        if base_keys is None:
            base_keys = keys
        else:
            pc = paired_compare(keys, base_keys)
            row["vs_base"] = {"wins": pc.candidate_wins, "losses": pc.parent_wins, "p": pc.p_value}
        confirm.append(row)
    print(f"\nconfirmation on {2 * n_seeds} fresh held-out episodes per version (seeds 50000+)")
    table([[r["version"], r["noise_scale"], f"{r['success']:.3f}", f"{r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}",
            f"{r['vs_base']['wins']}:{r['vs_base']['losses']} p={r['vs_base']['p']:.1e}" if "vs_base" in r else "-"]
           for r in confirm], ["version", "noise", "success", "95% CI", "vs base (paired)"])

    result = {"example": SLUG, "quick": a.quick, "campaign_dir": str(camp_dir), "stopped": state.stopped,
              "cycles": cycles, "lineage": st["lineage"], "effective_patch": st["effective_patch"],
              "confirmation": confirm, "elapsed_s": round(time.time() - t0, 1)}
    plot(a.out / "campaign_curve.png", cycles, confirm)
    write_result(a.out, result)


def plot(path, cycles, confirm) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=110)
    xs = list(range(len(confirm)))
    ys = [r["success"] for r in confirm]
    err = [[y - r["ci95"][0] for y, r in zip(ys, confirm)], [r["ci95"][1] - y for y, r in zip(ys, confirm)]]
    ax.errorbar(xs, ys, yerr=err, color="#1f5fa8", marker="o", capsize=3, lw=2,
                label=f"active version, {confirm[0]['n']} fresh held-out episodes")
    for c in cycles:
        for e in c["candidates"]:
            if e["search_val"] is None:
                continue
            promoted = e["verdict"] == "promoted"
            ax.scatter(c["cycle"] - 0.08 + (0.16 if promoted else 0), e["search_val"], s=36, zorder=3,
                       facecolor="#c8442d" if promoted else "none", edgecolor="#c8442d")
            ax.annotate(f"{e['noise_scale']:g}", (c["cycle"] + 0.1, e["search_val"]), fontsize=7, color="#555")
    ax.scatter([], [], facecolor="#c8442d", edgecolor="#c8442d", label="candidate on search_val (filled: promoted)")
    ticks = list(range(max(len(confirm), len(cycles) + 1)))
    ax.set_xticks(ticks, ["base"] + [f"after cycle {i}" if i < len(confirm) else f"cycle {i}:\nnothing promoted"
                                     for i in ticks[1:]])
    ax.set_xlim(-0.3, ticks[-1] + 0.5)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Campaign on MuJoCo reach: flow policy, noise scale tuned by the gate", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
