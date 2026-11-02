"""Rollback: a candidate that passed the gate in simulation regresses in the field; a monitor detects it and the
registry restores the previous version.

What it shows
    1. A campaign with manual approval: the parent is a flow policy (trained here at the simulator's 20 Hz) with too
       much sampling noise; the gate confirms a lower-noise candidate in simulation, which waits in
       `pending_approval.json` until `approve()` (what `hylo evolve approve` does) promotes it and rewrites
       `active.json`, the file a deploying process follows.
    2. The field: a workcell that runs the same task at 10 Hz, a condition the gate never saw. The parent's field
       success was recorded before the update. After the update, a field monitor runs the deployed version (read
       from `active.json`) in batches of 20 episodes on the same field tasks and tests, after each batch, whether
       the parent did better (one-sided exact McNemar, alpha 0.01 per look).
    3. When it fires, `rollback()` (what `hylo evolve rollback` does) restores the parent, records who and why in
       the ledger and republishes `active.json`; the restored version's field success is measured again on new
       field episodes.

Command (from the harness repository root)
    uv run python examples/evolve/04_field_rollback.py [--quick] [--out DIR]

Typical runtime: about 35 s on CPU (--quick: about 20 s).
Requires: hylo-env-mujoco (hylo-bench).
Output: <out>/result.json, the campaign directory (<out>/campaign: registry and ledger, archive, active.json).
"""

from __future__ import annotations

import json
import shutil
import time

import yaml

from _common import flow_agent, parse_args, pool_outcomes, reach_config, require, table, train_flow, write_result

SLUG = "field_rollback"
PATH = "agent.params.steering.noise_scale"
FIELD = {"env.params": {"control_hz": 10.0}}       # the field workcell: same task, half the control rate
ALPHA_FIELD = 0.01


def main() -> None:
    a = parse_args(__doc__, SLUG)
    require("envs", "mujoco.planar_reach")
    from hylo.core.deployment import read_active
    from hylo.eval.scoring import one_sided_exact_mcnemar
    from hylo.evolve.campaign import EvolveConfig, approve, rollback, run_campaign
    from hylo.evolve.candidate import CandidateRegistry

    t0 = time.time()
    flow = train_flow(a.out, epochs=60 if a.quick else 300)
    exp = reach_config(flow_agent(flow, noise_scale=3.0), "evolve-field-rollback", protocol="rsi", gate_looks=2)
    (a.out / "experiment.yaml").write_text(yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False))
    camp = a.out / "campaign"
    shutil.rmtree(camp, ignore_errors=True)
    cfg = EvolveConfig(name="examples-field-rollback", experiment=str(a.out / "experiment.yaml"),
                       proposer={"type": "rule.gain", "params": {"path": PATH, "factors": [1 / 3]}},
                       protocol={"alpha": 0.05, "gate_looks": 2, "stage1_pairs": 20}, cycles=1,
                       approval="manual", out_dir=str(camp))

    # ---- 1. the campaign stops at the approval step
    state = run_campaign(cfg, log=lambda s: print("  " + s))
    pending = json.loads((camp / "pending_approval.json").read_text()) if (camp / "pending_approval.json").exists() else None
    if pending is None:
        print(f"no candidate passed the gate ({state.stopped}); nothing to deploy")
        write_result(a.out, {"example": SLUG, "quick": a.quick, "stopped": state.stopped})
        return
    cid, gate = pending["candidate_id"], pending["gate"]
    print(f"\n{cid} passed the gate in simulation (gate look: {gate['stage3']['wins']}:{gate['stage3']['losses']}, "
          f"p={gate['stage3']['p_value']:.2g}; held-out {gate['heldout']['candidate_success']:.3f} vs "
          f"{gate['heldout']['parent_success']:.3f}); waiting for approval: {state.stopped}")

    # ---- 2. the field: parent baseline recorded before the update, on the field tasks
    n_batches, per_batch = (3, 5) if a.quick else (10, 10)      # seeds per batch; every seed runs both tasks
    field_seeds = list(range(200_000, 200_000 + n_batches * per_batch))
    parent_agent = read_active(camp).agent                      # what the workcell runs before the update
    field_base = exp.with_overrides(FIELD)
    baseline = {r["key"]: r for r in pool_outcomes(field_base.with_overrides({"agent": parent_agent}), field_seeds,
                                                   workers=a.workers, tmp_root=a.out / "tmp")}
    b_ok = [r["success"] for r in baseline.values() if r["success"] is not None]
    print(f"field baseline (parent, 10 Hz workcell): success {sum(b_ok) / len(b_ok):.3f} on {len(b_ok)} episodes")

    approved = approve(cfg, by="operator")                      # promotes and rewrites active.json
    deployed = read_active(camp)
    print(f"approved {approved}; active.json now: version {deployed.version}, patch {deployed.patch}")

    # ---- the field monitor: deployed version, batch by batch, paired with the recorded baseline
    looks, seen, rolled = [], {}, None
    for b in range(n_batches):
        seeds = field_seeds[b * per_batch:(b + 1) * per_batch]
        rows = pool_outcomes(field_base.with_overrides({"agent": read_active(camp).agent}), seeds, workers=a.workers,
                             tmp_root=a.out / "tmp")
        seen.update({r["key"]: r for r in rows})
        pairs = [(baseline[k]["success"], r["success"]) for k, r in seen.items()
                 if r["success"] is not None and baseline[k]["success"] is not None]
        parent_wins = sum(1 for p, c in pairs if p and not c)
        active_wins = sum(1 for p, c in pairs if c and not p)
        p = one_sided_exact_mcnemar(parent_wins, active_wins)          # H1: the parent is better in the field
        look = {"batch": b + 1, "episodes": len(pairs), "active_success": sum(c for _, c in pairs) / len(pairs),
                "parent_success": sum(q for q, _ in pairs) / len(pairs), "parent_wins": parent_wins,
                "active_wins": active_wins, "p": p}
        looks.append(look)
        if p < ALPHA_FIELD:
            rolled = rollback(cfg, reason=f"field regression at 10 Hz: {look['active_success']:.2f} vs parent "
                                          f"{look['parent_success']:.2f} on {len(pairs)} paired field episodes "
                                          f"(McNemar p={p:.1e})", by="field-monitor")
            look["rollback"] = True
            break
    print("\nfield monitor (deployed version vs recorded parent baseline, same field tasks)")
    table([[x["batch"], x["episodes"], f"{x['active_success']:.3f}", f"{x['parent_success']:.3f}",
            f"{x['active_wins']}:{x['parent_wins']}", f"{x['p']:.1e}", "ROLLBACK" if x.get("rollback") else ""] for x in looks],
          ["batch", "episodes", "deployed", "parent", "wins:losses", "p (parent better)", ""])

    after = read_active(camp)
    reg = CandidateRegistry(camp / "registry")
    ledger = [{k: v for k, v in e.items() if k in ("op", "id", "restored", "previous", "approved_by", "by", "reason")}
              for e in reg.ledger()]
    print(f"\nactive.json after the monitor: version {after.version}, patch {after.patch}")
    for e in ledger:
        print(f"  ledger {e}")

    # ---- 3. the restored version on new field episodes, and the candidate in simulation for reference
    check_seeds = range(210_000, 210_000 + (10 if a.quick else 100))
    restored = pool_outcomes(field_base.with_overrides({"agent": after.agent}), check_seeds, workers=a.workers,
                             tmp_root=a.out / "tmp")
    cand_sim = pool_outcomes(exp.with_overrides({"agent": deployed.agent}), check_seeds, workers=a.workers, tmp_root=a.out / "tmp")
    parent_sim = pool_outcomes(exp.with_overrides({"agent": parent_agent}), check_seeds, workers=a.workers, tmp_root=a.out / "tmp")
    cand_field = pool_outcomes(field_base.with_overrides({"agent": deployed.agent}), check_seeds, workers=a.workers,
                               tmp_root=a.out / "tmp")
    shutil.rmtree(a.out / "tmp", ignore_errors=True)
    rate = lambda rows: sum(r["success"] for r in rows if r["success"] is not None) / max(1, sum(r["success"] is not None for r in rows))  # noqa: E731
    summary = {"simulation (20 Hz)": {"parent": rate(parent_sim), "candidate": rate(cand_sim)},
               "field (10 Hz)": {"parent": rate(restored), "candidate": rate(cand_field)}}
    print(f"\nsuccess on {2 * len(check_seeds)} new episodes per cell (parent = the restored version)")
    table([[k, f"{v['parent']:.3f}", f"{v['candidate']:.3f}"] for k, v in summary.items()], ["cell", "parent", "candidate"])

    write_result(a.out, {"example": SLUG, "quick": a.quick, "candidate": cid, "gate": gate, "approved_by": "operator",
                         "field": {"condition": FIELD, "alpha_per_look": ALPHA_FIELD, "baseline_success": sum(b_ok) / len(b_ok),
                                   "looks": looks, "rolled_back_to": rolled, "rolled_back": rolled is not None or bool(looks and looks[-1].get("rollback"))},
                         "active_after": after.model_dump(mode="json"), "ledger": ledger, "check": summary,
                         "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
