"""Memory that survives across sessions: a fixed workcell repeats its scenes, and each session starts from the
verified experience the previous ones saved to disk (MuJoCo planar reach).

What it shows
    The online deployment mode of RFC-0003 action memory, one process per session:
    - the workcell repeats the same scenes (the search_val init states) every session, with fresh (but seeded)
      sampling noise each session;
    - session s loads the bank saved by session s - 1 (frozen during the session, its digest in the agent spec),
      runs every scene, then appends the verified successes of this session and saves bank v{s} with a new digest;
    - the control arm runs the same scenes with the same noise and no memory;
    - each session also appends a summary to a SQLite journal (`sqlite` memory plugin, EPISODIC items), which the
      next process reads back before it starts.
    The agent is a small flow-matching policy with a wide sampler (1 sample, noise scale 2.0); memory steering
    settings as in example 03 (k 5, similarity floor 0.99, fixed t0 0.3). Session 1 starts with an empty bank,
    so it equals the control by construction.

Command (from the harness repository root)
    uv run python examples/memory/05_memory_across_sessions.py [--quick] [--out DIR]

Runtime: about 6 s with --quick (2 sessions), about 1 min in full mode (6 sessions; 16 worker threads, CPU).
Requires: hylo-env-mujoco (hylo-bench), hylo-policies, hylo-learn.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _common import (PriorFlowSampler, Timer, evaluate_agent, paired, parse_args, print_table, rate, reach_config,
                     require_plugin, segments_from_run, train_flow_policy, write_result)

SLUG = "memory_across_sessions"
STEERING = {"n_samples": 1, "noise_scale": 2.0}
MEMORY_STEER = {"k": 5, "sim_floor": 0.99, "fixed_t0": 0.3}
SPLIT = "search_val"                     # the workcell's scenes (a development split: memory may be built on it)


def settings(quick: bool) -> dict:
    return {"rsi_states": 50, "epochs": 150, "sessions": 2} if quick else {"rsi_states": 250, "epochs": 300, "sessions": 6}


def run_session(args, s: int) -> dict:
    """One session in its own process: load bank v{s-1} from disk, run both arms, save bank v{s}."""
    from hylo.agents.memory_steer import MemorySteeredAgent
    from hylo.agents.steering import SteeredAgent
    from hylo.core.memory import MemoryItem, MemoryKind, MemoryQuery
    from hylo.memory.action_bank import ActionBank
    from hylo.memory.sqlite import SQLiteMemoryStore

    st = settings(args.quick)
    art = train_flow_policy(args.out, rsi_states=st["rsi_states"], epochs=st["epochs"], workers=args.workers)
    journal = SQLiteMemoryStore(args.out / "journal.sqlite")
    previous = sorted((h.item.payload for h in journal.query(MemoryQuery(kinds=frozenset({MemoryKind.EPISODIC}), k=100))),
                      key=lambda p: p["session"])
    prev_bank = args.out / f"bank-v{s - 1}"
    flow = {"type": "policy.flow", "params": {"path": str(art)}}
    cfg = reach_config({"type": "steered", "params": {"sampler": flow, "steering": STEERING}},
                       rsi_states=st["rsi_states"], workers=args.workers)
    root = args.out / f"session-{s}"

    control = evaluate_agent(cfg, SPLIT, lambda: SteeredAgent(PriorFlowSampler(art, salt=s), STEERING), root, "control")
    if prev_bank.exists():
        memory = {"type": "action_bank", "params": {"path": str(prev_bank)}}          # frozen for the session
        mem = evaluate_agent(cfg, SPLIT, lambda: MemorySteeredAgent(PriorFlowSampler(art, salt=s), steering=STEERING,
                                                                    memory=memory, memory_steer=MEMORY_STEER), root, "memory")
        bank = ActionBank(str(prev_bank), frozen=False)
        loaded = {"path": prev_bank.name, "digest": bank.snapshot(), "segments": len(bank)}
    else:                                # an empty bank never produces a prior: the memory agent is the control
        mem, bank, loaded = control, ActionBank(frozen=False), None
    new = segments_from_run(mem.run_dir, PriorFlowSampler(art), splits=(SPLIT,))
    bank.add(new)
    digest = bank.save(args.out / f"bank-v{s}")
    out = {"session": s, "loaded_bank": loaded, "control": rate(control.per_key().values()),
           "memory": rate(mem.per_key().values()), "paired": paired(control.per_key(), mem.per_key()),
           "added_segments": len(new), "saved_bank": {"path": f"bank-v{s}", "digest": digest, "segments": len(bank)},
           "previous_sessions_in_journal": len(previous)}
    journal.write([MemoryItem(kind=MemoryKind.EPISODIC, key=f"session:{s}", scope={"workcell": "planar-reach"},
                              content=f"session {s}: memory {out['memory']['success']}, control {out['control']['success']}, "
                                      f"bank {digest} ({len(bank)} segments)",
                              payload={"session": s, "memory": out["memory"]["success"], "control": out["control"]["success"],
                                       "bank": digest}, source="examples/memory/05")])
    (args.out / f"session-{s}.json").write_text(json.dumps(out, indent=2))
    last = previous[-1] if previous else None
    print(f"session {s}: journal has {len(previous)} earlier session(s)"
          + (f", last: memory {last['memory']} / control {last['control']} with bank {last['bank']}" if last else "")
          + f"; loaded {loaded['path'] + ' ' + loaded['digest'] if loaded else 'no bank'}; "
            f"memory {out['memory']['success']} vs control {out['control']['success']}; saved bank v{s} {digest}")
    return out


def main() -> None:
    args = parse_args(SLUG, __doc__, lambda ap: ap.add_argument("--session", type=int, help=argparse_hidden()))
    require_plugin("envs", "mujoco.planar_reach")
    if args.session is not None:
        run_session(args, args.session)
        return
    clock = Timer()
    st = settings(args.quick)
    for old in [*args.out.glob("bank-v*"), *args.out.glob("session-*"), args.out / "journal.sqlite"]:
        _remove(old)
    train_flow_policy(args.out, rsi_states=st["rsi_states"], epochs=st["epochs"], workers=args.workers)
    for s in range(1, st["sessions"] + 1):           # a new process per session: only the files carry memory over
        cmd = [sys.executable, __file__, "--session", str(s), "--out", str(args.out), "--workers", str(args.workers)]
        subprocess.run(cmd + (["--quick"] if args.quick else []), check=True)
    sessions = [json.loads((args.out / f"session-{s}.json").read_text()) for s in range(1, st["sessions"] + 1)]
    rows = [{"session": r["session"], "bank segments": r["loaded_bank"]["segments"] if r["loaded_bank"] else 0,
             "n": r["paired"]["n_pairs"], "control": r["control"]["success"], "memory": r["memory"]["success"],
             "diff": r["paired"]["diff"], "+/-": f"+{r['paired']['wins']} / -{r['paired']['losses']}",
             "p": r["paired"]["p_two_sided"]} for r in sessions]
    later = [r for r in sessions if r["loaded_bank"]]
    pooled = {"wins": sum(r["paired"]["wins"] for r in later), "losses": sum(r["paired"]["losses"] for r in later),
              "n_pairs": sum(r["paired"]["n_pairs"] for r in later)}
    if later:
        from _common import mcnemar_two_sided
        pooled["diff"] = round((pooled["wins"] - pooled["losses"]) / pooled["n_pairs"], 4)
        pooled["p_two_sided_naive"] = float(f"{mcnemar_two_sided(pooled['wins'], pooled['losses']):.4g}")
    write_result(args.out, {"example": SLUG, "quick": args.quick, **st, "steering": STEERING, "memory_steer": MEMORY_STEER,
                            "sessions": sessions, "pooled_sessions_2_plus": pooled, "runtime_s": clock()})
    print_table(rows, title="the same workcell scenes every session (paired with the control, exact two-sided McNemar)")
    if later:
        print(f"sessions 2+ pooled: diff {pooled['diff']:+.3f}, +{pooled['wins']} / -{pooled['losses']} "
              f"(p {pooled['p_two_sided_naive']}, naive: the same scenes repeat across sessions)")
    print(f"\nresult: {args.out / 'result.json'}  ({clock()} s)")


def argparse_hidden() -> str:
    import argparse

    return argparse.SUPPRESS


def _remove(p: Path) -> None:
    import shutil

    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
