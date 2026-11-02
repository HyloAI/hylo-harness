"""Action memory is scene recall: it helps on revisited scenes, not on new ones (MuJoCo planar reach).

What it shows
    The experience (action) memory of RFC-0003 on a small flow-matching policy: verified successful chunks are
    stored in an ActionBank (keyed by the policy's own observation vector), and MemorySteeredAgent starts each flow
    sample from a retrieved prior (Retrieve-then-Steer). The bank holds the successes of three earlier visits to
    each search_val scene (other noise draws and noise scales than the evaluated agent, as the recorded LIBERO-10
    study used other configurations). The agent (a noisy sampler: 1 sample, noise scale 2.0) is then evaluated,
    paired with common random numbers against the same agent without memory, on
      - revisited scenes: the search_val init states the bank was built on (a fixed workcell repeating scenes);
      - new scenes: held-out init states never seen by the policy or the bank.
    LIBERO-10 (recorded, SmolVLA): +8 points on revisited scenes, -8 on new ones.

Steps
    1. expert demonstrations (IK controller, RFC-0002 search split) -> a policy.flow artifact (hylo learn);
    2. three earlier visits to every search_val scene (noise 1.0 / 1.5 / 2.0, each with its own noise seeds) ->
       verified segments of the successful ones -> ActionBank, saved with its content digest, loaded frozen;
    3. control (policy.flow, 1 sample, noise 2.0) vs the same agent with action memory on search_val and heldout.

    The memory settings (k 5, similarity floor 0.99, fixed t0 0.3) were picked from a sweep of four settings on
    these same splits, so read the p-values as exploratory.

Command (from the harness repository root)
    uv run python examples/memory/03_action_memory_scene_recall.py [--quick] [--out DIR]

Runtime: about 10 s with --quick, about 50 s in full mode (16 worker threads, CPU).
Requires: hylo-env-mujoco (hylo-bench), hylo-policies, hylo-learn.
"""

from __future__ import annotations


from _common import (PriorFlowSampler, Timer, evaluate_agent, paired, parse_args, print_table, rate, reach_config,
                     require_plugin, segments_from_run, train_flow_policy, write_result)

SLUG = "action_memory_scene_recall"
CONTROL = {"n_samples": 1, "noise_scale": 2.0}
BANK_VISITS = {                                    # earlier visits: (noise seed salt, steering); salt 0 = the control's seeds
    "visit-1 noise 1.0": (1, {"n_samples": 1, "noise_scale": 1.0}),
    "visit-2 noise 1.5": (2, {"n_samples": 1, "noise_scale": 1.5}),
    "visit-3 noise 2.0": (3, {"n_samples": 1, "noise_scale": 2.0}),
}
MEMORY_STEER = {"k": 5, "sim_floor": 0.99, "fixed_t0": 0.3}


def main() -> None:
    args = parse_args(SLUG, __doc__)
    require_plugin("envs", "mujoco.planar_reach")
    from hylo.agents.memory_steer import MemorySteeredAgent
    from hylo.agents.steering import SteeredAgent
    from hylo.eval.experiment import run_experiment
    from hylo.memory.action_bank import ActionBank

    clock = Timer()
    rsi_states, epochs = (50, 150) if args.quick else (250, 300)
    art = train_flow_policy(args.out, rsi_states=rsi_states, epochs=epochs, workers=args.workers)
    flow = {"type": "policy.flow", "params": {"path": str(art)}}
    steered = lambda steering: {"type": "steered", "params": {"sampler": flow, "steering": steering}}  # noqa: E731

    cfg = reach_config(steered(CONTROL), rsi_states=rsi_states, workers=args.workers)

    # 2. the bank: verified successes of earlier visits to the scenes that will be revisited
    probe = PriorFlowSampler(art)
    bank, built = ActionBank(capacity=50_000, frozen=False), {}
    for i, (name, (salt, steering)) in enumerate(BANK_VISITS.items()):
        res = evaluate_agent(cfg, "search_val", lambda: SteeredAgent(PriorFlowSampler(art, salt=salt), steering),
                             args.out / "bank-runs", f"visit-{i + 1}")
        segs = segments_from_run(res.run_dir, probe, splits=("search_val",))
        bank.add(segs)
        built[name] = {"success": res.metrics["success_mean"], "segments": len(segs)}
    digest = bank.save(args.out / "bank")
    memory = {"type": "action_bank", "params": {"path": str(args.out / "bank")}}   # loaded frozen

    # 3. control vs memory, paired on the same (task, seed) keys and the same noise seeds
    make_memory = lambda: MemorySteeredAgent(PriorFlowSampler(art), steering=CONTROL, memory=memory,  # noqa: E731
                                             memory_steer=MEMORY_STEER)
    rows, arms = [], {}
    for split, scenes in (("search_val", "revisited"), ("heldout", "new")):
        ctrl = run_experiment(cfg, split=split, run_root=args.out / "eval", label=f"control-{split}")
        mem = evaluate_agent(cfg, split, make_memory, args.out / "eval", "memory")
        used = _prior_usage(mem.run_dir)
        cmp_ = paired(ctrl.per_key(), mem.per_key())
        arms[scenes] = {"split": split, "control": rate(ctrl.per_key().values()), "memory": rate(mem.per_key().values()),
                        "paired": cmp_, "prior_used_fraction": used,
                        "runs": {"control": str(ctrl.run_dir), "memory": str(mem.run_dir)}}
        rows.append({"scenes": scenes, "split": split, "n": cmp_["n_pairs"],
                     "control": arms[scenes]["control"]["success"], "memory": arms[scenes]["memory"]["success"],
                     "diff": cmp_["diff"], "+/-": f"+{cmp_['wins']} / -{cmp_['losses']}", "p": cmp_["p_two_sided"],
                     "prior used": used})

    result = {"example": SLUG, "quick": args.quick, "rsi_states": rsi_states, "control": CONTROL,
              "memory_steer": MEMORY_STEER, "bank_visits": {k: {"salt": s, "steering": c} for k, (s, c) in BANK_VISITS.items()}, "bank": {"digest": digest, "segments": len(bank), "built_from": built},
              "arms": arms, "runtime_s": clock()}
    write_result(args.out, result)
    print_table([{"visit": k, **v} for k, v in built.items()], title=f"bank {digest}: {len(bank)} verified segments from search_val successes")
    print_table(rows, title="control vs action memory (paired, exact two-sided McNemar)")
    print(f"\nresult: {args.out / 'result.json'}  ({clock()} s)")


def _prior_usage(run_dir) -> float:
    """Fraction of decisions where the memory produced a prior (from the memory.retrieval trace events)."""
    from hylo.core.trace import TraceStore

    store, used, total = TraceStore(run_dir), 0, 0
    for eid in store.episode_ids():
        for e in store.read(eid).of_type("memory.retrieval"):
            total += 1
            used += bool(e["payload"].get("used_prior"))
    return round(used / max(1, total), 3)


if __name__ == "__main__":
    main()
