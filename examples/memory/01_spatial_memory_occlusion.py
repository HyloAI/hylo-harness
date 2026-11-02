"""Spatial memory under occlusion: the agent must remember where the named pad was (MuJoCo table reach).

What it shows
    `composable.object_reach` (rule-based colour chooser + IK policy) on `mujoco.table_reach`: three coloured pads
    are visible only for the first 5 control steps, then the observation hides them. With spatial memory (the
    `sqlite` memory plugin) the reasoner writes each pad's pose while it is visible and recalls it once it is
    occluded; with memory off it has nothing to steer to. Same episodes, same seeds: a paired comparison per task,
    95% Wilson intervals and exact two-sided McNemar tests. A second table varies how long the pads stay visible
    (1, 5, 20 steps, always), which isolates what memory buys: nothing when nothing is hidden.

    Memory does not fix language: on the indirect task ("the colour of a ripe tomato") the rule-based chooser
    finds no colour word and guesses the nearest pad, with or without memory.

Command (from the harness repository root)
    uv run python examples/memory/01_spatial_memory_occlusion.py [--quick] [--out DIR]

Runtime: about 3 s with --quick, about 1.5 min in full mode (16 worker threads, CPU).
Requires: hylo-env-mujoco (hylo-bench), hylo-reasoners.
"""

from __future__ import annotations

from _common import Timer, paired, parse_args, print_table, rate, require_plugin, write_result

SLUG = "spatial_memory_occlusion"


def agent(memory: bool) -> dict:
    mem = {"type": "sqlite", "params": {"path": ":memory:"}} if memory else None
    return {"type": "composable.object_reach", "params": {"kp": 0.3, "chooser": {"type": "chooser.rule_color"}, "memory": mem}}


def main() -> None:
    args = parse_args(SLUG, __doc__)
    require_plugin("envs", "mujoco.table_reach")
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry
    from hylo.eval.experiment import build_agent, env_factory
    from hylo.eval.runner import evaluate

    clock = Timer()
    n_seeds = 10 if args.quick else 100
    visibility = (5, None) if args.quick else (1, 5, 20, None)
    base = ExperimentConfig(name="table-memory", benchmark={"type": "mujoco/table-v0", "params": {"validation": n_seeds}},
                            env={"type": "mujoco.table_reach"}, agent=agent(True), safety={"max_delta": 0.5},
                            workers=args.workers)
    bench0 = registry.build("benchmarks", base.benchmark.model_dump(exclude={"split"}))

    def run(memory: bool, visible: int | None):
        tasks = tuple(t.model_copy(update={"params": {**t.params, "visible_steps": visible}}) for t in bench0.tasks)
        bench = bench0.model_copy(update={"tasks": tasks})
        cfg = base.with_overrides({"agent": agent(memory)})
        label = f"{'memory' if memory else 'no-memory'}-visible-{visible or 'all'}"
        return evaluate(bench, "validation", env_factory=env_factory(cfg), agent_factory=lambda: build_agent(cfg),
                        run_root=args.out / "runs", run_id=label, workers=args.workers, safety=cfg.safety,
                        record_privileged=False)

    results, sweep = {}, []
    for visible in visibility:
        off, on = run(False, visible), run(True, visible)
        a, b = off.per_key(), on.per_key()
        per_task = {}
        for task in sorted({s["task_id"] for s in off.log["samples"]}):
            ids = {s["key"] for s in off.log["samples"] if s["task_id"] == task}
            ta, tb = {k: a[k] for k in ids}, {k: b[k] for k in ids}
            per_task[task] = {"memory_off": rate(ta.values()), "memory_on": rate(tb.values()), "paired": paired(ta, tb)}
        results[f"visible_{visible or 'all'}"] = {
            "visible_steps": visible, "memory_off": rate(a.values()), "memory_on": rate(b.values()),
            "paired": paired(a, b), "per_task": per_task,
            "steps_mean": {"memory_off": off.metrics["steps_mean"], "memory_on": on.metrics["steps_mean"]},
            "runs": {"memory_off": str(off.run_dir), "memory_on": str(on.run_dir)}}
        p = results[f"visible_{visible or 'all'}"]["paired"]
        sweep.append({"pads visible": f"{visible} steps" if visible else "always", "n": p["n_pairs"],
                      "memory off": results[f"visible_{visible or 'all'}"]["memory_off"]["success"],
                      "memory on": results[f"visible_{visible or 'all'}"]["memory_on"]["success"],
                      "diff": p["diff"], "+/-": f"+{p['wins']} / -{p['losses']}", "p": p["p_two_sided"]})

    main_res = results["visible_5"]
    rows = []
    for task, r in main_res["per_task"].items():
        rows.append({"task": task.split("/")[-1], "n": r["paired"]["n_pairs"],
                     "memory off [95% CI]": _ci(r["memory_off"]), "memory on [95% CI]": _ci(r["memory_on"]),
                     "+/-": f"+{r['paired']['wins']} / -{r['paired']['losses']}", "p": r["paired"]["p_two_sided"]})
    rows.append({"task": "all", "n": main_res["paired"]["n_pairs"], "memory off [95% CI]": _ci(main_res["memory_off"]),
                 "memory on [95% CI]": _ci(main_res["memory_on"]),
                 "+/-": f"+{main_res['paired']['wins']} / -{main_res['paired']['losses']}", "p": main_res["paired"]["p_two_sided"]})
    write_result(args.out, {"example": SLUG, "quick": args.quick, "seeds_per_task": n_seeds, "results": results,
                            "runtime_s": clock()})
    print_table(rows, title=f"pads visible for 5 steps, {n_seeds} validation seeds per task (exact two-sided McNemar)")
    print_table(sweep, title="how long the pads stay visible (all three tasks)")
    print(f"\nresult: {args.out / 'result.json'}  ({clock()} s)")


def _ci(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{r['success']:.2f} [{lo:.2f}, {hi:.2f}]"


if __name__ == "__main__":
    main()
