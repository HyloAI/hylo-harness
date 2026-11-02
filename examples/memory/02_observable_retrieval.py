"""Observable memory: every write and retrieval of one episode, read back from the trace and the SQLite file.

What it shows
    One MuJoCo table-reach episode ("touch the <colour> pad", pads visible for 5 steps) run by
    `composable.object_reach` with a file-backed `sqlite` spatial memory. Hylo records every memory access as a
    trace event (`memory.write`, `memory.retrieval`) next to the decisions they fed (`agent.decision`, with
    `object_source` = seen | remembered), so an outcome can be attributed to the memory that influenced it.
    The example prints that timeline, then opens the SQLite file the episode left behind and queries it directly
    (by key, and by radius around the true target pad, with scores), and checks the recalled pad positions against
    the simulator's privileged state. The same episode without memory is run for contrast.

Command (from the harness repository root)
    uv run python examples/memory/02_observable_retrieval.py [--quick] [--out DIR] [--seed-index I]
    uv run hylo trace runs/examples/memory/observable_retrieval/with-memory      # the generic trace summary

Runtime: under 1 s (the same with --quick: a single episode).
Requires: hylo-env-mujoco (hylo-bench), hylo-reasoners.
"""

from __future__ import annotations

import numpy as np

from _common import Timer, parse_args, print_table, require_plugin, write_result

SLUG = "observable_retrieval"


def main() -> None:
    args = parse_args(SLUG, __doc__, lambda ap: ap.add_argument("--seed-index", type=int, default=0,
                                                              help="which validation seed of the direct task"))
    require_plugin("envs", "mujoco.table_reach")
    from hylo.core.kernel import EpisodeRunner
    from hylo.core.memory import MemoryKind, MemoryQuery, Pose
    from hylo.core.registry import registry
    from hylo.core.task import TaskInstance
    from hylo.core.trace import TraceStore
    from hylo.memory.sqlite import SQLiteMemoryStore

    clock = Timer()
    bench = registry.build("benchmarks", {"type": "mujoco/table-v0"})
    task = next(t for t in bench.tasks if t.id == "mujoco/table_direct")
    inst = TaskInstance(spec=task, seed=bench.splits["validation"][args.seed_index], split="validation")
    db = args.out / "memory.sqlite"
    db.unlink(missing_ok=True)

    def episode(memory: dict | None, name: str):
        env = registry.build("envs", {"type": "mujoco.table_reach"})
        agent = registry.build("agents", {"type": "composable.object_reach", "params": {
            "kp": 0.3, "chooser": {"type": "chooser.rule_color"}, "memory": memory}})
        store = TraceStore(args.out / name)
        rec = EpisodeRunner(env, agent, trace=store, safety={"max_delta": 0.5}).run(inst)
        store.close()
        env.close()
        return rec.outcome, store.read(rec.outcome.episode_id)

    out_mem, trace = episode({"type": "sqlite", "params": {"path": str(db)}}, "with-memory")
    out_none, trace_none = episode(None, "without-memory")

    # 1. the timeline of memory events and the decisions they fed
    step, timeline, counts = 0, [], {"memory.write": 0, "memory.retrieval": 0, "agent.decision": 0}
    write_ops: dict[str, int] = {}
    first_remembered = None
    for e in trace.events:
        t, p = e["type"], e["payload"]
        if t == "step.result":
            step = p["step"]
        if t not in counts:
            continue
        counts[t] += 1
        if t == "memory.write":
            for op, _ in p.get("events", []):
                write_ops[op] = write_ops.get(op, 0) + 1
            what = f"{p.get('op')} n={p.get('n')} " + ",".join(sorted({op for op, _ in p.get("events", [])}))
        elif t == "memory.retrieval":
            what = f"{p['query']} -> {p['n_hits']} hits {p['hits']}"
        else:
            src = (p.get("inputs") or {}).get("object_source")
            if src == "remembered" and first_remembered is None:
                first_remembered = step
            what = (f"trigger={p.get('trigger')} target={(p.get('inputs') or {}).get('target')} source={src} "
                    f"subgoal=({p['output'][0]:+.3f}, {p['output'][1]:+.3f})")
        timeline.append({"step": step, "event": t, "detail": what})

    step, no_mem = 0, []
    for e in trace_none.events:
        if e["type"] == "step.result":
            step = e["payload"]["step"]
        elif e["type"] == "agent.decision":
            p = e["payload"]
            row = {"step": step, "trigger": p.get("trigger"), "rationale": p.get("rationale"),
                   "subgoal (first)": f"({p['output'][0]:+.3f}, {p['output'][1]:+.3f})"}
            if no_mem and all(no_mem[-1][k] == row[k] for k in ("trigger", "rationale")):
                no_mem[-1]["step"] = f"{str(no_mem[-1]['step']).split('-')[0]}-{step}"      # fold repeats
            else:
                no_mem.append(row)

    # 2. what the episode left in the SQLite file (a new store opened on the same file, as a later session would)
    mem = SQLiteMemoryStore(db)
    priv = trace.first("privileged.state")["payload"]
    pads, colors, target = np.asarray(priv["pad_pos"]), priv["colors"], int(priv["target"])
    spatial = mem.query(MemoryQuery(kinds=frozenset({MemoryKind.SPATIAL}), k=32))
    items = []
    for h in sorted(spatial, key=lambda h: h.item.entities[0]):
        i = int(h.item.entities[0].split("_")[1])
        err = float(np.linalg.norm(np.asarray(h.item.pose.position[:2]) - pads[i]))
        items.append({"key": h.item.key.split(":")[-1], "colour (sim)": colors[i], "target": i == target,
                      "recalled xy": f"({h.item.pose.position[0]:+.3f}, {h.item.pose.position[1]:+.3f})",
                      "error m": round(err, 6)})
    centre = Pose(frame_id="table", position=(float(pads[target][0]), float(pads[target][1]), 0.0))
    near = mem.query(MemoryQuery(near=(centre, 0.25), k=5))
    near_rows = [{"item": h.item.entities[0] if h.item.entities else h.item.kind.value, "score": round(h.score, 3),
                  **{k: round(v, 3) for k, v in h.details.items()}} for h in near]
    episodic = mem.query(MemoryQuery(kinds=frozenset({MemoryKind.EPISODIC}), k=5))

    result = {
        "example": SLUG, "task": task.id, "seed": inst.seed, "instruction": trace.first("observation")["payload"].get("instruction"),
        "with_memory": {"status": out_mem.status, "steps": out_mem.steps},
        "without_memory": {"status": out_none.status, "steps": out_none.steps},
        "event_counts": counts, "write_ops": write_ops, "first_remembered_decision_step": first_remembered,
        "recalled_pads": items, "max_recall_error_m": max(r["error m"] for r in items) if items else None,
        "radius_query": near_rows, "episodic": [h.item.content for h in episodic], "memory_snapshot": mem.snapshot(),
        "timeline": timeline, "decisions_without_memory": no_mem, "runtime_s": clock()}
    write_result(args.out, result)
    print(f"{task.id} seed {inst.seed}: \"{result['instruction']}\"")
    print_table(_compress(timeline), title="trace: memory events and decisions (consecutive repeats folded)")
    print_table(no_mem, title="the same episode without memory: decisions")
    print_table(items, title=f"spatial items in {db.name} (checked against privileged state)")
    print_table(near_rows, title="radius query: 0.25 m around the true target pad")
    print_table([{"run": "with memory", "status": out_mem.status, "steps": out_mem.steps,
                  "writes": counts["memory.write"], "retrievals": counts["memory.retrieval"]},
                 {"run": "without memory", "status": out_none.status, "steps": out_none.steps,
                  "writes": 0, "retrievals": 0}], title="outcome")
    print(f"\nwrite ops {write_ops}; first decision from memory at step {first_remembered}; "
          f"episodic note: {result['episodic'][:1]}")
    print(f"result: {args.out / 'result.json'}  ({clock()} s)")


def _compress(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if out and out[-1]["event"] == r["event"] and out[-1]["detail"] == r["detail"]:
            out[-1]["step"] = f"{str(out[-1]['step']).split('-')[0]}-{r['step']}"
            continue
        out.append(dict(r))
    return out


if __name__ == "__main__":
    main()
