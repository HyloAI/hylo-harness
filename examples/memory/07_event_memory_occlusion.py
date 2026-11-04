"""New EventMemoryAgent on MuJoCo table reach, using the same fixed controller with/without memory.

Structured visible object measurements are allowed in this local task. This experiment checks
runtime ingestion, episode reset, context compilation and control, NOT visual VLA performance.
The controller has no retained target and no learned weights. An identical observation with
hidden objects gives no action unless the external memory supplies observed scene evidence.

    python examples/memory/07_event_memory_occlusion.py --quick --out /tmp/hylo-event-occlusion
"""
from __future__ import annotations

import json
import time

import numpy as np

from _common import paired, parse_args, rate, require_plugin, write_result
from hylo.agents.composable import PlanarIKGoalPolicy
from hylo.agents.event_memory import EventMemoryAgent
from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.episode_memory import RecallQuery
from hylo.core.reasoning import SceneObject
from hylo.core.types import ActionChunk
from hylo.memory.episode import EventMemory
from hylo.memory.signals import Signal, SignalExtractor
from hylo.reasoners.rules.objects import RuleColorChooser

SLUG = "event_memory_occlusion"
MARKER = "\nObserved episode memory"


class SceneController(AgentBase):
    """Fixed, memoryless decoder/controller shared by BOTH experimental conditions."""
    def __init__(self):
        self.controller = PlanarIKGoalPolicy(kp=0.3)
        self.chooser = RuleColorChooser()

    def spec(self):
        return AgentSpec(name="reference-scene-controller", version="1",
                         required_state=("qpos", "tip_pos", "obj_pos", "obj_rgb", "objects_visible"),
                         action_semantics=self.controller.semantics, action_dim=2)

    def act(self, obs):
        instruction = (obs.instruction or "").split(MARKER)[0]
        if bool(obs.state["objects_visible"].item()):
            positions, colors = obs.state["obj_pos"], obs.state["obj_rgb"]
        else:
            text = obs.instruction or ""
            if MARKER not in text:
                return ActionChunk.single(np.zeros(2))
            payload = json.loads(text.split(MARKER, 1)[1].split("\n", 1)[1])
            known = {r["name"]: r["value"] for r in payload if r["status"] == "known"}
            if "positions" not in known or "colors" not in known:
                return ActionChunk.single(np.zeros(2))
            positions, colors = known["positions"], known["colors"]
        objects = [SceneObject(id=str(i), rgb=tuple(map(float, rgb)), position=tuple(map(float, p)), source="seen")
                   for i, (p, rgb) in enumerate(zip(positions, colors))]
        choice = self.chooser.choose(instruction, objects, tuple(map(float, obs.state["tip_pos"])))
        if choice.object_id is None:
            return ActionChunk.single(np.zeros(2))
        return self.controller.act(obs, np.asarray(positions[int(choice.object_id)]))


def agent(enabled):
    inner = SceneController()
    if not enabled:
        return inner
    signals = [Signal(kind="positions", subject="scene", value_key="obj_pos", visible_key="objects_visible"),
               Signal(kind="colors", subject="scene", value_key="obj_rgb", visible_key="objects_visible")]
    queries = [RecallQuery(name=k, op="latest", kind=k, subject="scene") for k in ("positions", "colors")]
    # evaluate() reuses each worker's agent across episodes; reset partitions its journal.
    return EventMemoryAgent(inner, EventMemory(), SignalExtractor(signals), queries)


def main():
    args = parse_args(SLUG, __doc__)
    require_plugin("envs", "mujoco.table_reach")
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry
    from hylo.eval.experiment import env_factory
    from hylo.eval.runner import evaluate

    n_seeds = 10 if args.quick else 100
    cfg = ExperimentConfig(name="event-memory-table", benchmark={"type": "mujoco/table-v0", "params": {"validation": n_seeds}},
                           env={"type": "mujoco.table_reach"}, agent={"type": "baseline.planar_ik"},
                           safety={"max_delta": 0.5}, workers=args.workers)
    original = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    results = {}
    started = time.perf_counter()
    for visible in (5, None):
        tasks = tuple(t.model_copy(update={"params": {**t.params, "visible_steps": visible}}) for t in original.tasks)
        benchmark = original.model_copy(update={"tasks": tasks})
        runs = []
        for enabled in (False, True):
            runs.append(evaluate(benchmark, "validation", env_factory=env_factory(cfg),
                                 agent_factory=lambda: agent(enabled), workers=args.workers,
                                 run_root=args.out / "runs", run_id=f"{'on' if enabled else 'off'}-{visible or 'all'}",
                                 safety=cfg.safety, record_privileged=False))
        off, on = runs
        a, b = off.per_key(), on.per_key()
        per_task = {}
        for task in sorted({s["task_id"] for s in off.log["samples"]}):
            ids = {s["key"] for s in off.log["samples"] if s["task_id"] == task}
            ta, tb = {k: a[k] for k in ids}, {k: b[k] for k in ids}
            per_task[task] = {"off": rate(ta.values()), "on": rate(tb.values()), "paired": paired(ta, tb)}
        results[f"visible_{visible or 'all'}"] = {
            "off": rate(a.values()), "on": rate(b.values()), "paired": paired(a, b), "per_task": per_task,
            "runs": {"off": str(off.run_dir), "on": str(on.run_dir)}}
        summary = results[f"visible_{visible or 'all'}"]["paired"]
        print(f"visible={visible or 'all'}: {summary}")
    write_result(args.out, {"example": SLUG, "quick": args.quick, "seeds_per_task": n_seeds,
                            "policy": "fixed reference controller, no VLA", "observations": "visible structured signals",
                            "official_benchmark": False, "results": results, "runtime_s": time.perf_counter() - started})
    print(f"Results: {args.out / 'result.json'}")


if __name__ == "__main__":
    main()
