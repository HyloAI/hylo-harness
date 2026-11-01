"""Language-conditioned object selection with spatial memory.

ObjectReasoner turns an instruction + perceived (or remembered) objects into a subgoal. Which object
the instruction refers to is decided by a pluggable `Chooser`: a rule-based colour matcher here, an
LLM/VLM in the wc-reasoner-llm adapter. Objects are written to spatial memory while visible and
recalled from it once they are occluded, so the same code path serves both cases.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from worldcurrent.core.memory import MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan
from worldcurrent.core.types import Observation

from worldcurrent.core.reasoning import Choice, Chooser, Decision, ReasonerContext, SceneObject  # noqa: F401

FRAME = "table"


COLOR_NAMES: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.2), "blue": (0.15, 0.35, 0.9),
    "yellow": (0.95, 0.85, 0.1), "purple": (0.55, 0.2, 0.75), "orange": (0.95, 0.5, 0.1),
}


class RuleColorChooser:
    """Matches an explicit colour word in the instruction to the nearest object colour.
    Falls back to the nearest object when no colour word is present."""

    name = "chooser.rule_color@0.1"

    def choose(self, instruction: str, objects: list[SceneObject], robot_xy: tuple[float, float],
               obs: Observation | None = None) -> Choice:
        words = set(re.findall(r"[a-z]+", instruction.lower()))
        named = [c for c in COLOR_NAMES if c in words]
        if named and objects:
            ref = np.array(COLOR_NAMES[named[0]])
            best = min(objects, key=lambda o: float(np.linalg.norm(np.array(o.rgb) - ref)))
            return Choice(object_id=best.id, rationale=f"instruction names '{named[0]}'; closest colour is {best.id}")
        if not objects:
            return Choice(object_id=None, rationale="no objects perceived or remembered")
        best = min(objects, key=lambda o: float(np.hypot(o.position[0] - robot_xy[0], o.position[1] - robot_xy[1])))
        return Choice(object_id=best.id, rationale="no colour word in instruction; guessing the nearest object")


class ObjectReasoner:
    required_state = ("tip_pos", "obj_pos", "obj_rgb", "objects_visible")
    visibility_key = "objects_visible"

    def __init__(self, chooser: Chooser) -> None:
        self.chooser = chooser
        self.name = f"object_reasoner[{chooser.name}]@0.1"

    # -- perception & memory -------------------------------------------------------------------
    def _perceive(self, obs: Observation) -> list[SceneObject]:
        s = obs.state
        return [SceneObject(id=f"obj_{i}", rgb=tuple(map(float, s["obj_rgb"][i])),
                            position=tuple(map(float, s["obj_pos"][i])), source="seen")
                for i in range(len(s["obj_pos"]))]

    def _remember(self, objects: list[SceneObject], obs: Observation, rctx: ReasonerContext) -> None:
        if rctx.memory is None:
            return
        items = [MemoryItem(kind=MemoryKind.SPATIAL, key=f"{rctx.episode_id}:{o.id}",
                            content=f"{o.id} rgb={o.rgb} at {o.position}", entities=(o.id,),
                            scope={"episode_id": rctx.episode_id, "task_id": rctx.task_id},
                            pose=Pose(frame_id=FRAME, position=(*o.position, 0.0)),
                            event_time=TimeSpan(start=obs.time), payload={"rgb": list(o.rgb)}, source=self.name)
                 for o in objects]
        res = rctx.memory.write(items)
        if any(op == "ADD" for op, _ in res.events):
            rctx.emit("memory.write", {"op": "write", "n": len(items), "events": [list(e) for e in res.events]})

    def _recall(self, rctx: ReasonerContext) -> list[SceneObject]:
        if rctx.memory is None:
            return []
        hits = rctx.memory.query(MemoryQuery(kinds=frozenset({MemoryKind.SPATIAL}),
                                             scope={"episode_id": rctx.episode_id}, k=32))
        rctx.emit("memory.retrieval", {"query": {"kind": "spatial", "scope": "episode"}, "n_hits": len(hits),
                                       "hits": [h.item.entities[0] for h in hits]})
        return [SceneObject(id=h.item.entities[0], rgb=tuple(h.item.payload["rgb"]),
                            position=tuple(h.item.pose.position[:2]), source="remembered")
                for h in hits if h.item.pose is not None]

    # -- Reasoner protocol -----------------------------------------------------------------------
    def begin(self, rctx: ReasonerContext) -> None:
        rctx.working.clear()

    def deliberate(self, obs: Observation, trigger: str, rctx: ReasonerContext) -> Decision:
        s, wm = obs.state, rctx.working
        tip = tuple(map(float, s["tip_pos"]))
        if s["objects_visible"][0] > 0.5:
            objects = self._perceive(obs)
            self._remember(objects, obs, rctx)
        else:
            objects = self._recall(rctx)
        by_id = {o.id: o for o in objects}
        target = wm.get("target")
        new_choice = False
        if target is None or target not in by_id:
            t0 = time.perf_counter()
            choice = self.chooser.choose(obs.instruction or "", objects, tip, obs=obs)
            latency = time.perf_counter() - t0
            wm.put("target", choice.object_id)
            wm.put("choice", choice)
            target, new_choice = choice.object_id, True
            meta = {**choice.meta, "latency_s": choice.meta.get("latency_s", latency)}
        choice = wm.get("choice")
        if target is None or target not in by_id:
            return Decision(kind="subgoal", output=list(tip), rationale=choice.rationale if choice else "nothing to do",
                            trigger=trigger, component=self.name, inputs={"new_choice": new_choice})
        obj = by_id[target]
        inputs: dict[str, Any] = {"new_choice": new_choice, "target": target, "object_source": obj.source,
                                  "instruction": obs.instruction, "chooser": self.chooser.name,
                                  "n_objects": len(objects)}
        if new_choice:
            inputs["chooser_meta"] = meta
        return Decision(kind="subgoal", output=list(obj.position), rationale=choice.rationale,
                        trigger=trigger, component=self.name, inputs=inputs)

    def end(self, success: bool | None, rctx: ReasonerContext) -> None:
        choice = rctx.working.get("choice")
        if rctx.memory is None or choice is None:
            return
        rctx.memory.write([MemoryItem(kind=MemoryKind.EPISODIC, content=f"chose {choice.object_id}: {choice.rationale}",
                                      scope={"episode_id": rctx.episode_id, "task_id": rctx.task_id},
                                      payload={"success": success, "chooser": self.chooser.name}, source=self.name)])
