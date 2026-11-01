"""A rule-based reference reasoner (the Reasoner contract is worldcurrent.core.reasoning).

A Reasoner is the slow, event-triggered part of an Agent. It turns observations + memory into
Decisions (here: a subgoal position). An LLM/VLM reasoner, a LangGraph graph or a DSPy program
implements the same protocol; the kernel never sees the difference.
"""

from __future__ import annotations


import numpy as np

from worldcurrent.core.reasoning import Decision, Emit, Reasoner, ReasonerContext  # noqa: F401 - re-exported
from worldcurrent.core.types import Observation
from worldcurrent.core.memory import MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan

# Lawnmower sweep over the test arena (testing.point_reach); spacing < 2 * sense_radius.
SWEEP = [(-0.8, 0.8), (0.8, 0.8), (0.8, 0.4), (-0.8, 0.4), (-0.8, 0.0), (0.8, 0.0),
         (0.8, -0.4), (-0.8, -0.4), (-0.8, -0.8), (0.8, -0.8)]
FRAME = "testing:world"


class LandmarkReasoner:
    """Go to the goal if visible; else to a remembered goal; else explore. Remembers what it saw.

    Uses working memory within an episode and spatial memory across episodes. A remembered
    location that turns out to be empty is invalidated (forgotten), not trusted forever.
    """

    name = "testing.rule_landmark@0.1"
    required_state = ("goal", "goal_visible")
    visibility_key = "goal_visible"

    def __init__(self, reach_tol: float = 0.06) -> None:
        self.reach_tol = reach_tol

    def _key(self, rctx: ReasonerContext) -> str:
        return f"goal:{rctx.task_id}"

    def begin(self, rctx: ReasonerContext) -> None:
        rctx.working.clear()
        rctx.working.put("waypoint", 0)
        if rctx.memory is None:
            return
        q = MemoryQuery(kinds=frozenset({MemoryKind.SPATIAL}), key=self._key(rctx), k=1)
        hits = rctx.memory.query(q)
        rctx.emit("memory.retrieval", {"query": {"kind": "spatial", "key": q.key}, "n_hits": len(hits),
                                       "hits": [{"id": h.item.id, "pos": list(h.item.pose.position) if h.item.pose else None,
                                                 "score": h.score} for h in hits]})
        if hits and hits[0].item.pose:
            rctx.working.put("belief", np.array(hits[0].item.pose.position[:2]))
            rctx.working.put("belief_id", hits[0].item.id)

    def deliberate(self, obs: Observation, trigger: str, rctx: ReasonerContext) -> Decision:
        s, wm = obs.state, rctx.working
        pos = s["pos"]
        if s["goal_visible"][0] > 0.5:
            goal = np.asarray(s["goal"], dtype=float)
            wm.put("belief", goal)
            wm.put("observed", (goal, obs.time))
            wm.put("belief_id", None)
            return Decision(kind="subgoal", output=goal, rationale="goal visible", trigger=trigger, component=self.name)
        belief = wm.get("belief")
        if belief is not None:
            from_long_term = wm.get("belief_id") is not None
            if from_long_term and np.linalg.norm(belief - pos) < self.reach_tol:
                # At a location recalled from long-term memory but nothing is sensed: stale memory.
                if rctx.memory is not None:
                    n = rctx.memory.forget(MemoryQuery(key=self._key(rctx)))
                    rctx.emit("memory.write", {"op": "forget", "key": self._key(rctx), "n": n})
                wm.put("belief", None)
                wm.put("belief_id", None)
            else:
                source = "remembered (long-term)" if from_long_term else "remembered (this episode)"
                return Decision(kind="subgoal", output=belief, rationale=f"{source} goal location", trigger=trigger, component=self.name)
        i = wm.get("waypoint", 0)
        wp = np.array(SWEEP[i % len(SWEEP)])
        if np.linalg.norm(wp - pos) < self.reach_tol:
            i += 1
            wm.put("waypoint", i)
            wp = np.array(SWEEP[i % len(SWEEP)])
        return Decision(kind="subgoal", output=wp, rationale=f"explore waypoint {i}", trigger=trigger, component=self.name)

    def end(self, success: bool | None, rctx: ReasonerContext) -> None:
        observed = rctx.working.get("observed")
        if rctx.memory is None or observed is None:
            return
        goal, t = observed
        item = MemoryItem(kind=MemoryKind.SPATIAL, key=self._key(rctx),
                          content=f"goal of {rctx.task_id} observed at ({goal[0]:.2f}, {goal[1]:.2f})",
                          entities=(rctx.task_id,), scope={"task_id": rctx.task_id, "episode_id": rctx.episode_id},
                          pose=Pose(frame_id=FRAME, position=(float(goal[0]), float(goal[1]), 0.0)),
                          event_time=TimeSpan(start=t), payload={"episode_success": success}, source=self.name)
        res = rctx.memory.write([item])
        rctx.emit("memory.write", {"op": "write", "key": item.key, "events": list(res.events)})
