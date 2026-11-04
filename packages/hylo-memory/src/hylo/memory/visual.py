"""Evidence-linked visual reports for frozen-model planners.

Reports are fallible model interpretations, never simulator facts. The journal records that
a model emitted a report, with its original confidence and frame citations intact. Retrieval
keeps early cues and recent updates without replacing the append-only evidence history.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from pydantic import Field

from hylo.core.contract import Contract
from hylo.core.episode_memory import EpisodeScope, MemoryEvent, RecallQuery

from .episode import EventMemory


class VisualClaim(Contract):
    topic: str = Field(min_length=1, max_length=120)
    claim: str = Field(min_length=1, max_length=1200)
    frame_ids: tuple[int, ...] = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class VisualPoint(Contract):
    id: str = Field(pattern=r"^point_[12]$")
    frame_id: int = Field(ge=0)
    x: int = Field(ge=0, le=999)
    y: int = Field(ge=0, le=999)


class VisualPlan(Contract):
    observations: tuple[VisualClaim, ...] = Field(default=(), max_length=12)
    subgoal: str = Field(min_length=1, max_length=300)
    points: tuple[VisualPoint, ...] = Field(default=(), max_length=2)


def render_subgoal(plan: VisualPlan, *, current_frame_id: int, width: int, height: int) -> str:
    """Render normalized current-view points in RoboMME's <row, column> convention."""
    if min(width, height) < 1:
        raise ValueError("invalid image size")
    rendered = plan.subgoal
    seen = set()
    for p in plan.points:
        if p.frame_id != current_frame_id or p.id in seen:
            raise ValueError("grounding must use unique points in the current front frame")
        seen.add(p.id)
        token = f"<{p.id}>"
        if token not in rendered:
            raise ValueError("unused grounding point")
        rendered = rendered.replace(token, f"<{round(p.y * (height - 1) / 999)}, {round(p.x * (width - 1) / 999)}>")
    if re.search(r"<point_", rendered):
        raise ValueError("unresolved grounding point")
    return rendered


class VisualReportMemory:
    """An episode-scoped, bounded prompt view over durable, unverified reports.

    Frame IDs are monotonically assigned by the observation adapter. Only citations actually
    supplied to the model in the current request are accepted. No success flag, reward, task
    oracle, environment instance, or model-generated completion claim can advance a counter.
    """

    def __init__(self, journal: EventMemory, *, max_chars: int = 8000) -> None:
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256")
        self.journal = journal
        self.max_chars = max_chars
        self._round = 0

    def reset(self, scope: EpisodeScope) -> None:
        self.journal.reset(scope)
        self._round = 0

    def write(self, plan: VisualPlan, *, supplied_frame_ids: Sequence[int], step: int) -> int:
        allowed = set(supplied_frame_ids)
        if step < 0 or any(i < 0 or i > step for i in allowed):
            raise ValueError("supplied frames must be causal nonnegative frame IDs")
        for claim in plan.observations:
            if not set(claim.frame_ids).issubset(allowed):
                raise ValueError("visual report cites a frame absent from this model request")
        # Validate the complete response before writing any part of it.
        events = [MemoryEvent(
            id=f"visual-report:{self._round}:{i}", step=step, order=100 + i,
            time=float(step), kind="visual.report", subject=c.topic,
            value={**c.model_dump(mode="json"), "step": step, "round": self._round},
            evidence=tuple(f"frame:{f}" for f in c.frame_ids),
            source="frozen-vlm/report", confidence=1.0,
        ) for i, c in enumerate(plan.observations)]
        # Confidence 1 means the emission was observed, NOT that the claim is correct.
        count = self.journal.append(events)
        self._round += 1
        return count

    def context(self, instruction: str) -> dict:
        result = self.journal.recall(RecallQuery(
            name="visual_reports", op="sequence", kind="visual.report",
            limit=self.journal.max_events, complete=False,
        ))
        reports = result.value or []
        terms = set(re.findall(r"[a-z]+", instruction.lower()))
        first, latest = {}, {}
        for i, report in enumerate(reports):
            first.setdefault(report["topic"], i)
            latest[report["topic"]] = i
        # First and latest versions of every topic are candidates, so an updated location
        # does not erase the demonstration cue or falsely present the old location as current.
        candidates = set(first.values()) | set(latest.values())
        candidates.update(range(max(0, len(reports) - 12), len(reports)))
        def priority(i: int) -> tuple:
            r = reports[i]
            overlap = len(terms & set(re.findall(r"[a-z]+", (r["topic"] + " " + r["claim"]).lower())))
            return (overlap, i in first.values(), i)
        chosen: list[int] = []
        def payload(indices: list[int]) -> dict:
            return {
                "status": "unverified_visual_reports",
                "reports": [reports[i] for i in sorted(indices)],
                "omitted_reports": len(reports) - len(indices),
                "snapshot": self.journal.snapshot(),
            }
        for i in sorted(candidates, key=priority, reverse=True):
            proposed = chosen + [i]
            if len(json.dumps(payload(proposed), ensure_ascii=False)) <= self.max_chars:
                chosen = proposed
        return payload(chosen)
