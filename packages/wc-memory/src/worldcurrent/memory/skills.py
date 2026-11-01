"""SkillLibrary: verified, versioned procedures for task families (RFC-0003 M3).

A skill is data, never code: a plan template, a steering or recovery config patch, or a weights reference.
Lifecycle: proposed -> verified (only through the RSI gate, with its statistics as evidence) | refuted;
verified -> deprecated. `match` returns verified skills only, with the slot bindings captured by the trigger's
instruction pattern. Usage feedback (useful / misleading / not_applicable) accumulates per version.

Storage: <root>/<name>@<version>.json, one file per skill version. A library opened with frozen=True (evaluation)
refuses every write, and `snapshot()` is the content digest recorded in the agent spec. `export_skill_md` writes
one SKILL.md per verified skill (name + description front matter, body below) so an LLM can scan the index and
read bodies on demand.

    skills: {type: skill_library, params: {path: runs/skills/libero10, frozen: true}}
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from worldcurrent.core.experience import Skill

from .action_bank import FrozenMemoryError

TRANSITIONS = {"proposed": {"verified", "refuted"}, "verified": {"deprecated"}, "refuted": set(), "deprecated": set()}


def normalise_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


class SkillLibrary:
    def __init__(self, path: str, frozen: bool = False) -> None:
        self.root = Path(path)
        self.frozen = frozen
        if not frozen:
            self.root.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        if self.root.exists():
            for f in sorted(self.root.glob("*@*.json")):
                s = Skill.model_validate_json(f.read_text())
                self._skills[self._id(s)] = s

    @staticmethod
    def _id(s: Skill) -> str:
        return f"{s.name}@{s.version}"

    def _write(self, s: Skill) -> None:
        if self.frozen:
            raise FrozenMemoryError("this skill library is a frozen snapshot")
        self._skills[self._id(s)] = s
        (self.root / f"{self._id(s)}.json").write_text(s.model_dump_json(indent=2))

    # -- lifecycle ------------------------------------------------------------------------------------
    def add(self, skill: Skill) -> str:
        if skill.status != "proposed":
            raise ValueError("new skills start as 'proposed'; verification happens through the gate")
        if self._id(skill) in self._skills:
            raise ValueError(f"{self._id(skill)} exists; bump the version")
        if skill.trigger.instruction_pattern:
            re.compile(skill.trigger.instruction_pattern)
        self._write(skill)
        return self._id(skill)

    def transition(self, skill_id: str, status: str, evidence: dict[str, Any] | None = None) -> Skill:
        s = self._skills[skill_id]
        if status not in TRANSITIONS[s.status]:
            raise ValueError(f"{skill_id}: {s.status} -> {status} is not allowed")
        s = s.model_copy(update={"status": status, "evidence": {**s.evidence, **(evidence or {})}})
        self._write(s)
        return s

    def annotate(self, skill_id: str, evidence: dict[str, Any]) -> Skill:
        """Attach evidence without a status change (e.g. an underpowered gate result on a proposed skill)."""
        s = self._skills[skill_id]
        s = s.model_copy(update={"evidence": {**s.evidence, **evidence}})
        self._write(s)
        return s

    def record_verdict(self, skill_id: str, verdict: str) -> None:
        if verdict not in ("useful", "misleading", "not_applicable"):
            raise ValueError(verdict)
        s = self._skills[skill_id]
        self._write(s.model_copy(update={"verdicts": {**s.verdicts, verdict: s.verdicts.get(verdict, 0) + 1}}))

    # -- read -------------------------------------------------------------------------------------------
    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def all(self, status: str | None = None) -> list[Skill]:
        return [s for s in self._skills.values() if status is None or s.status == status]

    def match(self, instruction: str, *, task_tags: tuple[str, ...] = (), env: str | None = None,
              kinds: tuple[str, ...] | None = None) -> list[tuple[Skill, dict[str, str]]]:
        """Verified skills whose trigger fits, most specific first (pattern > tags > env), then most useful."""
        text = normalise_instruction(instruction)
        out = []
        for s in self.all("verified"):
            t = s.trigger
            if kinds and s.kind not in kinds:
                continue
            if t.env and env and t.env != env:
                continue
            if t.task_tags and not set(t.task_tags) <= set(task_tags):
                continue
            binds: dict[str, str] = {}
            if t.instruction_pattern:
                m = re.fullmatch(t.instruction_pattern, text)
                if m is None:
                    continue
                binds = {k: v for k, v in m.groupdict().items() if v is not None}
            specificity = (2 if t.instruction_pattern else 0) + (1 if t.task_tags else 0)
            score = s.verdicts.get("useful", 0) - s.verdicts.get("misleading", 0)
            out.append(((specificity, score), s, binds))
        out.sort(key=lambda x: x[0], reverse=True)
        return [(s, b) for _, s, b in out]

    def snapshot(self) -> str:
        h = hashlib.sha256()
        for sid in sorted(self._skills):
            h.update(self._skills[sid].model_dump_json().encode())
        return h.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self._skills)

    # -- LLM-facing export --------------------------------------------------------------------------------
    def index(self, status: str | None = "verified", *, with_stats: bool = False) -> list[dict[str, Any]]:
        """What a proposer or planner may see. Gate numbers are withheld by default: a proposer that reads effect
        sizes and p-values can adaptively overfit the gate split, so it gets the status (a verdict) only."""
        out = []
        for s in self.all(status):
            item = {"id": self._id(s), "kind": s.kind, "status": s.status, "description": s.description,
                    "trigger": s.trigger.model_dump(exclude_none=True), "body": s.body, "verdicts": s.verdicts}
            if with_stats:
                item["evidence"] = {k: v for k, v in s.evidence.items() if k in ("effect", "p_value", "gate_split", "n_pairs")}
            out.append(item)
        return out

    def export_skill_md(self, out_dir: str) -> int:
        out = Path(out_dir)
        n = 0
        for s in self.all("verified"):
            d = out / s.name
            d.mkdir(parents=True, exist_ok=True)
            front = {"name": s.name, "description": s.description, "version": s.version, "kind": s.kind}
            body = json.dumps({"trigger": s.trigger.model_dump(exclude_none=True), "body": s.body,
                               "evidence": s.evidence}, indent=2)
            (d / "SKILL.md").write_text("---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in front.items())
                                        + f"\n---\n\n{s.description}\n\n```json\n{body}\n```\n")
            n += 1
        return n
