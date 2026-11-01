"""Turn improvement-loop outcomes into skills (RFC-0003 M3: the point where memory and RSI meet).

Every archived candidate becomes a skill version whose status follows the evidence, never a model's opinion:
- promoted                      -> verified   (gate statistics as evidence)
- gated_fail, effect <= 0       -> refuted
- gated_fail, effect > 0        -> proposed   (promising but underpowered: worth re-testing with more episodes)
- rejected (stage-1 deficit)    -> refuted
- archived (not significant on search_val) -> proposed
Refuted skills stay in the library so proposers can see what was tried and failed.

    hylo skills <campaign dir | archive.jsonl> --out runs/skills/libero10 --tags libero libero_10 --env libero
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hylo.core.experience import Skill, SkillTrigger

from .protocol import ArchiveEntry
from .steering import STEERING_PATH


def _stage(entry: ArchiveEntry, n: int):
    return next((s for s in entry.stages if s.stage == n), None)


def classify(entry: ArchiveEntry) -> tuple[str, dict[str, Any]]:
    """(target status, evidence) for one archive entry."""
    gate, val = _stage(entry, 3), _stage(entry, 2)
    ev: dict[str, Any] = {"verdict": entry.verdict, "candidate": entry.candidate.id, "digest": entry.candidate.digest(),
                          "generator": entry.candidate.generator}
    best = gate or val
    if best is not None and best.success is not None and best.parent_success is not None:
        ev.update({"effect": round(best.success - best.parent_success, 4), "p_value": best.p_value,
                   "n_pairs": best.n_pairs, "gate_split": "gate" if gate else "search_val",
                   "wins": best.wins, "losses": best.losses})
    if entry.verdict == "promoted":
        return "verified", ev
    if entry.verdict == "rejected":
        return "refuted", ev
    if entry.verdict == "gated_fail":
        return ("refuted" if ev.get("effect", 0.0) <= 0 else "proposed"), ev
    return "proposed", ev


def skill_from_entry(entry: ArchiveEntry, *, task_tags: tuple[str, ...], env: str | None,
                     steering_path: str = STEERING_PATH, prefix: str = "skill") -> Skill | None:
    patch = entry.candidate.patch
    if set(patch) == {steering_path}:
        kind, body = "steering", {"steering": patch[steering_path]}
    elif any(p.endswith(".repo_id") for p in patch):
        kind, body = "weights", {"patch": patch}
    else:
        return None                                   # not expressible as a per-episode skill (e.g. agent ref swaps)
    return Skill(name=f"{prefix}-{entry.candidate.id.removeprefix('cand-')}", version="1",
                 description=entry.hypothesis[:400], trigger=SkillTrigger(task_tags=task_tags, env=env),
                 kind=kind, body=body, evidence={"rationale": entry.candidate.rationale[:400]})


def read_archive(archive: str | Path) -> list[ArchiveEntry]:
    path = Path(archive)
    path = path / "archive.jsonl" if path.is_dir() else path
    return [ArchiveEntry.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


def skills_from_archive(archive: str | Path, library: Any, **kw: Any) -> dict[str, Any]:
    return skills_from_entries(read_archive(archive), library, **kw)


def skills_from_entries(entries: list[ArchiveEntry], library: Any, *, task_tags: tuple[str, ...] = (),
                        env: str | None = None, prefix: str = "skill") -> dict[str, Any]:
    """Add one skill per new archive entry and move it to the status its evidence supports. Idempotent."""
    summary: dict[str, Any] = {"added": [], "skipped": [], "by_status": {}}
    for entry in entries:
        if entry.verdict == "pending":
            continue
        skill = skill_from_entry(entry, task_tags=task_tags, env=env, prefix=prefix)
        if skill is None:
            summary["skipped"].append(entry.candidate.id)
            continue
        sid = f"{skill.name}@{skill.version}"
        status, evidence = classify(entry)
        try:
            library.get(sid)
            continue                                   # already recorded
        except KeyError:
            pass
        library.add(skill)
        if status != "proposed":
            library.transition(sid, status, evidence)
        else:
            library.annotate(sid, evidence)
        summary["added"].append(sid)
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
    return summary


def credit_lessons(book: Any, entries: list[ArchiveEntry]) -> list[dict[str, Any]]:
    """Outcome feedback: lessons a decided candidate cited get +1 (verified) or -1 (refuted). Once per candidate."""
    seen, done = book.sources(), []
    for e in entries:
        cited = list(e.candidate.evidence.get("lessons") or [])
        status, _ = classify(e)
        source = f"credit:{e.candidate.id}"
        if not cited or status == "proposed" or e.verdict == "pending" or source in seen:
            continue
        op = "UPVOTE" if status == "verified" else "DOWNVOTE"
        done += book.apply([{"op": op, "id": lid} for lid in cited], source=source)
    return done


class ExperienceProposer:
    """Wraps any proposer with the experience layer (RFC-0003 M4). Before each proposal round:
    1. the archive so far becomes skills (status from evidence);
    2. lessons cited by decided candidates are credited (+1 verified / -1 refuted);
    3. a reflector (optional) distils new lessons from the parent's search-split diagnosis;
    4. the inner proposer sees `diagnosis["lessons"]` and `diagnosis["skills"]` (skill status only, no gate numbers).

        proposer: {type: experience, params: {inner: {type: llm.steering, params: {...}},
                   lessons: {type: lesson_book, params: {path: runs/lessons/l10}},
                   skills: {type: skill_library, params: {path: runs/skills/l10}},
                   reflector: {type: llm.reflector, params: {...}}, skill_tags: [libero, libero_10], env: libero}}
    """

    def __init__(self, inner: dict[str, Any] | Any, lessons: dict[str, Any] | Any | None = None,
                 skills: dict[str, Any] | Any | None = None, reflector: dict[str, Any] | Any | None = None,
                 skill_tags: list[str] | None = None, env: str | None = None, prefix: str = "skill",
                 top_lessons: int = 8, max_skills: int = 16) -> None:
        from hylo.core.registry import registry

        build = lambda group, spec: registry.build(group, spec) if isinstance(spec, dict) else spec  # noqa: E731
        self.inner = build("generators", inner)
        self.lessons, self.skills = build("memory", lessons), build("memory", skills)
        self.reflector = build("reflectors", reflector)
        self.skill_tags, self.env, self.prefix = tuple(skill_tags or ()), env, prefix
        self.top_lessons, self.max_skills = top_lessons, max_skills
        self.name = f"experience({getattr(self.inner, 'name', type(self.inner).__name__)})"
        self.last: dict[str, Any] = {}

    def propose(self, diagnosis: Any, cfg: Any, parent_id: str | None, archive: list[ArchiveEntry]) -> list:
        report: dict[str, Any] = {}
        if self.skills is not None:
            report["skills"] = skills_from_entries(archive, self.skills, task_tags=self.skill_tags, env=self.env,
                                                   prefix=self.prefix)
        ctx = dict(diagnosis)
        if self.lessons is not None:
            report["credited"] = credit_lessons(self.lessons, archive)
            run_dir = diagnosis.get("run_dir")
            if self.reflector is not None and run_dir:
                source = f"reflect:{Path(run_dir).name}"
                if source not in self.lessons.sources():
                    ops = self.reflector.reflect(diagnosis, self.lessons)
                    report["reflected"] = self.lessons.apply(ops, source=source) if ops else []
            short = set(diagnosis.get("per_task") or {})            # diagnose_traces keys tasks by their short name
            scopes = ("global", *{l.scope for l in self.lessons.all("active") if l.scope.rsplit("/", 1)[-1] in short})
            ctx["lessons"] = [{"id": l.id, "scope": l.scope, "kind": l.kind, "title": l.title, "content": l.content,
                               "helpful": l.helpful, "harmful": l.harmful}
                              for l in self.lessons.query(scopes, top_k=self.top_lessons)]
        if self.skills is not None:
            ctx["skills"] = self.skills.index(status=None, with_stats=False)[: self.max_skills]
        self.last = report
        return self.inner.propose(ctx, cfg, parent_id, archive)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from hylo.core.registry import registry

    ap = argparse.ArgumentParser(prog="hylo skills")
    ap.add_argument("archive", help="campaign directory (with archive.jsonl) or an archive.jsonl file")
    ap.add_argument("--out", required=True, help="skill library directory")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--env")
    ap.add_argument("--prefix", default="skill")
    a = ap.parse_args(argv)
    lib = registry.build("memory", {"type": "skill_library", "params": {"path": a.out}})
    print(json.dumps(skills_from_archive(a.archive, lib, task_tags=tuple(a.tags), env=a.env, prefix=a.prefix), indent=2))
    return 0
