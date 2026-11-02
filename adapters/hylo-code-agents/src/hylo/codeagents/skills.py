"""Programs as sealed artifacts, and promoted programs as skills (RFC-0006 parts 5 and 6).

A program artifact is a directory with `solve.py` (defining `solve(robot, **kw)`), `program.json` (what it was made
for: task family, the instructions of its development cases, the session) and `artifact.json` (kind "program",
role "behaviour", content digest, provenance: development splits only). `code.program` runs it; a code candidate
points to it; once a campaign promotes it, it becomes a "program" skill in the skill library with its gate evidence
and digest, which `code.llm` retrieves by instruction similarity and calls as `skills.<name>(robot, **kw)`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from hylo.core.artifact import ArtifactManifest, Provenance, manifest_root, verify, write_manifest
from hylo.core.experience import Skill, SkillTrigger

from .sandbox import CodeRejected, defines, validate

PROGRAM_FILE = "solve.py"
META_FILE = "program.json"


def write_program(out_dir: str | Path, source: str, *, meta: Mapping[str, Any], provenance: Provenance,
                  metrics: Mapping[str, Any] | None = None, name: str = "") -> ArtifactManifest:
    """Validate and seal a program into `out_dir` (a new directory)."""
    tree = validate(source)
    if not defines(tree, "solve"):
        raise CodeRejected("a program must define solve(robot)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / PROGRAM_FILE).write_text(source.rstrip("\n") + "\n")
    (out / META_FILE).write_text(json.dumps({"entry": "solve", "file": PROGRAM_FILE, **dict(meta)}, indent=2, default=str))
    return write_manifest(out, kind="program", role="behaviour", provenance=provenance, name=name or out.name,
                          metrics=dict(metrics or {}))


def read_program(path: str | Path) -> tuple[str, ArtifactManifest, dict[str, Any]]:
    """(source, manifest, program.json) of a sealed program artifact; the digest is checked."""
    m = verify(path)
    if m.kind != "program":
        raise ValueError(f"{path} is a {m.kind} artifact, not a program")
    root = manifest_root(path)
    assert root is not None
    meta = json.loads((root / META_FILE).read_text()) if (root / META_FILE).exists() else {}
    return (root / meta.get("file", PROGRAM_FILE)).read_text(), m, meta


def identifier(name: str) -> str:
    ident = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return ident if ident and ident[0].isalpha() else f"skill_{ident}"


def program_skill(artifact: str | Path, *, name: str | None = None, version: str = "1", description: str | None = None,
                  family: str | None = None, preconditions: str = "", gate: Mapping[str, Any] | None = None) -> Skill:
    """A (proposed) program skill for a sealed program artifact."""
    _, m, meta = read_program(artifact)
    root = manifest_root(artifact)
    name = name or family                                # a family given by the caller names the skill too
    family = family or meta.get("family") or ""
    instructions = list(dict.fromkeys(meta.get("instructions") or []))[:12]
    desc = description or meta.get("description") or (f"solves tasks like: {instructions[0]}" if instructions
                                                       else "a verified program")
    return Skill(name=identifier(name or meta.get("name") or family or m.name), version=version, description=desc[:400],
                 trigger=SkillTrigger(task_tags=(family,) if family else ()), kind="program",
                 body={"artifact": str(root), "digest": m.digest, "entry": meta.get("entry", "solve"), "family": family,
                       "instructions": instructions, "preconditions": preconditions or meta.get("preconditions", "")},
                 evidence={"artifact_digest": m.digest, **({"gate": dict(gate)} if gate else {})})


def add_program_skill(library: Any, skill: Skill, gate: Mapping[str, Any] | None) -> str:
    """Record a program skill; it is verified only with a passed gate as evidence (else it stays proposed)."""
    sid = f"{skill.name}@{skill.version}"
    try:
        library.get(sid)
        return sid                                       # already recorded
    except KeyError:
        pass
    library.add(skill)
    if gate and gate.get("passed"):
        library.transition(sid, "verified", {"gate": dict(gate)})
    return sid


def promote_from_campaign(campaign_dir: str | Path, library: Any, *, family: str | None = None) -> list[str]:
    """Every promoted code candidate of a campaign (registry ledger) becomes a verified program skill."""
    reg = Path(campaign_dir) / "registry"
    ledger = reg / "ledger.jsonl"
    out = []
    if not ledger.exists():
        return out
    for line in ledger.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("op") != "promote":
            continue
        cand = json.loads((reg / "versions" / f"{rec['id']}.json").read_text())["candidate"]
        artifact = (cand.get("patch", {}).get("agent.params") or {}).get("artifact") or cand.get("patch", {}).get(
            "agent.params.artifact")
        if cand.get("kind") != "code" or not artifact:
            continue
        from hylo.core.store import Store, is_ref

        path = Store().path(artifact) if is_ref(artifact) else artifact
        skill = program_skill(path, family=family, gate=rec.get("gate"))
        skill = skill.model_copy(update={"evidence": {**skill.evidence, "candidate": rec["id"]}})
        out.append(add_program_skill(library, skill, rec.get("gate")))
    return out


def demote(library: Any, skill_id: str, evidence: Mapping[str, Any]) -> Skill:
    """A program skill whose re-validation failed is deprecated, like any demoted version."""
    return library.transition(skill_id, "deprecated", dict(evidence))


def load_program_skills(library: Any, instruction: str, define: Callable[[str, str], Callable[..., Any]], *,
                        k: int = 3, min_score: float = 0.2) -> tuple[SimpleNamespace, str, list[dict[str, Any]]]:
    """The verified program skills most similar to `instruction`, as callables `skills.<name>(robot, **kw)`.

    `define(source, entry)` compiles a program in the caller's sandbox (fresh namespace per skill). A skill whose
    artifact no longer verifies or no longer validates is skipped and reported."""
    funcs: dict[str, Callable[..., Any]] = {}
    lines, used = [], []
    for skill, score in library.similar(instruction, kinds=("program",), k=k, min_score=min_score):
        sid = f"{skill.name}@{skill.version}"
        try:
            source, m, _ = read_program(skill.body["artifact"])
            if m.digest != skill.body.get("digest"):
                raise ValueError(f"artifact digest {m.digest} != skill {skill.body.get('digest')}")
            validate(source)
        except Exception as exc:  # noqa: BLE001 - a broken skill is skipped, never run
            used.append({"skill": sid, "score": score, "skipped": str(exc)[:200]})
            continue
        entry = str(skill.body.get("entry", "solve"))
        cache: dict[str, Callable[..., Any]] = {}

        def call(robot: Any, _src: str = source, _entry: str = entry, _cache: dict = cache, **kw: Any) -> Any:
            if "fn" not in _cache:
                _cache["fn"] = define(_src, _entry)
            return _cache["fn"](robot, **kw)

        name = identifier(skill.name)
        funcs[name] = call
        pre = skill.body.get("preconditions") or ""
        lines.append(f"- skills.{name}(robot, **kw): {skill.description}"
                     + (f" Preconditions: {pre}" if pre else "") + (f" (family: {skill.body.get('family')})"
                                                                   if skill.body.get("family") else ""))
        used.append({"skill": sid, "score": score, "digest": m.digest})
    return SimpleNamespace(**funcs), "\n".join(lines), used


__all__ = ["META_FILE", "PROGRAM_FILE", "add_program_skill", "demote", "identifier", "load_program_skills",
           "program_skill", "promote_from_campaign", "read_program", "write_program"]
