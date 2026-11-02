"""Derived artifacts (RFC-0004): everything the system learns is a versioned, digest-pinned directory with provenance.

The trace store is the single source of truth. Action memories, critics, value models, fine-tuned weights, datasets,
judges are all *derived* from traces (or from external data sources) by a builder, and each one carries an
`artifact.json` manifest: what it is, which role it plays, a content digest, and which runs / splits it was built
from. Three roles, three trust paths:

- behaviour   changes what the robot does (memory, critic, noise policy, weights): only the RSI gate activates it;
- instrument  measures or labels (success judge, progress model, world model): trusted only after calibration
              against ground truth, never the gate's verdict;
- proposal    only changes what gets proposed (lessons, skill index); curated by outcome votes.

One split guard for every builder: development splits (search, search_val) and calibrated external sources may be
read; gate and held-out data never. The gate re-checks the provenance of every artifact a candidate references, so a
leak is caught even if a builder forgot the guard.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pydantic import Field

from .contract import Contract, JsonAny

ARTIFACT_SCHEMA = "hylo.artifact/v0.1"
MANIFEST = "artifact.json"
DEV_SPLITS: tuple[str, ...] = ("search", "search_val")
FORBIDDEN_SPLIT_PREFIXES: tuple[str, ...] = ("gate", "heldout")

ArtifactKind = Literal["dataset", "action_memory", "value_model", "noise_policy", "policy_weights", "policy_params", "success_judge",
                       "progress_model", "world_model", "lesson_book", "skill_library", "program", "other"]
ArtifactRole = Literal["behaviour", "instrument", "proposal", "data"]


class ArtifactError(ValueError):
    """Missing / corrupted manifest, digest mismatch, or forbidden data in an artifact's provenance."""


class SplitLeak(ArtifactError):
    """A builder or a candidate would use gate / held-out data."""


# ---------------------------------------------------------------- split guard

def forbidden_split(split: str | None) -> bool:
    return bool(split) and str(split).startswith(FORBIDDEN_SPLIT_PREFIXES)


def check_splits(splits: Iterable[str]) -> tuple[str, ...]:
    """Raise if any requested split is gate / held-out data (call this before reading anything)."""
    splits = tuple(splits)
    bad = [s for s in splits if forbidden_split(s)]
    if bad:
        raise SplitLeak(f"gate / heldout data must never be used to build anything that is gated on it: {bad}")
    return splits


class SourceRef(Contract):
    """One input of a build: a run directory (kind 'run') or an external data source (kind 'external')."""

    kind: Literal["run", "external", "artifact"] = "run"
    id: str                                   # run_id, dataset name, or artifact digest
    split: str | None = None                  # benchmark split of a run; "external:<name>" for outside data
    path: str | None = None
    n_episodes: int | None = None
    licence: str | None = None


def run_split(run_dir: str | Path) -> str | None:
    m = Path(run_dir) / "manifest.json"
    return json.loads(m.read_text()).get("benchmark", {}).get("split") if m.exists() else None


def synthetic_run(run_dir: str | Path) -> bool:
    """A run produced inside a learned model (world-model pre-screening), not by a real or simulated environment."""
    m = Path(run_dir) / "manifest.json"
    if not m.exists():
        return False
    env = (json.loads(m.read_text()).get("config") or {}).get("env") or {}
    return str(env.get("type", "")).startswith("world_model")


def find_runs(roots: Iterable[str | Path]) -> list[Path]:
    """Run directories (those with an evallog.json) given run directories, store ids or roots to search."""
    from .store import is_ref, resolve_refs

    out: list[Path] = []
    for r in roots:
        p = Path(resolve_refs(r) if is_ref(r) else r)
        out += [p] if (p / "evallog.json").exists() else sorted(q.parent for q in p.rglob("evallog.json"))
    return out


def development_runs(roots: Iterable[str | Path], splits: Iterable[str] = DEV_SPLITS, *,
                     strict: bool = False) -> list[tuple[Path, SourceRef]]:
    """The run directories under `roots` whose split is in `splits`, with their provenance records.

    strict=True raises on a run of any other split (a CLI given an explicit list of runs); strict=False skips them
    (a builder scanning a campaign directory, which also contains gate and held-out runs). Ingested external data
    (split external:<name>) is included when "external" is among `splits`."""
    allowed = check_splits(splits)
    out = []
    for rd in find_runs(roots):
        split = run_split(rd)
        if synthetic_run(rd) and "synthetic" not in allowed:
            continue                                   # model-generated episodes never become training data silently
        if split not in allowed and not ("external" in allowed and str(split).startswith("external:")):
            if strict:
                raise SplitLeak(f"{rd} is a {split!r} run; only {list(allowed)} may be used")
            continue
        log = json.loads((rd / "evallog.json").read_text())
        out.append((rd, SourceRef(kind="run", id=str(log.get("run_id") or rd.name), split=split, path=str(rd),
                                  n_episodes=len(log.get("samples", [])))))
    return out


# ---------------------------------------------------------------- manifest

class Provenance(Contract):
    builder: str                              # e.g. "hylo.agents.critic_train@0.1"
    sources: list[SourceRef] = Field(default_factory=list)
    params: dict[str, JsonAny] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def splits(self) -> set[str]:
        return {s.split for s in self.sources if s.split}


class ArtifactManifest(Contract):
    schema_version: Literal["hylo.artifact/v0.1"] = ARTIFACT_SCHEMA
    kind: ArtifactKind
    role: ArtifactRole
    name: str = ""
    digest: str                               # sha256[:16] over every file except the manifest
    mutable: bool = False                     # a live store (lesson book while curating): digest not enforced
    provenance: Provenance
    metrics: dict[str, JsonAny] = Field(default_factory=dict)
    calibration: dict[str, JsonAny] | None = None   # instruments only: how well it agrees with ground truth


def dir_digest(root: str | Path, exclude: tuple[str, ...] = (MANIFEST,)) -> str:
    root = Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for f in files:
        rel = f.name if root.is_file() else f.relative_to(root).as_posix()
        if rel in exclude:
            continue
        h.update(rel.encode() + b"\0")
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()[:16]


def write_manifest(root: str | Path, *, kind: ArtifactKind, role: ArtifactRole, provenance: Provenance,
                   name: str = "", metrics: dict[str, Any] | None = None, mutable: bool = False,
                   calibration: dict[str, Any] | None = None) -> ArtifactManifest:
    """Seal a built directory: compute its digest and write artifact.json (written last, after every other file)."""
    root = Path(root)
    for s in provenance.sources:
        if forbidden_split(s.split):
            raise SplitLeak(f"refusing to seal {root}: source {s.id} is a {s.split!r} run")
    m = ArtifactManifest(kind=kind, role=role, name=name or root.name, digest=dir_digest(root), mutable=mutable,
                         provenance=provenance, metrics=json.loads(json.dumps(metrics or {}, default=str)),
                         calibration=calibration)
    (root / MANIFEST).write_text(m.model_dump_json(indent=2))
    return m


def manifest_root(path: str | Path, max_up: int = 4) -> Path | None:
    """The artifact directory containing `path` (the path itself or one of its first `max_up` parents)."""
    p = Path(path)
    for _ in range(max_up + 1):
        if (p / MANIFEST).is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def read_manifest(path: str | Path) -> ArtifactManifest | None:
    root = manifest_root(path)
    return ArtifactManifest.model_validate_json((root / MANIFEST).read_text()) if root else None


def verify(path: str | Path) -> ArtifactManifest:
    """Load and check an artifact: manifest present, digest intact (unless mutable), no gate / held-out sources."""
    root = manifest_root(path)
    if root is None:
        raise ArtifactError(f"{path} is not inside an artifact (no {MANIFEST})")
    m = ArtifactManifest.model_validate_json((root / MANIFEST).read_text())
    if not m.mutable and dir_digest(root) != m.digest:
        raise ArtifactError(f"{root}: content digest {dir_digest(root)} != manifest {m.digest} (modified after sealing)")
    leaks = sorted(s for s in m.provenance.splits() if forbidden_split(s))
    if leaks:
        raise SplitLeak(f"{root} was built from {leaks} data")
    return m


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def referenced_artifacts(patch: Any) -> dict[str, ArtifactManifest]:
    """Every artifact a config (patch) points to, verified: {artifact dir: manifest}. Store ids are resolved (and
    their digest must match the content); strings that are not existing paths (repo ids, factory refs) are ignored;
    an existing path inside an artifact must verify or this raises."""
    from .store import Store, is_ref

    found: dict[str, ArtifactManifest] = {}
    store: Store | None = None
    for s in _strings(patch):
        if is_ref(s):
            store = store or Store()
            root = store.path(s)
            m = verify(root)
            want = s.partition("@")[2]
            if s.startswith("artifact:") and want and not m.digest.startswith(want):
                raise ArtifactError(f"{s}: {root} now holds digest {m.digest}")
            found[str(root)] = m
            continue
        if "/" not in s and "\\" not in s:
            continue
        try:
            if not Path(s).exists():
                continue
        except OSError:
            continue
        root = manifest_root(s)
        if root is not None and str(root) not in found:
            found[str(root)] = verify(root)
    return found
