"""Run and artifact store (hylo.store/v0.1): global ids, references instead of paths, deletion protection, sync.

A store is a directory tree on one machine (this machine's profile says where: `hylo store init PATH`, else $HYLO_STORE,
else ./runs). Runs (directories with an evallog.json) and sealed artifacts (an artifact.json) anywhere under it get
an id:

    run:<run_id>                       run ids are unique (they end in a random suffix)
    artifact:<name>@<digest>           name from the manifest, digest = content digest (a prefix of >= 8 hex works)
    artifact:<name>                    allowed when exactly one artifact has that name; the run manifest pins it

Configs reference artifacts by id (`critic: {path: artifact:l10-v1@3d8d6096f7111acc}`); the registry resolves ids to
this machine's paths when it builds a component, after the experiment digest is taken, so the same config means the
same content on every machine. Learner job inputs accept run and artifact ids too.

The index (<root>/.hylo-store/index.json) is a cache, rebuilt by walking the tree when an id is not found. It also
records who refers to what: an artifact's provenance (the runs and artifacts it was built from) and a campaign's
archive and promotion ledger. `hylo store rm` refuses to delete anything still referred to unless forced, and a forced
or plain removal leaves a tombstone, so a dangling reference says what happened instead of "not found".

`hylo store push|pull HOST:PATH [ID ...]` copies runs and artifacts between machines with rsync (runs from the
robot computer to the workstation, artifacts the other way). Sealed artifacts are never overwritten: the same path
holding a different digest on the other side is a conflict and nothing is copied.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

from pydantic import Field

from .contract import Contract, JsonAny

STORE_SCHEMA = "hylo.store/v0.1"
META_DIR = ".hylo-store"
SKIP_DIRS = {"blobs", "episodes", "__pycache__", META_DIR, "node_modules", ".venv"}
REF = re.compile(r"^(?P<kind>run|artifact):(?P<name>[A-Za-z0-9][A-Za-z0-9._\-]*)(?:@(?P<digest>[0-9a-f]{8,64}))?$")


class StoreError(ValueError):
    """Unknown or ambiguous id, a removed entry, a refused deletion or a sync conflict."""


class Entry(Contract):
    id: str
    kind: Literal["run", "artifact"]
    name: str
    digest: str | None = None                 # artifact: content digest; run: config digest
    path: str                                 # relative to the store root
    created_at: float | None = None
    detail: dict[str, JsonAny] = Field(default_factory=dict)
    refs: list[str] = Field(default_factory=list)     # ids this entry was built from


class Index(Contract):
    schema_version: Literal["hylo.store/v0.1"] = STORE_SCHEMA
    built_at: float = Field(default_factory=time.time)
    entries: dict[str, Entry] = Field(default_factory=dict)
    referrers: dict[str, list[str]] = Field(default_factory=dict)   # id -> ids / campaigns that refer to it


def is_ref(value: Any) -> bool:
    return isinstance(value, str) and REF.match(value) is not None


def default_root() -> Path:
    if os.environ.get("HYLO_STORE"):
        return Path(os.environ["HYLO_STORE"]).expanduser().resolve()
    from .profile import load_profile

    store = load_profile().store
    return Path(os.path.expandvars(os.path.expanduser(store))).resolve() if store else Path("runs").resolve()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class Store:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else default_root()
        self._index: Index | None = None

    @property
    def meta(self) -> Path:
        return self.root / META_DIR

    # ---------------------------------------------------------------- index
    def _walk(self) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            if "artifact.json" in filenames or "evallog.json" in filenames:
                yield Path(dirpath)

    def rebuild(self) -> Index:
        from .artifact import MANIFEST

        entries: dict[str, Entry] = {}
        by_digest: dict[str, str] = {}
        if self.root.is_dir():
            for d in self._walk():
                rel = d.relative_to(self.root).as_posix()
                m = _read_json(d / MANIFEST) if (d / MANIFEST).is_file() else None
                if isinstance(m, dict) and m.get("digest"):
                    prov = m.get("provenance") or {}
                    name = m.get("name") or d.name
                    e = Entry(id=f"artifact:{name}@{m['digest']}", kind="artifact", name=name, digest=m["digest"],
                              path=rel, created_at=prov.get("created_at"),
                              detail={"kind": m.get("kind"), "role": m.get("role"), "builder": prov.get("builder"),
                                      "mutable": bool(m.get("mutable")),
                                      "splits": sorted({s["split"] for s in prov.get("sources", []) if s.get("split")}),
                                      "sources": prov.get("sources", [])})
                    entries.setdefault(e.id, e)
                    by_digest.setdefault(m["digest"], e.id)
                log = _read_json(d / "evallog.json") if (d / "evallog.json").is_file() else None
                if isinstance(log, dict) and "samples" in log:
                    man = _read_json(d / "manifest.json") or {}
                    run_id = str(log.get("run_id") or d.name)
                    bench = man.get("benchmark") or log.get("benchmark") or {}
                    e = Entry(id=f"run:{run_id}", kind="run", name=run_id, digest=man.get("config_digest"), path=rel,
                              created_at=man.get("created_at"),
                              detail={"split": bench.get("split"), "benchmark": bench.get("id"),
                                      "episodes": len(log["samples"]), "config": (man.get("config") or {}).get("name")})
                    entries.setdefault(e.id, e)
        by_path: dict[str, list[str]] = {}                          # ingested data is both a run and an artifact
        for e in entries.values():
            by_path.setdefault(e.path, []).append(e.id)
        run_ids = {e.name: e.id for e in entries.values() if e.kind == "run"}
        for key, e in list(entries.items()):                        # what each artifact was built from
            if e.kind != "artifact":
                continue
            refs: set[str] = set()
            for s in e.detail.get("sources") or []:
                if s.get("kind") == "run" and f"run:{s.get('id')}" in entries:
                    refs.add(f"run:{s['id']}")
                elif s.get("kind") == "artifact" and s.get("id") in by_digest:
                    refs.add(by_digest[s["id"]])
                elif s.get("path"):
                    refs.update(self._ids_in(s["path"], by_path, run_ids, entries))
            refs -= set(by_path[e.path])
            entries[key] = e.model_copy(update={"refs": sorted(refs)})
        referrers: dict[str, set[str]] = {}
        for e in entries.values():
            for r in e.refs:
                referrers.setdefault(r, set()).add(e.id)
        for camp, found in self._campaign_refs(by_path, run_ids, entries, by_digest):
            for r in found:
                referrers.setdefault(r, set()).add(camp)
        idx = Index(entries=entries, referrers={k: sorted(v) for k, v in referrers.items()})
        self._save(idx)
        return idx

    def _ids_in(self, s: str, by_path: dict[str, list[str]], run_ids: dict[str, str], entries: dict[str, Entry]) -> set[str]:
        """The entries a string in a ledger or provenance points to: an id, a path inside an entry (absolute or
        relative to the store or its parent), or a run id as a path component (paths recorded on another machine)."""
        if m := REF.match(s):
            try:
                return {self._match(m, entries).id}
            except StoreError:
                return set()
        if "/" not in s:
            return {run_ids[s]} if s in run_ids else set()
        found: set[str] = set()
        p = Path(s)
        for base in ((p,) if p.is_absolute() else (self.root / p, self.root.parent / p)):
            try:
                rel = Path(os.path.normpath(base)).relative_to(self.root)
            except ValueError:
                continue
            parts = rel.parts
            for i in range(len(parts), 0, -1):
                key = "/".join(parts[:i])
                if key in by_path:
                    found.update(by_path[key])
                    break
        if not found:
            found.update(run_ids[c] for c in p.parts if c in run_ids)
        return found

    def _campaign_refs(self, by_path, run_ids, entries, by_digest) -> Iterator[tuple[str, set[str]]]:
        if not self.root.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            d = Path(dirpath)
            ledgers = [d / f for f in ("archive.jsonl",) if f in filenames]
            if d.name == "registry" and "ledger.jsonl" in filenames:
                ledgers.append(d / "ledger.jsonl")
            for f in ledgers:
                camp = f"campaign:{(d.parent if d.name == 'registry' else d).relative_to(self.root).as_posix()}"
                found: set[str] = set()
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for s in _strings(rec):
                        if s in by_digest:
                            found.add(by_digest[s])
                        else:
                            found |= self._ids_in(s, by_path, run_ids, entries)
                if found:
                    yield camp, found

    def _save(self, idx: Index) -> None:
        self._index = idx
        try:
            self.meta.mkdir(parents=True, exist_ok=True)
            tmp = self.meta / f"index.json.{os.getpid()}"
            tmp.write_text(idx.model_dump_json())
            os.replace(tmp, self.meta / "index.json")
        except OSError:
            pass                                        # a read-only store still resolves from memory

    def index(self, *, refresh: bool = False) -> Index:
        if refresh:
            return self.rebuild()
        if self._index is None:
            raw = _read_json(self.meta / "index.json")
            self._index = Index.model_validate(raw) if isinstance(raw, dict) else self.rebuild()
        return self._index

    def entries(self, kind: str | None = None, *, refresh: bool = False) -> list[Entry]:
        return [e for e in self.index(refresh=refresh).entries.values() if kind in (None, e.kind)]

    # ---------------------------------------------------------------- lookup
    def _match(self, m: re.Match, entries: dict[str, Entry]) -> Entry:
        kind, name, digest = m["kind"], m["name"], m["digest"]
        if kind == "run":
            e = entries.get(f"run:{name}")
            if e is None:
                raise StoreError(f"run:{name} not found")
            return e
        hits = [e for e in entries.values() if e.kind == "artifact" and e.name == name
                and (digest is None or (e.digest or "").startswith(digest))]
        if len(hits) == 1:
            return hits[0]
        ref = f"artifact:{name}" + (f"@{digest}" if digest else "")
        if not hits:
            known = sorted(e.id for e in entries.values() if e.kind == "artifact" and e.name == name)
            raise StoreError(f"{ref} not found" + (f"; with that name: {known}" if known else ""))
        raise StoreError(f"{ref} is ambiguous: {sorted(e.id for e in hits)}; add the digest")

    def get(self, ref: str) -> Entry:
        m = REF.match(ref)
        if m is None:
            raise StoreError(f"{ref!r} is not a store id (run:<id> or artifact:<name>[@<digest>])")
        for refresh in (False, True):                   # the index is a cache: a miss rebuilds it once
            try:
                e = self._match(m, self.index(refresh=refresh).entries)
            except StoreError as exc:
                if not refresh:
                    continue
                gone = self.tombstone(ref)
                if gone:
                    raise StoreError(f"{ref} was removed at {time.ctime(gone['removed_at'])} "
                                     f"({gone.get('reason') or 'no reason given'})") from None
                raise StoreError(f"{exc} in the store at {self.root}") from None
            if (self.root / e.path).is_dir():
                return e
        raise StoreError(f"{ref}: indexed at {e.path} but missing in {self.root}")

    def path(self, ref: str) -> Path:
        return self.root / self.get(ref).path

    def id_for(self, path: str | Path) -> str | None:
        """The id of the run or artifact at `path`, if it is one inside this store."""
        try:
            rel = Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return None
        for refresh in (False, True):
            for e in self.index(refresh=refresh).entries.values():
                if e.path == rel:
                    return e.id
        return None

    def ref_for(self, path: str | Path, *, artifact_only: bool = True) -> str:
        """An id for `path` when it is a sealed artifact (or a run) in this store, else the path itself: what a
        generator writes into a candidate patch."""
        rid = self.id_for(path)
        if rid and (rid.startswith("artifact:") or not artifact_only):
            return rid
        return str(Path(path).resolve())

    def referrers(self, ref: str) -> list[str]:
        e = self.get(ref)
        return list(self.index().referrers.get(e.id, []))

    # ---------------------------------------------------------------- removal
    def tombstones(self) -> list[dict[str, Any]]:
        f = self.meta / "tombstones.jsonl"
        if not f.exists():
            return []
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

    def tombstone(self, ref: str) -> dict[str, Any] | None:
        m = REF.match(ref)
        for t in reversed(self.tombstones()):
            if m is None or t["kind"] != m["kind"] or t["name"] != m["name"]:
                continue
            if m["digest"] is None or (t.get("digest") or "").startswith(m["digest"]):
                return t
        return None

    def remove(self, ref: str, *, force: bool = False, reason: str = "") -> dict[str, Any]:
        """Delete a run or artifact. Refused while anything refers to it (an artifact built from it, a campaign's
        archive or ledger) unless forced; either way a tombstone records what was removed and why."""
        e = self.get(ref)
        refs = self.index(refresh=True).referrers.get(e.id, [])
        if refs and not force:
            raise StoreError(f"{e.id} is referred to by {refs}; remove those first or pass force (with a reason)")
        stone = {"id": e.id, "kind": e.kind, "name": e.name, "digest": e.digest, "path": e.path,
                 "removed_at": time.time(), "reason": reason, "referrers": refs, "forced": bool(refs)}
        self.meta.mkdir(parents=True, exist_ok=True)
        with open(self.meta / "tombstones.jsonl", "a") as f:
            f.write(json.dumps(stone) + "\n")
        shutil.rmtree(self.root / e.path)
        self.rebuild()
        return stone

    # ---------------------------------------------------------------- references in configs
    def resolve(self, obj: Any) -> Any:
        """`obj` with every store id replaced by this machine's absolute path."""
        if isinstance(obj, str):
            return str(self.path(obj)) if is_ref(obj) else obj
        if isinstance(obj, dict):
            return {k: self.resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.resolve(v) for v in obj)
        return obj

    def pin(self, obj: Any) -> dict[str, dict[str, Any]]:
        """The ids `obj` references, each with its full id and current path (for a run's manifest)."""
        out: dict[str, dict[str, Any]] = {}
        for s in _strings(obj):
            if is_ref(s) and s not in out:
                e = self.get(s)
                out[s] = {"id": e.id, "path": str(self.root / e.path)}
        return out

    # ---------------------------------------------------------------- sync
    def closure(self, ids: Iterable[str]) -> list[Entry]:
        """The entries and, recursively, what they were built from."""
        idx = self.index()
        seen: dict[str, Entry] = {}
        todo = [self.get(i).id for i in ids]
        while todo:
            i = todo.pop()
            if i in seen or i not in idx.entries:
                continue
            seen[i] = idx.entries[i]
            todo += idx.entries[i].refs
        return list(seen.values())


def has_refs(obj: Any) -> bool:
    return any(is_ref(s) for s in _strings(obj))


def resolve_refs(obj: Any, store: Store | None = None) -> Any:
    """Replace store ids in a component spec with paths (a no-op without ids, so no store is opened)."""
    return (store or Store()).resolve(obj) if has_refs(obj) else obj


# ---------------------------------------------------------------- sync between machines

class Location:
    """This machine's store, or another one: `PATH` or `HOST:PATH` (reached with ssh and rsync)."""

    def __init__(self, spec: str | Path, *, command: str = "hylo") -> None:
        s = str(spec)
        host, sep, rest = s.partition(":")
        remote = bool(sep) and "/" not in host and not Path(s).exists()
        self.host, self.path, self.command = (host, rest, command) if remote else (None, str(Path(s).expanduser().resolve()), command)

    def __str__(self) -> str:
        return f"{self.host}:{self.path}" if self.host else self.path

    def src(self, rel: str) -> str:
        return f"{self}/./{rel}"

    def entries(self) -> dict[str, Entry]:
        if self.host is None:
            return Store(self.path).index(refresh=True).entries
        out = subprocess.run(["ssh", self.host, self.command, "store", "--root", self.path, "ls", "--json"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise StoreError(f"cannot list the store at {self}: {out.stderr.strip()}")
        return {e["id"]: Entry.model_validate(e) for e in json.loads(out.stdout)}

    def reindex(self) -> None:
        if self.host is None:
            Store(self.path).rebuild()
        else:
            subprocess.run(["ssh", self.host, self.command, "store", "--root", self.path, "index"], capture_output=True)


def plan_sync(src: dict[str, Entry], dst: dict[str, Entry], ids: Sequence[str] | None,
              with_sources: bool) -> dict[str, list[Any]]:
    """What a sync copies: selected entries not on the other side (runs and mutable artifacts also when present,
    to bring them up to date), and conflicts (the same path holding something else)."""
    sel_store = _MemoryStore(src)
    chosen = sel_store.closure(ids) if (ids and with_sources) else ([sel_store.get(i) for i in ids] if ids else list(src.values()))
    dst_paths = {e.path: e for e in dst.values()}
    copy, skip, conflicts = [], [], []
    for e in sorted(chosen, key=lambda e: e.path):
        other = dst_paths.get(e.path)
        if other is not None and other.id != e.id:
            conflicts.append({"path": e.path, "here": e.id, "there": other.id})
        elif e.id in dst and e.kind == "artifact" and not e.detail.get("mutable"):
            skip.append(e.id)
        else:
            copy.append(e)
    return {"copy": copy, "skip": skip, "conflicts": conflicts}


class _MemoryStore(Store):
    def __init__(self, entries: dict[str, Entry]) -> None:
        self.root = Path("/nonexistent-store")
        self._index = Index(entries=entries)

    def index(self, *, refresh: bool = False) -> Index:
        return self._index

    def get(self, ref: str) -> Entry:
        m = REF.match(ref)
        if m is None:
            raise StoreError(f"{ref!r} is not a store id")
        return self._match(m, self._index.entries)


def sync(local: Store, other: str | Location, *, direction: Literal["push", "pull"], ids: Sequence[str] | None = None,
         with_sources: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Copy runs and artifacts between this machine's store and another (push: here -> there, pull: there -> here)."""
    there = other if isinstance(other, Location) else Location(other)
    here = Location(local.root)
    src, dst = (here, there) if direction == "push" else (there, here)
    plan = plan_sync(src.entries(), dst.entries(), ids, with_sources)
    report = {"from": str(src), "to": str(dst), "copied": [e.id for e in plan["copy"]], "skipped": plan["skip"],
              "conflicts": plan["conflicts"]}
    if plan["conflicts"]:
        raise StoreError(f"sync refused, {len(plan['conflicts'])} conflict(s): {plan['conflicts']}")
    if dry_run or not plan["copy"]:
        return {**report, "dry_run": dry_run}
    if dst.host is None:
        Path(dst.path).mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        cmd = ["rsync", "-a", "--relative", *[src.src(e.path) for e in plan["copy"]], f"{dst}/"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise StoreError(f"rsync failed ({r.returncode}): {r.stderr.strip()}")
    elif src.host is None and dst.host is None:
        for e in plan["copy"]:
            shutil.copytree(Path(src.path) / e.path, Path(dst.path) / e.path, dirs_exist_ok=True)
    else:
        raise StoreError("rsync is needed to sync with another machine")
    dst.reindex()
    return report


# ---------------------------------------------------------------- command line

def _print_entries(entries: list[Entry], as_json: bool) -> None:
    if as_json:
        print(json.dumps([e.model_dump(mode="json") for e in entries]))
        return
    for e in sorted(entries, key=lambda e: (e.kind, e.path)):
        extra = e.detail.get("kind") if e.kind == "artifact" else e.detail.get("split")
        print(f"{e.id:<72} {extra or '-':<16} {e.path}")


def main(argv: Sequence[str] | None = None) -> int:
    """the run and artifact store: ids, references, deletion protection, sync between machines"""
    import argparse

    ap = argparse.ArgumentParser(prog="hylo store", description=main.__doc__)
    ap.add_argument("--root", help="the store directory (default: this machine's profile, $HYLO_STORE, ./runs)")
    sub = ap.add_subparsers(dest="action", required=True)
    it = sub.add_parser("init", help="make PATH this machine's store (recorded in the profile) and index it")
    it.add_argument("path")
    sub.add_parser("where", help="print the store directory")
    sub.add_parser("index", help="rebuild the index")
    ls = sub.add_parser("ls", help="list runs and artifacts")
    ls.add_argument("--kind", choices=("run", "artifact"))
    ls.add_argument("--name")
    ls.add_argument("--json", action="store_true")
    sh = sub.add_parser("show", help="an entry, its path and what refers to it")
    sh.add_argument("ref")
    rf = sub.add_parser("ref", help="the id of a run or artifact directory")
    rf.add_argument("path")
    rm = sub.add_parser("rm", help="delete a run or artifact (refused while something refers to it)")
    rm.add_argument("ref")
    rm.add_argument("--force", action="store_true")
    rm.add_argument("--reason", default="")
    for direction, what in (("push", "copy runs and artifacts from here to another store"),
                            ("pull", "copy runs and artifacts from another store to here")):
        sy = sub.add_parser(direction, help=what)
        sy.add_argument("location", help="PATH or HOST:PATH")
        sy.add_argument("ids", nargs="*", help="what to copy (default: everything)")
        sy.add_argument("--with-sources", action="store_true", help="also what the selected artifacts were built from")
        sy.add_argument("--dry-run", action="store_true")
        sy.add_argument("--remote-command", default="hylo", help="the hylo command on the other machine")
    ad = sub.add_parser("adopt", help="seal a directory built before artifacts had manifests (RFC-0004)")
    ad.add_argument("path")
    ad.add_argument("--kind", required=True)
    ad.add_argument("--role", required=True, choices=("behaviour", "instrument", "proposal", "data"))
    ad.add_argument("--name", default="")
    ad.add_argument("--source", action="append", default=[], metavar="RUN_DIR|run:ID",
                    help="a run it was built from (repeat); without any, the provenance is recorded as unknown")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.action == "init":
        from .profile import load_profile, save_profile

        root = Path(a.path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        save_profile(load_profile().model_copy(update={"store": str(root)}))
        idx = Store(root).rebuild()
        print(json.dumps({"store": str(root), "runs": sum(e.kind == "run" for e in idx.entries.values()),
                          "artifacts": sum(e.kind == "artifact" for e in idx.entries.values())}))
        return 0
    store = Store(a.root)
    try:
        if a.action == "where":
            print(store.root)
        elif a.action == "index":
            idx = store.rebuild()
            print(json.dumps({"store": str(store.root), "entries": len(idx.entries),
                              "referred_to": len(idx.referrers)}))
        elif a.action == "ls":
            es = [e for e in store.entries(a.kind, refresh=True) if a.name in (None, e.name)]
            _print_entries(es, a.json)
        elif a.action == "show":
            e = store.get(a.ref)
            print(json.dumps({**e.model_dump(mode="json"), "abs_path": str(store.root / e.path),
                              "referrers": store.referrers(e.id)}, indent=2))
        elif a.action == "ref":
            rid = store.id_for(a.path)
            if rid is None:
                print(f"{a.path} is not a run or sealed artifact in {store.root}", file=sys.stderr)
                return 1
            print(rid)
        elif a.action == "rm":
            print(json.dumps(store.remove(a.ref, force=a.force, reason=a.reason)))
        elif a.action in ("push", "pull"):
            rep = sync(store, Location(a.location, command=a.remote_command), direction=a.action,
                       ids=a.ids or None, with_sources=a.with_sources, dry_run=a.dry_run)
            print(json.dumps(rep, indent=2))
        elif a.action == "adopt":
            print(json.dumps(adopt(a.path, kind=a.kind, role=a.role, name=a.name, sources=a.source, store=store)))
    except StoreError as exc:
        print(f"hylo store: {exc}", file=sys.stderr)
        return 1
    return 0


def adopt(path: str | Path, *, kind: str, role: str, name: str = "", sources: Sequence[str] = (),
          store: Store | None = None) -> dict[str, Any]:
    """Seal a directory that predates manifests. Its sources are what the caller states (run directories or run
    ids); none means unknown provenance, which is recorded, and which the RSI gate refuses for behaviour artifacts."""
    from .artifact import Provenance, SourceRef, run_split, write_manifest

    store = store or Store()
    refs = []
    for s in sources:
        rd = store.path(s) if is_ref(s) else Path(s)
        log = _read_json(rd / "evallog.json") or {}
        refs.append(SourceRef(kind="run", id=str(log.get("run_id") or rd.name), split=run_split(rd), path=str(rd),
                              n_episodes=len(log.get("samples", [])) or None))
    prov = Provenance(builder="hylo.core.store.adopt@0.1", sources=refs,
                      params={"adopted": True, **({} if refs else {"unknown_provenance": True})})
    m = write_manifest(path, kind=kind, role=role, provenance=prov, name=name)    # type: ignore[arg-type]
    rid = store.id_for(path)
    return {"id": rid or f"artifact:{m.name}@{m.digest}", "digest": m.digest, "sources": [r.id for r in refs],
            "unknown_provenance": not refs}


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Entry", "Index", "Location", "STORE_SCHEMA", "Store", "StoreError", "adopt", "default_root", "has_refs",
           "is_ref", "plan_sync", "resolve_refs", "sync"]
