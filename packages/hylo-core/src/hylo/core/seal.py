"""Run evidence: hash-chain checks, run seals and a scan of what models were shown (RFC-0007 part 5).

    hylo seal RUN_DIR            verify every episode's hash chain, then write RUN_DIR/seal.json (every file with its
                                 size and sha256, the chain heads, the inventory's own digest) and make the files
                                 read-only; refused when a chain is broken or a scan finds privileged keys
    hylo seal --verify RUN_DIR   recompute the inventory and compare it with seal.json (exit 1 on any difference)
    hylo seal --chain RUN_DIR    only check the hash chains

A sealed run cannot change silently: a changed, added or removed file shows up in `--verify`. The scan looks at the
events that carry what a model saw or wrote (agent turns and tool results) for keys that only privileged state
has (simulator object poses, segmentation ids given as ground truth, success predicates).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .trace import line_hash

SEAL_SCHEMA = "hylo.seal/v0.1"
SEAL = "seal.json"
MODEL_VISIBLE = ("agent.tool.turn", "agent.tool.result", "agent.tool.start", "agent.code.turn")
PRIVILEGED = re.compile(r"\b(privileged|ground_truth|object_poses|success_predicate|sim_state|qpos_objects)\b")


def verify_chain(path: Path) -> dict[str, Any]:
    """One episode file: {"lines", "chained" (lines with a hash), "ok", "broken_at" (1-based line or None), "head"}."""
    prev, n, chained = "", 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return {"lines": n, "chained": chained, "ok": False, "broken_at": n, "head": prev,
                        "why": "not JSON"}
            if "h" not in rec:
                if chained:
                    return {"lines": n, "chained": chained, "ok": False, "broken_at": n, "head": prev,
                            "why": "an unchained line after chained ones"}
                continue
            chained += 1
            if rec.get("prev", "") != prev:
                return {"lines": n, "chained": chained, "ok": False, "broken_at": n, "head": prev,
                        "why": "prev does not match the line before (a line was removed or reordered)"}
            if line_hash(rec) != rec["h"]:
                return {"lines": n, "chained": chained, "ok": False, "broken_at": n, "head": prev,
                        "why": "the line's content does not match its hash (it was changed)"}
            prev = rec["h"]
    return {"lines": n, "chained": chained, "ok": True, "broken_at": None, "head": prev}


def chains(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(run_dir)
    return {p.stem: verify_chain(p) for p in sorted((root / "episodes").glob("*.jsonl"))}


def scan_visible(run_dir: str | Path) -> list[dict[str, Any]]:
    """Privileged-looking keys in model-visible events: [{"episode", "line", "type", "key"}]."""
    out = []
    for p in sorted((Path(run_dir) / "episodes").glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not any(f'"type":"{t}"' in line[:60] for t in MODEL_VISIBLE):
                    continue
                m = PRIVILEGED.search(line)
                if m:
                    out.append({"episode": p.stem, "line": i, "type": json.loads(line)["type"], "key": m.group(1)})
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != SEAL and not p.name.startswith(".") and p.suffix != ".lock":
            files.append({"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": _sha(p)})
    return files


def _digest(files: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SealRefused(RuntimeError):
    pass


def seal(run_dir: str | Path, *, read_only: bool = True, allow_unchained: bool = True,
         allow_privileged: bool = False) -> dict[str, Any]:
    root = Path(run_dir)
    if (root / SEAL).exists():
        raise SealRefused(f"{root} is already sealed; `hylo seal --verify` checks it")
    ch = chains(root)
    broken = {k: v for k, v in ch.items() if not v["ok"]}
    if broken:
        raise SealRefused(f"broken hash chains: " + "; ".join(f"{k} line {v['broken_at']}: {v['why']}"
                                                              for k, v in broken.items()))
    if not allow_unchained and any(v["chained"] < v["lines"] for v in ch.values()):
        raise SealRefused("some episodes were written without a hash chain")
    leaks = scan_visible(root)
    if leaks and not allow_privileged:
        raise SealRefused(f"privileged keys in model-visible events: {leaks[:5]}")
    files = inventory(root)
    doc = {"schema": SEAL_SCHEMA, "sealed_at": round(time.time(), 3), "files": files,
           "inventory_sha256": _digest(files),
           "chains": {k: {"head": v["head"], "lines": v["lines"], "chained": v["chained"]} for k, v in ch.items()},
           "privileged_scan": leaks}
    (root / SEAL).write_text(json.dumps(doc, indent=1))
    if read_only:
        for f in files + [{"path": SEAL}]:
            p = root / f["path"]
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return doc


def verify_seal(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not (root / SEAL).is_file():
        return {"ok": False, "why": "not sealed"}
    doc = json.loads((root / SEAL).read_text())
    now = {f["path"]: f for f in inventory(root)}
    then = {f["path"]: f for f in doc["files"]}
    changed = sorted(p for p in then.keys() & now.keys() if then[p]["sha256"] != now[p]["sha256"])
    added, removed = sorted(now.keys() - then.keys()), sorted(then.keys() - now.keys())
    ok = not (changed or added or removed) and _digest(doc["files"]) == doc["inventory_sha256"]
    return {"ok": ok, "changed": changed, "added": added, "removed": removed, "files": len(then)}


def main(argv: Sequence[str] | None = None) -> int:
    """seal a run directory (file inventory, hash chains) or verify its seal"""
    ap = argparse.ArgumentParser(prog="hylo seal", description="seal a run directory or verify its seal")
    ap.add_argument("run")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true", help="compare the run with its seal")
    g.add_argument("--chain", action="store_true", help="only check the episodes' hash chains")
    ap.add_argument("--writable", action="store_true", help="do not make the files read-only")
    ap.add_argument("--allow-privileged", action="store_true", help="seal even when the scan finds privileged keys")
    a = ap.parse_args(argv)
    if a.verify:
        r = verify_seal(a.run)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    if a.chain:
        ch = chains(a.run)
        print(json.dumps(ch, indent=1))
        return 0 if all(v["ok"] for v in ch.values()) else 1
    try:
        doc = seal(a.run, read_only=not a.writable, allow_privileged=a.allow_privileged)
    except SealRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"sealed {a.run}: {len(doc['files'])} files, inventory {doc['inventory_sha256'][:16]}")
    return 0


__all__ = ["SealRefused", "chains", "inventory", "main", "scan_visible", "seal", "verify_chain", "verify_seal"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
