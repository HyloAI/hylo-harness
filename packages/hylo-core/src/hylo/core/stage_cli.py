"""`hylo stage`: stages (hylo.stage/v0.1) on disk, with their revisions.

    hylo stage list                       stages under the root (default runs/stages, or $HYLO_STAGES)
    hylo stage new NAME [--from FILE]     create a stage (empty, or from a stage JSON file)
    hylo stage show NAME [--rev N]        print a stage (the newest revision, or revision N)
    hylo stage patch NAME PATCH.json      apply a patch (its expected_revision must be the current revision)
    hylo stage history NAME               one line per revision
    hylo stage restore NAME N             commit revision N again as the newest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .stage import Patch, Stage, StageConflict, StageRefused, StageStore


def default_root() -> Path:
    return Path(os.environ.get("HYLO_STAGES") or "runs/stages")


def main(argv: Sequence[str] | None = None) -> int:
    """stages (hylo.stage/v0.1): create, show, patch, history, restore"""
    ap = argparse.ArgumentParser(prog="hylo stage", description="stages and their revisions")
    ap.add_argument("--root", default=None, help="stage root (default runs/stages or $HYLO_STAGES)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    n = sub.add_parser("new")
    n.add_argument("name")
    n.add_argument("--from", dest="src")
    s = sub.add_parser("show")
    s.add_argument("name")
    s.add_argument("--rev", type=int)
    p = sub.add_parser("patch")
    p.add_argument("name")
    p.add_argument("patch")
    h = sub.add_parser("history")
    h.add_argument("name")
    r = sub.add_parser("restore")
    r.add_argument("name")
    r.add_argument("revision", type=int)
    a = ap.parse_args(argv)
    store = StageStore(a.root or default_root())
    try:
        if a.cmd == "list":
            for name in store.names():
                st = store.load(name)
                print(f"{name}  revision {st.revision}  {len(st.entities)} entities")
        elif a.cmd == "new":
            data = json.loads(Path(a.src).read_text()) if a.src else {}
            st = store.create(Stage.model_validate({**data, "name": a.name, "revision": 0}))
            print(f"created {st.name} at revision 0 ({len(st.entities)} entities) in {store.root}")
        elif a.cmd == "show":
            print(store.load(a.name, a.rev).model_dump_json(indent=1))
        elif a.cmd == "patch":
            st = store.commit(a.name, Patch.model_validate_json(Path(a.patch).read_text()))
            print(f"{a.name} is at revision {st.revision}")
        elif a.cmd == "history":
            for line in store.history(a.name):
                print(json.dumps(line))
        elif a.cmd == "restore":
            st = store.restore(a.name, a.revision)
            print(f"{a.name} is at revision {st.revision} (= revision {a.revision})")
    except (StageConflict, StageRefused, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
