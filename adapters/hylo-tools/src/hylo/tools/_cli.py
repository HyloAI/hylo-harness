"""Shared CLI plumbing for `hylo tool <name> ...`: common options and printing a result."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from .runner import ToolError, ToolResult, _json_default


def parser(name: str, doc: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"hylo tool {name}", description=(doc or "").strip().split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=doc)
    ap.add_argument("--out", help="copy the output files into this directory")
    ap.add_argument("--threads", type=int, help="CPU threads (default $HYLO_TOOLS_THREADS or 32)")
    ap.add_argument("--workdir", help="work directory (default: a new one under the tools cache)")
    ap.add_argument("--verbose", action="store_true", help="show the worker's output")
    return ap


def run_kw(a: argparse.Namespace) -> dict[str, Any]:
    return {"threads": a.threads, "workdir": a.workdir, "verbose": a.verbose}


def finish(res: ToolResult, out: str | None) -> int:
    body = res.to_json()
    if out:
        dst = Path(out)
        dst.mkdir(parents=True, exist_ok=True)
        body["copied"] = {}
        for k, p in res.files.items():
            if p.exists():
                shutil.copy2(p, dst / p.name)
                body["copied"][k] = str(dst / p.name)
    json.dump(body, sys.stdout, indent=2, default=_json_default)
    sys.stdout.write("\n")
    return 0


def guard(fn: Callable[[], int]) -> int:
    try:
        return fn()
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.blocked:
            print(f"blocked: {exc.blocked}", file=sys.stderr)
        return 1
