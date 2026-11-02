"""Model-written scripts run in a directory, under a guard, with tool calls served over a pipe (free mode of the
motion studio; nothing here is specific to motion).

- check_source: the static check before anything runs: imports from an allow-list or the run directory's lib/
  (checked too), no eval / exec / getattr, no dunder or private attributes, no async code.
- ScriptRunner.run(code): saves the script as scripts/turnNN.py and runs it with motion_freerun.py (python -I,
  cwd = the run directory, resource limits, an audit hook refusing files outside the directory, processes,
  sockets and ctypes). The script talks to `handler(op, args)` in this process through two pipes (one JSON line
  per request and per answer); PermissionError from the handler is reported as a refusal. The time limit
  excludes the time spent in handler calls. Returns the code-agents BlockResult, so a CodeLoop can drive it.
- inside: resolve a path given by a script, refusing anything outside the run directory.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sandbox import BlockResult, truncate

RUNNER = Path(__file__).with_name("motion_freerun.py")
ALLOWED_IMPORTS = {"numpy", "scipy", "math", "json", "statistics", "itertools", "functools", "collections",
                   "dataclasses", "typing", "random", "re", "copy", "pathlib", "studio", "ops", "bisect", "heapq",
                   "fractions", "decimal", "string", "textwrap", "operator", "enum", "time"}
DENIED_NAMES = {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
                "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit", "os", "sys", "subprocess",
                "socket", "shutil", "ctypes", "importlib", "builtins", "__builtins__", "__loader__", "__spec__"}
DENIED_ATTRS = {"os", "sys", "subprocess", "system", "popen", "spawn", "fork", "builtins", "ctypes", "ctypeslib",
                "f2py", "distutils", "importlib", "modules", "load_library", "dlopen", "loader"}
ALLOWED_DUNDERS = {"__name__", "__file__", "__doc__", "__init__", "__len__", "__repr__", "__str__", "__iter__",
                   "__getitem__", "__call__", "__eq__", "__lt__", "__main__"}


class ScriptRejected(ValueError):
    """The script failed the static check; nothing ran."""


def check_source(source: str, root: Path, *, depth: int = 0) -> None:
    """Raise ScriptRejected listing what the script may not do. Local modules it imports (root/lib/NAME.py) are
    checked as well."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ScriptRejected(f"SyntaxError: {exc.msg} (line {exc.lineno})") from None
    bad: list[str] = []

    def module_ok(name: str) -> bool:
        top = name.split(".")[0]
        if top in ALLOWED_IMPORTS:
            return True
        local = root / "lib" / f"{top}.py"
        if local.is_file() and depth < 3:
            try:
                check_source(local.read_text(), root, depth=depth + 1)
            except ScriptRejected as exc:
                bad.append(f"lib/{top}.py: {exc}")
            return True
        return False
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            bad += [f"line {line}: import {a.name} is not allowed" for a in node.names if not module_ok(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level or not module_ok(node.module or ""):
                bad.append(f"line {line}: from {node.module} import ... is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in DENIED_NAMES or (node.id.startswith("__") and node.id not in ALLOWED_DUNDERS):
                bad.append(f"line {line}: name {node.id!r} is not available")
        elif isinstance(node, ast.Attribute):
            if node.attr in DENIED_ATTRS or (node.attr.startswith("_") and node.attr not in ALLOWED_DUNDERS):
                bad.append(f"line {line}: attribute .{node.attr} is not available")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            bad.append(f"line {line}: async code is not allowed")
    if bad:
        raise ScriptRejected("rejected: " + "; ".join(bad[:8]) + (" ..." if len(bad) > 8 else "")
                             + f" (imports allowed: {', '.join(sorted(ALLOWED_IMPORTS))}, and your own modules in lib/)")


def inside(root: Path, path: Any, *, must_exist: bool = True) -> Path:
    """`path` (relative to root, or absolute) resolved; PermissionError if it leaves root."""
    if not isinstance(path, str) or not path:
        raise PermissionError(f"refused: {path!r} is not a path")
    p = Path(path)
    p = (p if p.is_absolute() else root / p).resolve()
    if p != root and root not in p.parents:
        raise PermissionError(f"refused: {path} is outside the run directory")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"{path} does not exist in the run directory")
    return p


def short_traceback(tb: str, root: Path) -> str:
    """A script's traceback with only the frames of files in the run directory (scripts/, lib/)."""
    lines = tb.strip().splitlines()
    if not lines or not lines[0].startswith("Traceback"):
        return tb.strip()
    out, keep = [lines[0]], False
    body = lines[1:]
    for i, ln in enumerate(body):
        if ln.startswith("  File "):
            keep = str(root) in ln
            if keep:
                out.append(ln.replace(str(root) + "/", ""))
        elif ln.startswith("    "):
            if keep and not set(ln.strip()) <= set("^~ "):
                out.append(ln)
        elif not ln.startswith(" "):
            out += body[i:]
            break
    return "\n".join(out)


def plain(v: Any) -> Any:
    """json.dumps default for tool results: small arrays as lists, numpy scalars, paths; anything else as repr."""
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else f"array{tuple(v.shape)}"
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return repr(v)[:200]


class ScriptRunner:
    """Runs scripts in `root` (the CodeLoop's sandbox interface: run(code) -> BlockResult) and serves their
    requests with `handler(op, args)`; every script and request is passed to `emit` for the log."""

    def __init__(self, root: Path, handler: Callable[[str, dict], Any], *, emit: Callable[[dict], None],
                 timeout_s: float = 900.0, python: str | None = None, threads: int = 8) -> None:
        self.root = Path(root).resolve()
        self.handler, self.emit, self.timeout_s = handler, emit, timeout_s
        self.python = python or sys.executable
        self.threads = threads
        self.scripts = 0
        self.refusals = 0
        self._busy_s = 0.0                              # time spent in finished handler calls
        self._busy_since: float | None = None           # start of the handler call in progress
        self._lock = threading.Lock()

    def _busy(self) -> float:
        with self._lock:
            run = time.monotonic() - self._busy_since if self._busy_since is not None else 0.0
            return self._busy_s + run

    def _serve(self, rf, wf) -> None:
        for line in rf:
            t0 = time.perf_counter()
            with self._lock:
                self._busy_since = time.monotonic()
            rec: dict[str, Any] = {"event": "request"}
            try:
                req = json.loads(line)
                op, args = str(req.get("op")), dict(req.get("args") or {})
                rec.update(op=op, args={k: repr(v)[:200] for k, v in args.items()})
                res = {"result": self.handler(op, args)}
            except PermissionError as exc:
                self.refusals += 1
                res = {"error": str(exc)}
                rec["refused"] = str(exc)[:500]
            except Exception as exc:  # noqa: BLE001 - the script sees the error
                res = {"error": f"{type(exc).__name__}: {exc}"[:1500]}
                rec["error"] = res["error"][:500]
            dt = time.perf_counter() - t0
            with self._lock:
                self._busy_s += time.monotonic() - self._busy_since
                self._busy_since = None
            rec["seconds"] = round(dt, 3)
            self.emit(rec)
            try:
                wf.write(json.dumps(res, default=plain) + "\n")
                wf.flush()
            except (BrokenPipeError, ValueError):
                return

    def run(self, code: str) -> BlockResult:
        """Check, save and run one script; its output, error and time (handler time excluded)."""
        self.scripts += 1
        sdir = self.root / "scripts"
        sdir.mkdir(exist_ok=True)
        path = sdir / f"turn{self.scripts:02d}.py"
        path.write_text(code)
        rec: dict[str, Any] = {"event": "script", "script": str(path.relative_to(self.root)), "chars": len(code)}
        try:
            check_source(code, self.root)
        except ScriptRejected as exc:
            rec["rejected"] = str(exc)[:800]
            self.emit(rec)
            return BlockResult(error=str(exc), rejected=True)
        log = sdir / f"turn{self.scripts:02d}.out"
        req_r, req_w = os.pipe()
        resp_r, resp_w = os.pipe()
        tmp = self.root / "work" / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        n = str(self.threads)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "TMPDIR": str(tmp), "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n,
               "HYLO_STUDIO_RPC": f"{req_w},{resp_r}", "HYLO_SCRIPT_CPU_S": str(int(self.timeout_s * 8))}
        with open(log, "w") as lf:
            proc = subprocess.Popen([self.python, "-I", str(RUNNER), str(path), str(self.root)], cwd=self.root,
                                    env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                    pass_fds=(req_w, resp_r))
        os.close(req_w)
        os.close(resp_r)
        rf, wf = os.fdopen(req_r, "r"), os.fdopen(resp_w, "w")
        busy0 = self._busy()
        server = threading.Thread(target=self._serve, args=(rf, wf), daemon=True, name="hylo-script-rpc")
        server.start()
        t0 = time.monotonic()
        timed_out = False
        while proc.poll() is None:
            time.sleep(0.05)
            if time.monotonic() - t0 - (self._busy() - busy0) > self.timeout_s:
                proc.kill()
                timed_out = True
                break
        proc.wait()
        server.join(timeout=5)
        for f in (rf, wf):
            try:
                f.close()
            except OSError:
                pass
        busy = self._busy() - busy0
        secs = time.monotonic() - t0
        output = log.read_text(errors="replace")
        err = None
        if timed_out:
            err = f"TimeoutError: the script ran longer than {self.timeout_s:g} s (tool time excluded) and was stopped"
        elif proc.returncode:
            k = output.rfind("Traceback")
            err = truncate(short_traceback(output[k:] if k >= 0 else output[-1500:], self.root), 2500) \
                or f"exit status {proc.returncode}"
            output = output[:k] if k >= 0 else output
        rec.update(exit=proc.returncode, seconds=round(secs, 2), tool_seconds=round(busy, 2), timed_out=timed_out,
                   output_tail=output[-600:], error=(err or "")[:800] or None)
        self.emit(rec)
        return BlockResult(output=truncate(output, 6000), error=err, seconds=round(secs - busy, 2),
                           timed_out=timed_out)


__all__ = ["ALLOWED_IMPORTS", "RUNNER", "ScriptRejected", "ScriptRunner", "check_source", "inside", "plain",
           "short_traceback"]
