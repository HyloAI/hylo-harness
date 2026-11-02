"""A restricted namespace for model-written Python (RFC-0006: "programs are data until gated").

Every block is parsed and checked against an allow-list before it runs:
  - imports: only `math` and `numpy` (and numpy.linalg / random / fft)
  - no names or attributes that start with an underscore (no dunders, no private toolkit state)
  - no open / exec / eval / compile / globals / getattr / ... (they are not in the builtins either)
  - no frame, generator or code-object internals, no numpy file I/O
  - no class definitions, no async code
Builtins are a small allow-list; `print` writes to a buffer the agent returns to the model. The namespace persists
between blocks (one per episode or research session). Each block has a wall-clock limit that excludes the time the
robot spends executing (`paused()`, used around Executor.run): on expiry an exception is raised inside the running
thread (CPython async exception), so a busy loop stops; a call stuck inside C code stops when it returns.

This is a guard against mistakes and casual escapes by a model, not an OS-level jail: run untrusted models in a
container as well. Memory use is not limited.
"""

from __future__ import annotations

import ast
import builtins
import ctypes
import math
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

MAX_SOURCE_CHARS = 40_000

ALLOWED_IMPORTS = {"math", "numpy", "numpy.linalg", "numpy.random", "numpy.fft"}
# names a session provides that models often try to import ("from skills import stack"): the import resolves to the
# namespace's own object, or fails with a clear message when the session has none
NAMESPACE_IMPORTS = {"skills"}

DENIED_NAMES = {
    "open", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr", "hasattr",
    "input", "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super", "dir", "classmethod",
    "staticmethod", "property", "__import__", "__builtins__", "importlib", "os", "sys", "subprocess",
}

DENIED_ATTRS = {
    # frames, generators, coroutines, code objects: the way back to real builtins
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame", "ag_code", "f_globals",
    "f_locals", "f_builtins", "f_back", "f_code", "tb_frame", "tb_next", "co_code", "co_consts", "func_globals",
    "func_code", "mro",
    # numpy (and array) file I/O, raw memory and compilers
    "tofile", "dump", "dumps", "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "fromregex", "memmap", "ctypes", "ctypeslib", "lib", "f2py", "distutils", "testing",
    "DataSource", "core", "system", "popen",
}

SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hash", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "ord",
    "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "IndexError", "KeyError", "LookupError", "NotImplementedError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)


class CodeRejected(ValueError):
    """The source failed validation; nothing ran."""


class ProgramTimeout(BaseException):
    """Raised inside a block that ran past its wall-clock limit (BaseException: `except Exception` cannot eat it)."""


class ProgramCancelled(BaseException):
    """Raised inside a block whose episode or session is over."""


class _SafeModule:
    """A read-only view of a module that refuses private and denied attributes."""

    def __init__(self, module: Any, name: str) -> None:
        object.__setattr__(self, "_m", module)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_") or attr in DENIED_ATTRS:
            raise AttributeError(f"{self._name}.{attr} is not available in programs")
        value = getattr(self._m, attr)
        if attr in ("linalg", "random", "fft"):
            return _SafeModule(value, f"{self._name}.{attr}")
        return value

    def __setattr__(self, attr: str, value: Any) -> None:
        raise AttributeError(f"{self._name} is read-only")

    def __repr__(self) -> str:
        return f"<module {self._name}>"


SAFE_NUMPY = _SafeModule(np, "numpy")
SAFE_MATH = _SafeModule(math, "math")


def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if not level and name not in ALLOWED_IMPORTS and (name.startswith("numpy.") or name in sys.modules):
        # Not a program's import (validation allows only ALLOWED_IMPORTS): numpy's C code imports its private
        # modules lazily through the builtins of the topmost Python frame, which is the program's.
        return builtins.__import__(name, globals, locals, fromlist, level)
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import {name!r} is not allowed (only math and numpy)")
    if name == "math":
        return SAFE_MATH
    if fromlist and name != "numpy":
        return getattr(SAFE_NUMPY, name.split(".", 1)[1])
    return SAFE_NUMPY


# ---------------------------------------------------------------- validation

class CodeChecker(ast.NodeVisitor):
    """The AST allow-list check behind validate(); subclasses widen or narrow it (scene programs)."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def _bad(self, node: ast.AST, msg: str) -> None:
        self.problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name not in ALLOWED_IMPORTS | NAMESPACE_IMPORTS:
                self._bad(node, f"import {a.name} is not allowed (only math and numpy)")
            if a.asname and a.asname.startswith("_"):
                self._bad(node, f"name {a.asname!r} starts with an underscore")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module not in ALLOWED_IMPORTS | NAMESPACE_IMPORTS:
            self._bad(node, f"from {node.module} import ... is not allowed (only math and numpy)")
        for a in node.names:
            if a.name == "*" or a.name.startswith("_") or a.name in DENIED_ATTRS:
                self._bad(node, f"cannot import {a.name!r}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id in DENIED_NAMES:
            self._bad(node, f"name {node.id!r} is not available")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in DENIED_ATTRS:
            self._bad(node, f"attribute .{node.attr} is not available")
        self.generic_visit(node)

    def _def(self, node: Any) -> None:
        if node.name.startswith("_"):
            self._bad(node, f"function name {node.name!r} starts with an underscore")
        for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                  *([node.args.vararg] if node.args.vararg else []), *([node.args.kwarg] if node.args.kwarg else [])]:
            if a.arg.startswith("_"):
                self._bad(node, f"argument {a.arg!r} starts with an underscore")
        self.generic_visit(node)

    visit_FunctionDef = _def

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bad(node, "class definitions are not allowed (use functions, dicts and lists)")

    def _async(self, node: ast.AST) -> None:
        self._bad(node, "async code is not allowed")

    visit_AsyncFunctionDef = visit_AsyncFor = visit_AsyncWith = visit_Await = _async

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg and node.arg.startswith("_"):
            self._bad(node, f"keyword {node.arg!r} starts with an underscore")
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        pass


_Checker = CodeChecker                     # the earlier private name


def validate(source: str, *, filename: str = "<program>") -> ast.Module:
    """Parse and check a program; raises CodeRejected listing every problem found."""
    if len(source) > MAX_SOURCE_CHARS:
        raise CodeRejected(f"program too long ({len(source)} characters > {MAX_SOURCE_CHARS})")
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise CodeRejected(f"SyntaxError: {exc.msg} (line {exc.lineno})") from None
    c = CodeChecker()
    c.visit(tree)
    if c.problems:
        raise CodeRejected("rejected: " + "; ".join(c.problems[:8]) + (" ..." if len(c.problems) > 8 else ""))
    return tree


def defines(tree: ast.Module, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)


# ---------------------------------------------------------------- execution

@dataclass
class BlockResult:
    output: str = ""
    error: str | None = None
    value: Any = None
    seconds: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    rejected: bool = False


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters cut] ...\n{text[-tail:]}"


@dataclass
class _Clock:
    """Wall clock of one block, minus the time spent paused (the robot executing)."""

    start: float = field(default_factory=time.monotonic)
    paused_total: float = 0.0
    paused_since: float | None = None
    depth: int = 0

    def elapsed(self) -> float:
        now = time.monotonic()
        extra = now - self.paused_since if self.paused_since is not None else 0.0
        return now - self.start - self.paused_total - extra


class Sandbox:
    """A persistent restricted namespace. `names` are what programs see (robot, finish, skills, ...)."""

    def __init__(self, names: dict[str, Any] | None = None, *, block_timeout_s: float = 60.0,
                 max_output: int = 6000) -> None:
        self.block_timeout_s, self.max_output = block_timeout_s, max_output
        self._out: list[str] = []
        self._builtins = {n: getattr(builtins, n) for n in SAFE_BUILTINS}
        self._builtins.update(print=self._print, __import__=self._import)
        self._base: dict[str, Any] = {"__builtins__": self._builtins, "__name__": "program", "np": SAFE_NUMPY,
                                      "numpy": SAFE_NUMPY, "math": SAFE_MATH, **(names or {})}
        self.ns: dict[str, Any] = dict(self._base)
        self._sources: dict[str, list[str]] = {}
        self._n = 0
        self._lock = threading.Lock()
        self._clock: _Clock | None = None
        self._thread: int | None = None
        self._fired: type[BaseException] | None = None

    def _import(self, name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if not level and name in NAMESPACE_IMPORTS:
            if name not in self._base:
                raise ImportError(f"there is no {name!r} in this session")
            return self._base[name]
        return _safe_import(name, globals, locals, fromlist, level)

    def add(self, **names: Any) -> None:
        """Make names visible to programs (this namespace, and fresh ones made by `define`)."""
        self.ns.update(names)
        self._base.update(names)

    # -- output
    def _print(self, *args: Any, sep: str = " ", end: str = "\n", **_: Any) -> None:
        self._out.append(sep.join(str(a) for a in args) + end)
        if sum(len(s) for s in self._out) > 20 * self.max_output:          # keep memory bounded on print loops
            text = "".join(self._out)
            self._out = [text[: 5 * self.max_output], text[-5 * self.max_output:]]

    def take_output(self) -> str:
        text, self._out = "".join(self._out), []
        return truncate(text, self.max_output)

    # -- time limit
    @contextmanager
    def paused(self) -> Iterator[None]:
        """Stop the block's clock (and the watchdog) while the robot executes."""
        clock = self._clock
        if clock is None:
            yield
            return
        with self._lock:
            clock.depth += 1
            if clock.depth == 1:
                clock.paused_since = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                clock.depth -= 1
                if clock.depth == 0 and clock.paused_since is not None:
                    clock.paused_total += time.monotonic() - clock.paused_since
                    clock.paused_since = None

    def _inject(self, exc: type[BaseException]) -> bool:
        """Raise `exc` in the thread running the current block (only while it runs and is not paused)."""
        with self._lock:
            if self._thread is None or self._fired is not None:
                return False
            if exc is ProgramTimeout and self._clock is not None and self._clock.depth > 0:
                return False
            self._fired = exc
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._thread), ctypes.py_object(exc))
            return True

    def interrupt(self) -> bool:
        """Stop the running block now (the episode or session ended)."""
        return self._inject(ProgramCancelled)

    def _watch(self, clock: _Clock, limit: float, stop: threading.Event) -> None:
        while not stop.wait(0.02):
            with self._lock:
                paused = clock.depth > 0
            if not paused and clock.elapsed() > limit:
                if self._inject(ProgramTimeout):
                    return

    def _guarded(self, fn: Callable[[], Any], limit: float | None) -> BlockResult:
        res = BlockResult()
        clock = _Clock()
        stop = threading.Event()
        with self._lock:
            self._clock, self._thread, self._fired = clock, threading.get_ident(), None
        watcher = None
        if limit is not None and limit > 0:
            watcher = threading.Thread(target=self._watch, args=(clock, limit, stop), daemon=True,
                                       name="hylo-sandbox-watchdog")
            watcher.start()
        try:
            try:
                res.value = fn()
            finally:
                stop.set()
                with self._lock:
                    fired, self._thread = self._fired, None
                if fired is not None:                       # a raise still pending (it landed in C code): clear it
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(threading.get_ident()), None)
        except ProgramTimeout:
            res.timed_out = True
            res.error = f"TimeoutError: the block ran longer than {limit:g} s (robot motion excluded) and was stopped"
        except ProgramCancelled:
            res.cancelled = True
            res.error = "Cancelled: the episode is over"
        except CodeRejected as exc:
            res.rejected, res.error = True, str(exc)
        except Exception as exc:  # noqa: BLE001 - the program's own errors go back to the model
            res.error = self.short_traceback(exc)
        finally:
            if watcher is not None:
                watcher.join(timeout=1.0)
            with self._lock:
                self._clock = None
        res.seconds = round(clock.elapsed(), 3)
        res.output = self.take_output()
        return res

    def short_traceback(self, exc: BaseException) -> str:
        frames = traceback.extract_tb(exc.__traceback__)
        lines = []
        for f in frames:
            if f.filename in self._sources:
                src = self._sources[f.filename]
                text = src[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(src) else ""
                where = "" if f.name == "<module>" else f", in {f.name}"
                lines.append(f"  {f.filename} line {f.lineno}{where}: {text}")
        inner = frames[-1] if frames and frames[-1].filename not in self._sources else None
        if inner is not None:
            mod = inner.filename.replace("\\", "/").rsplit("/", 2)
            lines.append(f"  (raised inside {inner.name} in {'/'.join(mod[-2:])})")
        msg = str(exc)
        head = "Traceback (program frames):\n" if lines else ""
        return truncate(head + "\n".join(lines[-8:]) + ("\n" if lines else "") + f"{type(exc).__name__}: {msg}", 1500)

    # -- running code
    def _compile(self, source: str) -> tuple[ast.Module, str]:
        self._n += 1
        filename = f"<block {self._n}>"
        tree = validate(source, filename=filename)
        self._sources[filename] = source.splitlines()
        return tree, filename

    def run(self, source: str, *, timeout_s: float | None = None) -> BlockResult:
        """Validate and run one block in the persistent namespace. A final bare expression is printed (REPL style)."""
        try:
            tree, filename = self._compile(source)
        except CodeRejected as exc:
            return BlockResult(error=str(exc), rejected=True)
        body, last = tree.body, None
        if body and isinstance(body[-1], ast.Expr) and not (
                isinstance(body[-1].value, ast.Call) and isinstance(body[-1].value.func, ast.Name)
                and body[-1].value.func.id == "print"):
            body, last = body[:-1], body[-1]
        code = compile(ast.Module(body=body, type_ignores=[]), filename, "exec")
        tail = compile(ast.Expression(body=last.value), filename, "eval") if last is not None else None

        def go() -> Any:
            exec(code, self.ns)                                             # noqa: S102 - validated, restricted
            if tail is not None:
                value = eval(tail, self.ns)                                 # noqa: S307
                if value is not None:
                    self._print(repr(value))
                return value
            return None

        return self._guarded(go, self.block_timeout_s if timeout_s is None else timeout_s)

    def program_source(self, entry: str = "solve") -> str | None:
        """A self-contained program from the blocks run so far: the imports, function definitions and constant
        assignments made at the top level of any block (the latest definition of each name), in the order last
        defined. None if no block defined `entry`. Statements that act (calls, loops, other assignments) are left
        out, so submitting it does not replay an experiment."""
        items: dict[str, tuple[int, str]] = {}
        order = 0
        for filename, lines in self._sources.items():
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError:
                continue
            for node in tree.body:
                key = None
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    key = "import:" + ast.unparse(node)
                elif isinstance(node, ast.FunctionDef):
                    key = node.name
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                        continue
                    key = node.targets[0].id
                if key is not None:
                    order += 1
                    items[key] = (order, ast.get_source_segment("\n".join(lines), node) or ast.unparse(node))
        if entry not in items:
            return None
        imports = [src for k, (_, src) in sorted(items.items(), key=lambda kv: kv[1][0]) if k.startswith("import:")]
        rest = [src for k, (_, src) in sorted(items.items(), key=lambda kv: kv[1][0]) if not k.startswith("import:")]
        return "\n".join(imports) + ("\n\n" if imports else "") + "\n\n\n".join(rest) + "\n"

    def call(self, fn: Callable[..., Any], *args: Any, timeout_s: float | None = None, **kwargs: Any) -> BlockResult:
        """Call a function defined in this namespace under the time limit (code.program runs solve(robot))."""
        return self._guarded(lambda: fn(*args, **kwargs), self.block_timeout_s if timeout_s is None else timeout_s)

    def define(self, source: str, name: str, *, fresh: bool = True) -> Callable[..., Any]:
        """Validate `source`, run its definitions (in a fresh namespace with this one's builtins, print and initial
        names, unless fresh=False) and return the function `name`. Raises CodeRejected. Not time-limited by itself:
        call it inside `call` or a block."""
        tree, filename = self._compile(source)
        if not defines(tree, name):
            raise CodeRejected(f"the program must define a top-level function {name}(...)")
        ns = dict(self._base) if fresh else self.ns
        exec(compile(tree, filename, "exec"), ns)                           # noqa: S102 - validated, restricted
        return ns[name]


__all__ = ["ALLOWED_IMPORTS", "NAMESPACE_IMPORTS", "BlockResult", "CodeRejected", "DENIED_ATTRS", "DENIED_NAMES", "ProgramCancelled",
           "ProgramTimeout", "SAFE_BUILTINS", "Sandbox", "defines", "truncate", "validate"]
