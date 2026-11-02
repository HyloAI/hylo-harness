"""The common runner: every tool runs as a worker script in its own uv environment, on the CPU.

A worker is a standalone Python script under `workers/` with inline PEP 723 dependencies (`uv run --script`), so
heavy and conflicting dependencies (torch, transformers, mediapipe, coacd, ...) never enter the Hylo environment.
The exchange is files: the runner writes the inputs (images as PNG, arrays as .npz, meshes as they are) and a
request JSON into a work directory; the worker writes `result.json` and its output files (.npz, .png, .glb, .obj,
.mp4, .xml) next to it. Each result carries timing and provenance (model id, revision, licence, backend, device).

Two ways to run a worker:

    run_worker("segment", {"prompts": ["red can"]}, inputs={"image": rgb})      # one shot: new process per call
    with ToolServer("segment") as srv:                                         # persistent: load the model once
        srv.call({"prompts": [...]}, inputs={"image": rgb})

Environment variables:
    HYLO_TOOLS_CACHE    work directories and small model files (default ~/.cache/hylo/tools)
    HYLO_TOOLS_THREADS  CPU threads per worker (default 32)
    HYLO_TOOLS_UV       the uv executable (default: `uv` on PATH)
Model weights go to the default Hugging Face / torch caches.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

WORKERS = Path(__file__).resolve().parent / "workers"
RESPONSE_PREFIX = "@@hylo-tool "


def cache_dir(*parts: str) -> Path:
    """A directory under the tools cache (created)."""
    root = Path(os.environ.get("HYLO_TOOLS_CACHE") or Path.home() / ".cache" / "hylo" / "tools").expanduser()
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_threads() -> int:
    return int(os.environ.get("HYLO_TOOLS_THREADS", "32"))


class ToolError(RuntimeError):
    """A worker failed; `blocked` names a user action when the failure is a missing licence, login or file."""

    def __init__(self, message: str, *, blocked: str | None = None, log: str = "") -> None:
        super().__init__(message)
        self.blocked, self.log = blocked, log


@dataclass
class ToolResult:
    """What a worker returned: JSON outputs, named output files, provenance, timings and the work directory."""
    tool: str
    outputs: dict[str, Any]
    files: dict[str, Path]
    provenance: dict[str, Any]
    timing: dict[str, float]
    workdir: Path
    log: str = ""

    def arrays(self, name: str = "arrays") -> dict[str, np.ndarray]:
        """The arrays of an output .npz file."""
        with np.load(self.files[name], allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def to_json(self) -> dict[str, Any]:
        return {"tool": self.tool, "outputs": self.outputs, "files": {k: str(v) for k, v in self.files.items()},
                "provenance": self.provenance, "timing": self.timing, "workdir": str(self.workdir)}


@dataclass
class _Prepared:
    workdir: Path
    request_path: Path
    out_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out_dir = self.workdir / "out"


def worker_path(name: str) -> Path:
    p = WORKERS / f"{name}.py"
    if not p.exists():
        raise FileNotFoundError(f"no worker {name!r} in {WORKERS}")
    return p


def _uv() -> str:
    exe = os.environ.get("HYLO_TOOLS_UV") or shutil.which("uv")
    if not exe:
        raise ToolError("uv not found: install uv (https://docs.astral.sh/uv/) or set HYLO_TOOLS_UV")
    return exe


def worker_env(threads: int | None = None, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The worker's environment: CPU only, bounded threads, Hylo's own virtualenv hidden."""
    n = str(threads or default_threads())
    env = dict(os.environ)
    for k in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
                "NUMEXPR_NUM_THREADS": n, "HYLO_TOOLS_THREADS": n, "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1", "HYLO_TOOLS_CACHE": str(cache_dir())})
    env.update(extra or {})
    return env


def _save_input(value: Any, dst_dir: Path, name: str) -> str:
    """Inputs: an existing file path is passed as is; an (H, W, 3|4) uint8 array or PIL image becomes PNG; other
    arrays become .npy; a dict of arrays becomes .npz."""
    if isinstance(value, (str, Path)):
        p = Path(value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)
    if hasattr(value, "save") and hasattr(value, "mode"):   # PIL image
        p = dst_dir / f"{name}.png"
        value.save(p)
        return str(p)
    if isinstance(value, dict):
        p = dst_dir / f"{name}.npz"
        np.savez_compressed(p, **{k: np.asarray(v) for k, v in value.items()})
        return str(p)
    arr = np.asarray(value)
    if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] in (3, 4):
        from PIL import Image
        p = dst_dir / f"{name}.png"
        Image.fromarray(arr).save(p)
        return str(p)
    p = dst_dir / f"{name}.npy"
    np.save(p, arr)
    return str(p)


def prepare(tool: str, request: Mapping[str, Any], inputs: Mapping[str, Any] | None, workdir: str | Path | None,
            threads: int | None) -> _Prepared:
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix=f"{time.strftime('%Y%m%d-%H%M%S')}-", dir=cache_dir("runs", tool)))
    wd = Path(workdir).expanduser().resolve()
    (wd / "in").mkdir(parents=True, exist_ok=True)
    prep = _Prepared(wd, wd / f"request-{uuid.uuid4().hex[:8]}.json")
    prep.out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: _save_input(v, wd / "in", k) for k, v in (inputs or {}).items() if v is not None}
    body = {"tool": tool, "params": dict(request), "inputs": files, "out_dir": str(prep.out_dir),
            "threads": threads or default_threads()}
    prep.request_path.write_text(json.dumps(body, indent=2, default=_json_default))
    return prep


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _result(tool: str, reply: dict[str, Any], workdir: Path, wall: float, log: str) -> ToolResult:
    if not reply.get("ok"):
        raise ToolError(f"{tool}: {reply.get('error', 'worker failed')}", blocked=reply.get("blocked"), log=log)
    timing = dict(reply.get("timing", {}))
    timing["wall_s"] = round(wall, 3)
    files = {k: Path(v) for k, v in reply.get("files", {}).items()}
    return ToolResult(tool, reply.get("outputs", {}), files, reply.get("provenance", {}), timing, workdir, log)


def command(tool: str, *args: str) -> list[str]:
    return [_uv(), "run", "--quiet", "--script", str(worker_path(tool)), *args]


def run_worker(tool: str, request: Mapping[str, Any] | None = None, inputs: Mapping[str, Any] | None = None, *,
               workdir: str | Path | None = None, threads: int | None = None, timeout: float | None = None,
               env: Mapping[str, str] | None = None, verbose: bool = False) -> ToolResult:
    """Run worker `tool` once in its own environment and return its result (raises ToolError on failure). The
    first call of a worker builds its environment (uv caches it); wall time includes that and the model load."""
    prep = prepare(tool, request or {}, inputs, workdir, threads)
    t0 = time.perf_counter()
    proc = subprocess.run(command(tool, str(prep.request_path)), env=worker_env(threads, env), text=True,
                          stdout=subprocess.PIPE, stderr=None if verbose else subprocess.STDOUT, timeout=timeout)
    wall = time.perf_counter() - t0
    log = proc.stdout or ""
    (prep.workdir / "worker.log").write_text(log)
    res_path = prep.out_dir / "result.json"
    if not res_path.exists():
        tail = "\n".join(log.strip().splitlines()[-25:])
        raise ToolError(f"{tool}: worker exited with {proc.returncode} and no result\n{tail}", log=log)
    return _result(tool, json.loads(res_path.read_text()), prep.workdir, wall, log)


class ToolServer:
    """A persistent worker: the environment is built and the model loaded once; each call is one request. Replies
    come back on the worker's stdout as single lines with a fixed prefix (other output goes to the log)."""

    def __init__(self, tool: str, *, threads: int | None = None, env: Mapping[str, str] | None = None,
                 workdir: str | Path | None = None, startup_timeout: float = 3600.0) -> None:
        self.tool, self.threads, self.startup_timeout = tool, threads, startup_timeout
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(dir=cache_dir("runs", tool)))
        self._env = worker_env(threads, env)
        self._proc: subprocess.Popen | None = None
        self._log: list[str] = []
        self._replies: list[str] = []
        self._cv = threading.Condition()
        self._n = 0

    def __enter__(self) -> "ToolServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        self._proc = subprocess.Popen(command(self.tool, "--serve"), env=self._env, text=True, bufsize=1,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._read, daemon=True).start()
        ready = self._wait(self.startup_timeout)
        if not ready.get("ok"):
            raise ToolError(f"{self.tool}: worker failed to start: {ready.get('error')}", blocked=ready.get("blocked"),
                            log=self.log)

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            with self._cv:
                if line.startswith(RESPONSE_PREFIX):
                    self._replies.append(line[len(RESPONSE_PREFIX):])
                else:
                    self._log.append(line)
                self._cv.notify_all()
        with self._cv:
            self._replies.append(json.dumps({"ok": False, "error": "worker exited"}))
            self._cv.notify_all()

    def _wait(self, timeout: float | None) -> dict[str, Any]:
        with self._cv:
            if not self._cv.wait_for(lambda: self._replies, timeout=timeout):
                raise ToolError(f"{self.tool}: no reply within {timeout} s", log=self.log)
            return json.loads(self._replies.pop(0))

    @property
    def log(self) -> str:
        return "".join(self._log)

    def call(self, request: Mapping[str, Any] | None = None, inputs: Mapping[str, Any] | None = None, *,
             timeout: float | None = None) -> ToolResult:
        if self._proc is None or self._proc.poll() is not None:
            raise ToolError(f"{self.tool}: worker is not running", log=self.log)
        self._n += 1
        prep = prepare(self.tool, request or {}, inputs, self.workdir / f"call-{self._n:04d}", self.threads)
        t0 = time.perf_counter()
        assert self._proc.stdin
        self._proc.stdin.write(str(prep.request_path) + "\n")
        self._proc.stdin.flush()
        reply = self._wait(timeout)
        return _result(self.tool, reply, prep.workdir, time.perf_counter() - t0, "")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                assert self._proc.stdin
                self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None


def print_result(res: ToolResult, stream: Any = None) -> None:
    json.dump(res.to_json(), stream or sys.stdout, indent=2, default=_json_default)
    (stream or sys.stdout).write("\n")
