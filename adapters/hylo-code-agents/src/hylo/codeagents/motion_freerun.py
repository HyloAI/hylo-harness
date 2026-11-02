# hylo: worker-side (the process that runs one free-mode script; started by motion_free with python -I)
"""Runs one model-written script of free mode inside the run directory, under a guard:

- resource limits: address space, file size, CPU time (the studio also enforces a wall-clock limit);
- an audit hook (PEP 578) refuses: opening a file for writing outside the run directory, reading files outside
  the run directory and the Python installation, deleting / renaming / creating / linking paths outside the run
  directory, symbolic and hard links, changing directory out of it, starting processes, sockets, ctypes;
- the script sees the studio kit as `studio` (`from studio import *`) and motion_ops as `ops`.

The studio has already checked the script's source (imports, names) before starting this process. This is a
guard against a model's mistakes and casual escapes, not an operating-system jail: run untrusted models in a
container as well.

    python -I motion_freerun.py SCRIPT RUN_DIR
"""

from __future__ import annotations

import importlib.util
import os
import resource
import runpy
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
BLOCKED = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
           "os.kill", "os.killpg", "pty.spawn", "socket.__new__", "socket.connect", "socket.bind",
           "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata", "ctypes.call_function",
           "ctypes.addressof", "os.symlink", "os.link", "sys.setprofile", "sys.settrace", "winreg.OpenKey",
           "urllib.Request", "webbrowser.open", "os.startfile", "_posixsubprocess.fork_exec"}
PATH_EVENTS = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 1, "os.chmod": 1, "os.chown": 1, "os.utime": 1,
               "os.truncate": 1, "shutil.rmtree": 1, "os.rename": 2, "os.replace": 2, "shutil.copyfile": 2,
               "shutil.copytree": 2, "shutil.move": 2, "os.chdir": 1, "os.listdir": 1, "os.scandir": 1,
               "glob.glob": 1, "os.chflags": 1, "os.lchown": 1, "os.setxattr": 1, "os.removexattr": 1}
READ_ONLY_EVENTS = {"os.listdir", "os.scandir", "glob.glob"}


def _limits() -> None:
    gb = 1 << 30
    for res, val in ((resource.RLIMIT_AS, 48 * gb), (resource.RLIMIT_FSIZE, 2 * gb),
                     (resource.RLIMIT_CPU, int(os.environ.get("HYLO_SCRIPT_CPU_S", "7200")))):
        try:
            soft, hard = resource.getrlimit(res)
            val = val if hard == resource.RLIM_INFINITY else min(val, hard)
            resource.setrlimit(res, (val, hard))
        except (ValueError, OSError):
            pass


def install_guard(run_dir: Path) -> None:
    run = os.path.realpath(run_dir)
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sysconfig.get_paths()["purelib"],
                                           sysconfig.get_paths()["platlib"], sysconfig.get_paths()["stdlib"], HERE)}
    roots |= {os.path.realpath(p) for p in sys.path if p and os.path.isdir(p)}
    roots |= {"/dev/null", "/dev/urandom", "/proc/self", "/proc/cpuinfo", "/sys/devices/system/cpu", "/etc/localtime",
              "/usr/share/zoneinfo", "/usr/lib/locale"}
    roots.discard("/")
    roots.discard(os.path.realpath(os.path.expanduser("~")))

    def inside(path, base) -> bool:
        return path == base or path.startswith(base.rstrip("/") + "/")

    def check(path, write: bool, what: str) -> None:
        if isinstance(path, int):                      # an already open descriptor
            return
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if path is None:
            return
        p = os.path.realpath(os.fspath(path))
        if inside(p, run):
            return
        if not write and any(inside(p, r) for r in roots):
            return
        raise PermissionError(f"refused: {what} {path!s} is outside the run directory {run}")

    def hook(event: str, args: tuple) -> None:
        if event in BLOCKED or event.startswith(("os.exec", "os.spawn", "os.posix_spawn")):
            raise PermissionError(f"refused: {event} is not allowed in free-mode scripts (use the studio tools)")
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            write = bool(mode and any(c in str(mode) for c in "wax+")) or bool(isinstance(flags, int)
                                                                              and flags & WRITE_FLAGS)
            check(path, write, "writing" if write else "reading")
        elif event in PATH_EVENTS:
            for path in args[:PATH_EVENTS[event]]:
                check(path, event not in READ_ONLY_EVENTS, event)
    sys.addaudithook(hook)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    script, run_dir = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    _limits()
    os.chdir(run_dir)
    import numpy                                         # noqa: F401 - warm up before the guard (shared libraries)
    try:
        import scipy                                     # noqa: F401
        import scipy.interpolate                         # noqa: F401
        import scipy.ndimage                             # noqa: F401
        import scipy.signal                              # noqa: F401
        import scipy.spatial.transform                   # noqa: F401
    except ImportError:
        pass
    ops = _load("motion_ops", HERE / "motion_ops.py")
    kit = _load("studio", HERE / "motion_freekit.py")
    kit.bind(ops)
    sys.modules["ops"] = ops
    for p in (run_dir / "lib", run_dir):
        sys.path.insert(0, str(p))
    install_guard(run_dir)
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
