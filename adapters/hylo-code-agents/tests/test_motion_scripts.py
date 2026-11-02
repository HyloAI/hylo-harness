"""motion_scripts + motion_freerun: the static check and the guarded script process of free mode (fast: plain
Python scripts, a toy request handler, no model and no tools)."""

from __future__ import annotations

import time

import pytest

from hylo.codeagents.motion_scripts import ScriptRejected, ScriptRunner, check_source, inside, short_traceback


@pytest.mark.parametrize("src, why", [
    ("import os", "import os"), ("import subprocess as sp", "import subprocess"), ("from .x import y", "from"),
    ("eval('1')", "eval"), ("x = ().__class__", "__class__"), ("getattr(1, 'real')", "getattr"),
    ("import numpy\nnumpy.ctypeslib", "ctypeslib"), ("async def f():\n    pass", "async"), ("x = (", "SyntaxError"),
    ("import mymod", "import mymod")])
def test_check_source_rejects(tmp_path, src, why):
    with pytest.raises(ScriptRejected, match=why.split()[0]):
        check_source(src, tmp_path)


def test_check_source_accepts_the_kit_scipy_and_checked_local_modules(tmp_path):
    check_source("from studio import *\nimport numpy as np\nfrom scipy.signal import savgol_filter\n"
                 "if __name__ == '__main__':\n    print(np.pi)", tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "helper.py").write_text("import math\ndef f(x):\n    return math.sqrt(x)\n")
    check_source("import helper\nprint(helper.f(4))", tmp_path)
    (tmp_path / "lib" / "evil.py").write_text("import socket\n")
    with pytest.raises(ScriptRejected, match="lib/evil.py"):
        check_source("import evil", tmp_path)


def test_inside_refuses_paths_outside(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert inside(tmp_path, "a.txt") == (tmp_path / "a.txt").resolve()
    for bad in ("/etc/passwd", "../a.txt", str(tmp_path) + "/../x"):
        with pytest.raises(PermissionError):
            inside(tmp_path, bad, must_exist=False)
    with pytest.raises(FileNotFoundError):
        inside(tmp_path, "missing.npz")


def test_short_traceback_keeps_the_run_directory_frames(tmp_path):
    tb = (f'Traceback (most recent call last):\n  File "/usr/lib/runpy.py", line 1, in run\n    x()\n'
          f'  File "{tmp_path}/scripts/turn01.py", line 3, in <module>\n    1 / 0\n    ~~^~~\nZeroDivisionError: division by zero')
    s = short_traceback(tb, tmp_path)
    assert "runpy" not in s and 'File "scripts/turn01.py", line 3' in s and s.endswith("division by zero")


@pytest.fixture()
def runner(tmp_path):
    calls, log = [], []

    def handler(op, args):
        calls.append(op)
        if op == "echo":
            return args
        if op == "slow":
            time.sleep(args["s"])
            return "done"
        raise PermissionError(f"refused: {op}")
    (tmp_path / "work").mkdir()
    r = ScriptRunner(tmp_path, handler, emit=log.append, timeout_s=3.0)
    r.calls, r.log = calls, log
    return r


def test_script_writes_inside_and_talks_to_the_handler(runner, tmp_path):
    res = runner.run("from studio import *\nimport numpy as np\nnp.save('work/a.npy', np.arange(3))\n"
                     "print(load_motion.__doc__.split()[0])\nprint(len(NAMES))")
    assert res.error is None, res.error
    assert (tmp_path / "work" / "a.npy").exists() and "17" in res.output
    assert (tmp_path / "scripts" / "turn01.py").exists()
    ev = [e for e in runner.log if e.get("event") == "script"]
    assert ev and ev[-1]["exit"] == 0


def test_runtime_guard_refuses_escapes(runner, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    src = ("import pathlib\nfor what, f in [('write', lambda: pathlib.Path(%r).write_text('x')),\n"
           "                  ('read', lambda: pathlib.Path('/etc/hostname').read_text()),\n"
           "                  ('mkdir', lambda: pathlib.Path(%r).mkdir())]:\n"
           "    try:\n        f()\n        print(what, 'ALLOWED')\n"
           "    except PermissionError as e:\n        print(what, 'refused')\n") % (str(outside), str(outside) + "-d")
    res = runner.run(src)
    assert res.error is None, res.error
    assert res.output.count("refused") == 3 and "ALLOWED" not in res.output and not outside.exists()


def test_runtime_guard_refuses_processes_and_sockets(runner, tmp_path):
    (tmp_path / "lib").mkdir()
    # a local module passes the static check only if clean; this one hides nothing, the guard acts at run time
    res = runner.run("import pathlib\nimport json\nprint(json.dumps(sorted(p.name for p in pathlib.Path('.').iterdir())))")
    assert res.error is None and "scripts" in res.output
    blocked = ScriptRunner(tmp_path, lambda op, a: None, emit=lambda r: None, timeout_s=5)
    script = tmp_path / "raw.py"                     # bypass the static check: the guard alone must stop it
    script.write_text("import subprocess\ntry:\n    subprocess.run(['true'])\n    print('RAN')\n"
                      "except PermissionError as e:\n    print('refused', e)\n"
                      "import socket\ntry:\n    socket.socket()\n    print('SOCKET')\nexcept PermissionError:\n"
                      "    print('refused socket')\n")
    import subprocess
    import sys
    from hylo.codeagents.motion_scripts import RUNNER
    p = subprocess.run([sys.executable, "-I", str(RUNNER), str(script), str(tmp_path)], capture_output=True, text=True,
                       timeout=60)
    assert "refused" in p.stdout and "refused socket" in p.stdout and "RAN" not in p.stdout and "SOCKET" not in p.stdout
    assert blocked.scripts == 0


def test_handler_requests_refusals_and_timeouts(runner):
    def handler(op, args):
        runner.calls.append(op)
        if op == "characters":
            return ["echo"]
        if op == "check":
            time.sleep(4.0)                                           # longer than the 3 s script limit
            return {"problems": ["none found"]}
        raise PermissionError(f"refused: {op}")
    runner.handler = handler
    res = runner.run("from studio import *\nprint(characters())\ntry:\n    look('x.png')\nexcept ToolError as e:\n"
                     "    print('tool error:', e)\nprint(check('m.npz', 'g1'))")
    assert res.error is None, res.error
    lines = res.output.strip().splitlines()
    assert lines[0] == "['echo']" and lines[1] == "tool error: refused: look" and "none found" in lines[2]
    assert runner.refusals == 1 and runner.calls == ["characters", "look", "check"]
    assert [e["op"] for e in runner.log if e.get("event") == "request"] == ["characters", "look", "check"]
    res = runner.run("while True:\n    pass")
    assert res.timed_out and "TimeoutError" in res.error
    res = runner.run("x = 1 / 0")
    assert "ZeroDivisionError" in res.error and "turn" in res.error and "runpy" not in res.error


def test_rejected_scripts_do_not_run(runner, tmp_path):
    res = runner.run("import os\nos.system('touch HACK')")
    assert res.rejected and not (tmp_path / "HACK").exists()
    assert runner.log[-1]["event"] == "script" and "rejected" in runner.log[-1]
