import time

import pytest

from hylo.codeagents.sandbox import CodeRejected, Sandbox, validate

UNSAFE = {
    "import os": "import os",
    "from subprocess import run": "subprocess",
    "import numpy.ctypeslib": "numpy.ctypeslib",
    "().__class__.__bases__": "__class__",
    "x = [].__len__": "__len__",
    "__import__('os')": "__import__",
    "open('/etc/passwd')": "open",
    "eval('1+1')": "eval",
    "exec('x=1')": "exec",
    "compile('1', 'f', 'eval')": "compile",
    "globals()": "globals",
    "getattr(print, 'x')": "getattr",
    "type(1)": "type",
    "g = (i for i in [1])\nf = g.gi_frame": "gi_frame",
    "import numpy as np\nnp.save('x.npy', np.zeros(2))": "save",
    "import numpy as np\nnp.zeros(2).tofile('x')": "tofile",
    "import numpy as np\nnp.load('x.npy')": "load",
    "class A:\n    pass": "class",
    "async def f():\n    pass": "async",
    "def _hidden():\n    pass": "_hidden",
    "robot._show": "_show",
    "def f(**kw):\n    return kw\nf(_x=1)": "_x",
}


@pytest.mark.parametrize("code", list(UNSAFE))
def test_unsafe_code_is_rejected_before_it_runs(code):
    with pytest.raises(CodeRejected, match=UNSAFE[code].replace("(", r"\(").replace(".", r"\.")):
        validate(code)
    res = Sandbox({"robot": object()}).run(code)
    assert res.rejected and res.output == ""


def test_safe_code_runs_with_math_numpy_and_a_persistent_namespace():
    sb = Sandbox()
    r1 = sb.run("import math\nimport numpy as np\nfrom numpy.linalg import norm\nv = np.array([3.0, 4.0])\n"
                "print(norm(v), math.hypot(3, 4))\nv.mean()")
    assert r1.error is None and r1.output.splitlines() == ["5.0 5.0", "np.float64(3.5)"]
    r2 = sb.run("w = v * 2\nprint(w.sum(), np.random.default_rng(0).integers(10) >= 0, sorted({3, 1}))")
    assert r2.error is None and r2.output.strip() == "14.0 True [1, 3]"


def test_builtins_are_restricted_at_run_time_too():
    sb = Sandbox()
    assert "NameError" in sb.run("x = bytearray").error                    # not in the restricted builtins
    assert sb.run("import numpy as np\nnp.memmap").rejected
    blocked = sb.run("import numpy as np\nname = 'm' + 'emmap'\nprint(np.linalg)")
    assert blocked.error is None and blocked.output.strip() == "<module numpy.linalg>"   # a read-only module view


def test_errors_come_back_as_short_tracebacks():
    sb = Sandbox()
    res = sb.run("def f(x):\n    return 1 / x\n\nf(0)")
    assert res.error.startswith("Traceback (program frames)")
    assert "line 2, in f: return 1 / x" in res.error and res.error.endswith("ZeroDivisionError: division by zero")


def test_a_busy_block_is_stopped_but_robot_time_does_not_count():
    sb = Sandbox(block_timeout_s=0.3)

    def wait(seconds):
        with sb.paused():                                                  # what TimedExecutor does around run()
            time.sleep(seconds)
        return "moved"

    sb.add(wait=wait)
    busy = sb.run("n = 0\nwhile True:\n    n += 1")
    assert busy.timed_out and "TimeoutError" in busy.error and 0.25 < busy.seconds < 1.0
    eaten = sb.run("try:\n    while True:\n        pass\nexcept Exception:\n    print('caught')")
    assert eaten.timed_out and "caught" not in eaten.output                # cannot be swallowed by the program
    slow_robot = sb.run("print(wait(0.6))")
    assert slow_robot.error is None and slow_robot.output.strip() == "moved"
    assert sb.run("print(n > 0)").output.strip() == "True"                 # the namespace survived the timeout


def test_output_is_captured_and_truncated():
    sb = Sandbox(max_output=200)
    res = sb.run("for i in range(1000):\n    print('line', i)")
    assert len(res.output) < 300 and "characters cut" in res.output and res.output.rstrip().endswith("line 999")


def test_define_needs_the_entry_function_and_runs_in_a_fresh_namespace():
    sb = Sandbox({"robot": "R"})
    sb.run("secret = 42")
    with pytest.raises(CodeRejected, match="solve"):
        sb.define("x = 1", "solve")
    fn = sb.define("def solve(robot, k=1):\n    return (robot, k)", "solve")
    assert fn("R", k=2) == ("R", 2)
    g = sb.define("def solve(robot):\n    return secret", "solve")
    assert "NameError" in sb.call(g, "R").error                           # user variables do not leak into programs
