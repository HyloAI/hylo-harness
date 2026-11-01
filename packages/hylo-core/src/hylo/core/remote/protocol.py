"""EnvServer protocol `hylo.envserver/v0.1`: newline-delimited JSON over a subprocess's stdio.

Contracts are pydantic models, so the wire format is just `model_dump(mode="json")` and
`model_validate`; arrays travel through core.codec. No pickle.

    -> {"hello": "hylo.envserver/v0.1"}                       (server, once)
    <- {"id": 1, "method": "spec" | "reset" | "step" | "privileged_state" | "close", "args": {...}}
    -> {"id": 1, "ok": true, "result": ...} | {"id": 1, "ok": false, "error": "...", "traceback": "..."}
"""

PROTOCOL = "hylo.envserver/v0.1"
METHODS = ("spec", "reset", "step", "privileged_state", "close")


def child_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Environment for a component subprocess: the parent's, plus `extra` with ~ and $VARS expanded."""
    import os

    env = dict(os.environ)
    for k, v in (extra or {}).items():
        env[k] = os.path.expandvars(os.path.expanduser(v))
    return env


def claim_stdout():
    """Reserve the real stdout for the protocol and send everything else written to fd 1 (Python prints,
    C extensions, library banners) to stderr, so third-party output can never corrupt the stream."""
    import os
    import sys

    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "w", buffering=1)
