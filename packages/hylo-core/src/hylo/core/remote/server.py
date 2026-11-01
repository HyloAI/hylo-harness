"""EnvServer: hosts one Environment in its own process (in practice its own venv/container), so
dependency-incompatible benchmarks (LIBERO: py3.8 + robosuite 1.4; RoboCasa: py3.11 + robosuite>=1.5.2)
run side by side.

    python -m hylo.core.remote.server <module:Class> '<json params>'
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from pydantic import BaseModel

from .. import codec
from ..task import TaskInstance
from ..types import Action
from .protocol import PROTOCOL, claim_stdout
from .handlers import load_component


def _dump(result: Any) -> Any:
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else codec.encode(result)


def main(argv: list[str]) -> int:
    out = claim_stdout()
    ref, params = argv[1], json.loads(argv[2]) if len(argv) > 2 else {}
    env = load_component(ref, params, "envs")      # "module:attr" or a registered plugin name
    out.write(json.dumps({"hello": PROTOCOL}) + "\n")
    out.flush()
    for line in sys.stdin:
        msg = json.loads(line)
        rid, method, args = msg["id"], msg["method"], msg.get("args", {})
        try:
            if method == "spec":
                # Omit unset optional fields so older clients (extra="forbid") still accept newer servers' specs.
                result = env.spec().model_dump(mode="json", exclude_none=True)
            elif method == "reset":
                result = env.reset(TaskInstance.model_validate(args["task"]))
            elif method == "step":
                result = env.step(Action.model_validate(args["action"]))
            elif method == "privileged_state":
                result = env.privileged_state()
            elif method == "close":
                env.close()
                out.write(json.dumps({"id": rid, "ok": True, "result": None}) + "\n")
                out.flush()
                return 0
            else:
                raise ValueError(f"unknown method {method!r}")
            reply = {"id": rid, "ok": True, "result": _dump(result)}
        except Exception as exc:  # noqa: BLE001 - errors are data on the wire
            reply = {"id": rid, "ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
