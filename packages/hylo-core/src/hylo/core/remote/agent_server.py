"""AgentServer: hosts one Agent in its own process (and venv), mirroring the EnvServer.

Protocol `hylo.agentserver/v0.1`, newline-delimited JSON on stdio:
    -> {"hello": "hylo.agentserver/v0.1"}
    <- {"id", "method": "spec" | "reset" | "act" | "on_step" | "on_episode_end" | "close", "args"}
    -> {"id", "ok": true, "result": ..., "events": [[type, payload], ...]} | {"id", "ok": false, "error", "traceback"}

Events the agent emits through EpisodeContext.emit are buffered and returned with each reply, so the
kernel in the parent process writes them to the trace as if the agent were local.

    python -m hylo.core.remote.agent_server <module:attr> '<json params>'
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from pydantic import BaseModel

from .. import codec
from ..agent import EpisodeContext, EpisodeOutcome
from ..task import TaskInstance
from ..types import EnvSpec, Observation, StepResult
from .protocol import claim_stdout
from .handlers import load_component

PROTOCOL = "hylo.agentserver/v0.1"


def _dump(result: Any) -> Any:
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else codec.encode(result)


def main(argv: list[str]) -> int:
    out, pending = claim_stdout(), []
    ref, params = argv[1], json.loads(argv[2]) if len(argv) > 2 else {}
    agent = load_component(ref, params, "agents")      # "module:attr" or a registered plugin name
    out.write(json.dumps({"hello": PROTOCOL}) + "\n")
    out.flush()
    for line in sys.stdin:
        msg = json.loads(line)
        rid, method, args = msg["id"], msg["method"], msg.get("args", {})
        try:
            if method == "spec":
                result = agent.spec()
            elif method == "reset":
                ctx = EpisodeContext(episode_id=args["episode_id"], task=TaskInstance.model_validate(args["task"]),
                                     env_spec=EnvSpec.model_validate(args["env_spec"]), seed=args["seed"],
                                     emit=lambda t, p: pending.append([t, codec.encode(dict(p))]))
                result = agent.reset(ctx)
            elif method == "act":
                result = agent.act(Observation.model_validate(args["observation"]))
            elif method == "on_step":
                result = agent.on_step(StepResult.model_validate(args["result"]))
            elif method == "on_episode_end":
                result = agent.on_episode_end(EpisodeOutcome.model_validate(args["outcome"]))
            elif method == "close":
                out.write(json.dumps({"id": rid, "ok": True, "result": None, "events": []}) + "\n")
                out.flush()
                return 0
            else:
                raise ValueError(f"unknown method {method!r}")
            reply = {"id": rid, "ok": True, "result": _dump(result), "events": pending}
        except Exception as exc:  # noqa: BLE001 - errors are data on the wire
            reply = {"id": rid, "ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "events": pending}
        pending = []
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
