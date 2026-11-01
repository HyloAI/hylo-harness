"""Transport-agnostic request dispatch for the EnvServer and AgentServer protocols.

A handler turns one decoded request {"id", "method", "args"} into one JSON-safe reply. The stdio servers
and the WebSocket server (ws.py) share these, so both transports speak exactly the same messages.
"""

from __future__ import annotations

import importlib
import traceback
from typing import Any

from pydantic import BaseModel

from .. import codec
from ..agent import EpisodeContext, EpisodeOutcome
from ..task import TaskInstance
from ..types import Action, EnvSpec, Observation, StepResult


def _dump(result: Any) -> Any:
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else codec.encode(result)


def load_component(ref: str, params: dict[str, Any] | None, group: str) -> Any:
    """Build from 'module:attr' or, failing that, from a registered plugin name in `group`."""
    if ":" in ref:
        module, _, attr = ref.partition(":")
        return getattr(importlib.import_module(module), attr)(**(params or {}))
    from ..registry import registry

    return registry.build(group, {"type": ref, "params": params or {}})


class EnvHandler:
    protocol = "wc.envserver/v0.1"

    def __init__(self, env: Any) -> None:
        self.env = env
        self.closed = False

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        rid, method, args = msg["id"], msg["method"], msg.get("args", {})
        try:
            if method == "spec":
                # Omit unset optional fields so older clients (extra="forbid") still accept newer servers' specs.
                result = self.env.spec().model_dump(mode="json", exclude_none=True)
            elif method == "reset":
                result = _dump(self.env.reset(TaskInstance.model_validate(args["task"])))
            elif method == "step":
                result = _dump(self.env.step(Action.model_validate(args["action"])))
            elif method == "privileged_state":
                result = _dump(self.env.privileged_state())
            elif method == "close":
                self.close()
                result = None
            else:
                raise ValueError(f"unknown method {method!r}")
            return {"id": rid, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - errors are data on the wire
            return {"id": rid, "ok": False, "error": repr(exc), "traceback": traceback.format_exc()}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.env.close()


class AgentHandler:
    protocol = "wc.agentserver/v0.1"

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.closed = False
        self._pending: list[list[Any]] = []

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        rid, method, args = msg["id"], msg["method"], msg.get("args", {})
        try:
            if method == "spec":
                result = self.agent.spec()
            elif method == "reset":
                ctx = EpisodeContext(episode_id=args["episode_id"], task=TaskInstance.model_validate(args["task"]),
                                     env_spec=EnvSpec.model_validate(args["env_spec"]), seed=args["seed"],
                                     emit=lambda t, p: self._pending.append([t, codec.encode(dict(p))]))
                result = self.agent.reset(ctx)
            elif method == "act":
                result = self.agent.act(Observation.model_validate(args["observation"]))
            elif method == "on_step":
                result = self.agent.on_step(StepResult.model_validate(args["result"]))
            elif method == "on_episode_end":
                result = self.agent.on_episode_end(EpisodeOutcome.model_validate(args["outcome"]))
            elif method == "close":
                self.close()
                result = None
            else:
                raise ValueError(f"unknown method {method!r}")
            reply = {"id": rid, "ok": True, "result": _dump(result), "events": self._pending}
        except Exception as exc:  # noqa: BLE001 - errors are data on the wire
            reply = {"id": rid, "ok": False, "error": repr(exc), "traceback": traceback.format_exc(),
                     "events": self._pending}
        self._pending = []
        return reply

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            close = getattr(self.agent, "close", None)
            if callable(close):
                close()
