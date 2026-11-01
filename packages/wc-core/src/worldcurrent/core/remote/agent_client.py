"""RemoteAgent: an Agent proxy for an AgentServer subprocess (e.g. a VLA in its own torch venv)."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from typing import Any

from .. import codec
from ..profile import resolve
from .protocol import child_env
from ..agent import AgentSpec, EpisodeContext, EpisodeOutcome
from ..errors import AgentError
from ..types import ActionChunk, Observation, StepResult
from .agent_server import PROTOCOL


class RemoteAgent:
    def __init__(self, ref: str, params: dict[str, Any] | None = None, python: str | None = None,
                 env: dict[str, str] | None = None, forward_steps: bool = True, runtime: str | None = None) -> None:
        # forward_steps: send each step's state (images stripped) so stateful agents can track progress.
        # runtime: the venv in this machine's profile (core.profile); explicit python / env override it.
        self.forward_steps = forward_steps
        python, env = resolve(runtime, python, env)
        self._proc = subprocess.Popen(
            [os.path.expanduser(python) if python else sys.executable, "-m", "worldcurrent.core.remote.agent_server", ref, json.dumps(params or {})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env(env))
        hello = json.loads(self._readline())
        if hello.get("hello") != PROTOCOL:
            raise AgentError(f"protocol mismatch: {hello}")
        self._ids = itertools.count(1)
        self._spec: AgentSpec | None = None
        self._ctx: EpisodeContext | None = None

    def _readline(self) -> str:
        line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise AgentError(f"agent server exited with code {self._proc.poll()}")
        return line

    def _call(self, method: str, **args: Any) -> Any:
        rid = next(self._ids)
        self._proc.stdin.write(json.dumps({"id": rid, "method": method, "args": codec.encode(args)}) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        reply = json.loads(self._readline())
        if self._ctx is not None:
            for type_, payload in reply.get("events", []):
                self._ctx.emit(type_, codec.decode(payload))
        if not reply["ok"]:
            raise AgentError(f"remote {method} failed: {reply['error']}\n{reply.get('traceback', '')}")
        return reply["result"]

    def spec(self) -> AgentSpec:
        if self._spec is None:
            self._spec = AgentSpec.model_validate(self._call("spec"))
        return self._spec

    def reset(self, ctx: EpisodeContext) -> None:
        self._ctx = ctx
        self._call("reset", episode_id=ctx.episode_id, task=ctx.task, env_spec=ctx.env_spec, seed=ctx.seed)

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk.model_validate(self._call("act", observation=observation))

    def on_step(self, result: StepResult) -> None:
        if self.forward_steps:
            light = result.model_copy(update={"observation": result.observation.model_copy(update={"images": {}})})
            self._call("on_step", result=light)

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        self._call("on_episode_end", outcome=outcome)

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._call("close")
            except AgentError:
                pass
            self._proc.wait(timeout=30)
