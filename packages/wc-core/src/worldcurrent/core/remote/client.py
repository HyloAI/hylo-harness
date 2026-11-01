"""RemoteEnvironment: an Environment proxy for an EnvServer subprocess."""

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
from ..errors import EnvironmentFault
from ..task import TaskInstance
from ..types import Action, EnvSpec, Observation, StepResult
from .protocol import PROTOCOL


class RemoteEnvironment:
    def __init__(self, ref: str, params: dict[str, Any] | None = None, *, python: str | None = None,
                 env: dict[str, str] | None = None, runtime: str | None = None) -> None:
        # The child interpreter must have wc-core and the env's package installed (its own venv). `runtime` names
        # it in this machine's profile (core.profile); explicit python / env override the runtime's.
        python, env = resolve(runtime, python, env)
        self._proc = subprocess.Popen(
            [os.path.expanduser(python) if python else sys.executable, "-m", "worldcurrent.core.remote.server", ref, json.dumps(params or {})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env(env))
        hello = json.loads(self._readline())
        if hello.get("hello") != PROTOCOL:
            raise EnvironmentFault(f"protocol mismatch: {hello}")
        self._ids = itertools.count(1)
        self._spec: EnvSpec | None = None

    def _readline(self) -> str:
        line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise EnvironmentFault(f"env server exited with code {self._proc.poll()}")
        return line

    def _call(self, method: str, **args: Any) -> Any:
        rid = next(self._ids)
        self._proc.stdin.write(json.dumps({"id": rid, "method": method, "args": codec.encode(args)}) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        reply = json.loads(self._readline())
        if reply["id"] != rid:
            raise EnvironmentFault(f"out-of-order reply {reply['id']} != {rid}")
        if not reply["ok"]:
            raise EnvironmentFault(f"remote {method} failed: {reply['error']}")
        return reply["result"]

    def spec(self) -> EnvSpec:
        if self._spec is None:
            self._spec = EnvSpec.model_validate(self._call("spec"))
        return self._spec

    def reset(self, task: TaskInstance) -> Observation:
        return Observation.model_validate(self._call("reset", task=task))

    def step(self, action: Action) -> StepResult:
        return StepResult.model_validate(self._call("step", action=action))

    def privileged_state(self) -> dict[str, Any]:
        return codec.decode(self._call("privileged_state"))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._call("close")
            except EnvironmentFault:
                pass
            self._proc.wait(timeout=10)
