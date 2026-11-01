"""WebSocket transport for the EnvServer / AgentServer protocols: run a simulator or a policy on another
machine, container or GPU box, with the same JSON messages as the stdio transport (handlers.py).

    # serve (in the component's own venv; binds localhost unless --host is given)
    python -m hylo.core.remote.ws env  libero --params '{"image_size": 256}' --port 8765
    python -m hylo.core.remote.ws agent lerobot --params '{...}' --port 8766 --token-env HYLO_TOKEN

    # use from any experiment config
    env:   {type: remote.ws, params: {url: "ws://gpu-box:8765"}}
    agent: {type: remote.ws, params: {url: "ws://gpu-box:8766", token_env: HYLO_TOKEN}}

Each connection gets its own component instance (parallel workers = parallel connections, as with stdio
subprocesses). One text frame per request and per reply; the server sends {"hello": PROTOCOL} first.
GET /healthz answers 200 for load balancers. Auth is an optional shared bearer token; use TLS (wss://)
or an SSH tunnel across untrusted networks.
"""

from __future__ import annotations

import argparse
import hmac
import http
import itertools
import json
import os
import sys
import threading
import time
from typing import Any, Callable

from .. import codec
from ..agent import AgentSpec, EpisodeContext, EpisodeOutcome
from ..errors import AgentError, EnvironmentFault
from ..task import TaskInstance
from ..types import ActionChunk, EnvSpec, Observation, StepResult
from .handlers import AgentHandler, EnvHandler, load_component


def _ws():
    try:
        import websockets.sync.client  # noqa: F401
        import websockets.sync.server  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise ImportError("the WebSocket transport needs `websockets>=13` in this environment") from exc
    import websockets

    return websockets


# ---------------------------------------------------------------- server

class ComponentServer:
    """Serves one component type; `factory()` builds a fresh handler per connection."""

    def __init__(self, factory: Callable[[], EnvHandler | AgentHandler], *, host: str = "127.0.0.1", port: int = 8765,
                 token: str | None = None, max_connections: int = 64,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> None:
        self.factory, self.host, self.port, self.token, self.log = factory, host, port, token, log
        self._slots = threading.BoundedSemaphore(max_connections)
        self._server = None

    def _process_request(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self.token is not None:
            got = request.headers.get("Authorization", "")
            if not hmac.compare_digest(got, f"Bearer {self.token}"):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    def _handle(self, conn) -> None:
        if not self._slots.acquire(blocking=False):
            conn.close(code=1013, reason="server at max_connections")
            return
        handler = None
        try:
            handler = self.factory()
            conn.send(json.dumps({"hello": handler.protocol, "transport": "ws"}))
            self.log(f"[ws] {conn.remote_address} connected ({handler.protocol})")
            for frame in conn:
                reply = handler.handle(json.loads(frame))
                conn.send(json.dumps(reply))
                if handler.closed:
                    break
        except Exception as exc:  # noqa: BLE001 - one bad connection must not take the server down
            self.log(f"[ws] {conn.remote_address} dropped: {exc!r}")
        finally:
            if handler is not None:
                try:
                    handler.close()
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[ws] close failed: {exc!r}")
            self._slots.release()
            self.log(f"[ws] {conn.remote_address} closed")

    def start(self) -> "ComponentServer":
        """Start serving on a background thread (tests, embedding). `port=0` picks a free port."""
        ws = _ws()
        self._server = ws.sync.server.serve(self._handle, self.host, self.port, max_size=None,
                                            process_request=self._process_request)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True, name="hylo-ws-server").start()
        return self

    def serve_forever(self) -> None:
        ws = _ws()
        with ws.sync.server.serve(self._handle, self.host, self.port, max_size=None,
                                  process_request=self._process_request) as server:
            self._server = server
            self.log(f"[ws] serving on ws://{self.host}:{server.socket.getsockname()[1]}")
            server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


# ---------------------------------------------------------------- clients

class _Client:
    protocol = ""
    fault: type[Exception] = RuntimeError

    def __init__(self, url: str, *, token: str | None = None, token_env: str | None = None,
                 connect_timeout_s: float = 60.0, call_timeout_s: float | None = 600.0) -> None:
        ws = _ws()
        token = token or (os.environ.get(token_env) if token_env else None)
        headers = {"Authorization": f"Bearer {token}"} if token else None
        import inspect

        # websockets>=17 deprecates holding a connection outside `with`; a long-lived proxy opts into that explicitly.
        direct = {"legacy": True} if "legacy" in inspect.signature(ws.sync.client.connect).parameters else {}
        deadline, last = time.monotonic() + connect_timeout_s, None
        while True:                                   # wait for a server that is still starting
            try:
                self._conn = ws.sync.client.connect(url, max_size=None, additional_headers=headers, open_timeout=10,
                                                    **direct)
                break
            except ws.exceptions.InvalidHandshake as exc:   # e.g. 401: retrying will not help
                raise self.fault(f"{url} rejected the connection: {exc}") from exc
            except (ConnectionRefusedError, OSError, TimeoutError) as exc:
                last = exc
                if time.monotonic() > deadline:
                    raise self.fault(f"cannot reach {url}: {last!r}") from exc
                time.sleep(1.0)
        hello = json.loads(self._conn.recv(timeout=connect_timeout_s))
        if hello.get("hello") != self.protocol:
            raise self.fault(f"protocol mismatch at {url}: {hello}")
        self.url, self._timeout, self._ids = url, call_timeout_s, itertools.count(1)

    def _request(self, method: str, **args: Any) -> dict[str, Any]:
        rid = next(self._ids)
        try:
            self._conn.send(json.dumps({"id": rid, "method": method, "args": codec.encode(args)}))
            reply = json.loads(self._conn.recv(timeout=self._timeout))
        except Exception as exc:  # noqa: BLE001 - surface transport failures as component faults
            raise self.fault(f"{method} over {self.url} failed: {exc!r}") from exc
        if reply["id"] != rid:
            raise self.fault(f"out-of-order reply {reply['id']} != {rid}")
        return reply

    def _disconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


class WSEnvironment(_Client):
    """Environment proxy for a remote `ws.py env` server."""

    protocol = EnvHandler.protocol
    fault = EnvironmentFault

    def __init__(self, url: str, **kw: Any) -> None:
        super().__init__(url, **kw)
        self._spec: EnvSpec | None = None

    def _call(self, method: str, **args: Any) -> Any:
        reply = self._request(method, **args)
        if not reply["ok"]:
            raise EnvironmentFault(f"remote {method} failed: {reply['error']}")
        return reply["result"]

    def spec(self) -> EnvSpec:
        if self._spec is None:
            self._spec = EnvSpec.model_validate(self._call("spec"))
        return self._spec

    def reset(self, task: TaskInstance) -> Observation:
        return Observation.model_validate(self._call("reset", task=task))

    def step(self, action) -> StepResult:
        return StepResult.model_validate(self._call("step", action=action))

    def privileged_state(self) -> dict[str, Any]:
        return codec.decode(self._call("privileged_state"))

    def close(self) -> None:
        try:
            self._call("close")
        except EnvironmentFault:
            pass
        self._disconnect()


class WSAgent(_Client):
    """Agent proxy for a remote `ws.py agent` server; relays the agent's trace events like RemoteAgent."""

    protocol = AgentHandler.protocol
    fault = AgentError

    def __init__(self, url: str, forward_steps: bool = True, **kw: Any) -> None:
        super().__init__(url, **kw)
        self.forward_steps = forward_steps
        self._spec: AgentSpec | None = None
        self._ctx: EpisodeContext | None = None

    def _call(self, method: str, **args: Any) -> Any:
        reply = self._request(method, **args)
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
        try:
            self._call("close")
        except AgentError:
            pass
        self._disconnect()


# ---------------------------------------------------------------- entry point

def make_server(kind: str, ref: str, params: dict[str, Any] | None = None, **kw: Any) -> ComponentServer:
    if kind == "env":
        factory = lambda: EnvHandler(load_component(ref, params, "envs"))  # noqa: E731
    elif kind == "agent":
        factory = lambda: AgentHandler(load_component(ref, params, "agents"))  # noqa: E731
    else:
        raise ValueError(f"kind must be 'env' or 'agent', not {kind!r}")
    return ComponentServer(factory, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hylo.core.remote.ws", description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["env", "agent"])
    ap.add_argument("ref", help="plugin name (e.g. libero) or module:Class")
    ap.add_argument("--params", default="{}", help="JSON constructor params")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 exposes it on the network)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token-env", help="name of an environment variable holding the shared bearer token")
    ap.add_argument("--max-connections", type=int, default=64)
    a = ap.parse_args(argv)
    token = os.environ.get(a.token_env) if a.token_env else None
    if a.token_env and not token:
        ap.error(f"${a.token_env} is empty")
    if a.host not in ("127.0.0.1", "localhost", "::1") and token is None:
        print("[ws] warning: listening on a non-loopback address without a token", file=sys.stderr)
    make_server(a.kind, a.ref, json.loads(a.params), host=a.host, port=a.port, token=token,
                max_connections=a.max_connections).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
