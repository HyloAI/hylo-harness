"""`hylo mcp`: any registered tool surface (group `hylo.surfaces`) as a Model Context Protocol server over stdio, so a
coding agent (Claude Code, Codex, ...) can build and run robot worlds with the same tools a Hylo session uses.

    hylo mcp --surface mujoco.world --param root=runs/stages --param stage=lab

    # e.g. in a coding agent's MCP configuration:
    {"command": "uv", "args": ["run", "hylo", "mcp", "--surface", "mujoco.world", "--param", "stage=lab"]}

JSON-RPC 2.0, one message per line (the MCP stdio transport). Implemented methods: initialize, ping, tools/list,
tools/call (results as text content with the tool's JSON, plus PNG image content for images; a tool error is a
result with isError true), and the initialized notification. Every call is appended to `--log` (JSON lines) when
given. Standard library only, besides the surface itself.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from typing import Any, Sequence, TextIO

from hylo.core.registry import registry
from hylo.core.tooling import Surface

PROTOCOL = "2025-06-18"


def _png(image: Any) -> str:
    from hylo.reasoners.llm.chat import png_bytes

    return base64.b64encode(png_bytes(image)).decode()


class MCPServer:
    def __init__(self, surface: Surface, name: str = "hylo", log: str | None = None) -> None:
        self.surface, self.name, self.log = surface, name, log

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:                                   # a notification: nothing to answer
            return None
        try:
            result = self._dispatch(method, params)
        except KeyError as exc:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {exc}"}}
        except Exception as exc:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": "0.1"},
                    "instructions": f"Hylo tool surface {self.surface.name} ({self.surface.digest()}): tools that "
                                    "only read are marked readOnlyHint; the others change the stage or the world."}
        if method == "ping":
            return {}
        if method == "tools/list":
            tools = []
            for spec in self.surface.specs():
                kind = self.surface.kind(spec["name"])
                tools.append({"name": spec["name"], "description": spec["description"],
                              "inputSchema": spec["parameters"],
                              "annotations": {"readOnlyHint": kind == "read", "destructiveHint": False}})
            return {"tools": tools}
        if method == "tools/call":
            t0 = time.monotonic()
            r = self.surface.call(params["name"], params.get("arguments") or {})
            content: list[dict[str, Any]] = [{"type": "text", "text": r.text()}]
            content += [{"type": "image", "data": _png(im), "mimeType": "image/png"} for im in r.images]
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"time": round(time.time(), 3), "tool": params["name"],
                                        "arguments": params.get("arguments"), "ok": r.ok,
                                        "seconds": round(time.monotonic() - t0, 3), "result": r.text()[:4000]}) + "\n")
            return {"content": content, "isError": not r.ok}
        raise KeyError(method)

    def serve(self, rd: TextIO = sys.stdin, wr: TextIO = sys.stdout) -> None:
        for line in rd:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                out = self.handle(msg)
            if out is not None:
                wr.write(json.dumps(out) + "\n")
                wr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """serve a tool surface (e.g. mujoco.world) over MCP on stdio"""
    ap = argparse.ArgumentParser(prog="hylo mcp", description="serve a tool surface over MCP (stdio)")
    ap.add_argument("--surface", default="mujoco.world")
    ap.add_argument("--param", action="append", default=[], help="key=value for the surface (repeatable)")
    ap.add_argument("--log", help="append every tool call to this JSON-lines file")
    a = ap.parse_args(argv)
    params = dict(p.split("=", 1) for p in a.param)
    surface = registry.build("surfaces", {"type": a.surface, "params": params})
    MCPServer(surface, f"hylo-{a.surface}", a.log).serve()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["MCPServer", "main"]
