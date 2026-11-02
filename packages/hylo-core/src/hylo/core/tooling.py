"""Typed tool surfaces (hylo.tools/v0.1): what a tool-calling model may call, declared once and checked on every call.

A `Surface` is an ordered set of `ToolDef`s. Each tool has

```text
  name         lower_snake_case, unique in the surface
  kind         read  (looks, measures, computes: never changes the world; several may share one model reply)
               act   (changes the world: must be the only call of its reply; ends the model's turn)
               end   (ends the episode or session)
  help         purpose, when to use it, what it changes, what it returns, what it does not do, its errors:
               rendered into one description, so every tool explains itself the same way
  params       a closed JSON Schema object (no unknown keys; every field described); optional fields also accept
               null, which means "not given"
  fn           fn(args) -> dict (the result) or ToolOutput (a result with images); raises ToolError
```

`Surface.call(name, args)` checks the arguments against the schema (types, enums, bounds, required fields, unknown
keys), runs the tool and returns a `ToolReply` (`ok`, `result` or `error`, `images`). A failure is a structured
`ToolError` (code, message, field, whether retrying with the same arguments can help) that goes back to the model as
data, never an exception out of the agent loop. `Surface.digest()` hashes the declarations as the model sees them,
so a run records exactly which surface it had.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

TOOLS_SCHEMA = "hylo.tools/v0.1"
Kind = Literal["read", "act", "end"]
NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class ToolError(Exception):
    """A tool call that failed in a way the model should read: `code` (machine-readable), `message` (what went wrong
    and what to do), `field` (the argument at fault), `retry` ("never" | "changed_arguments" | "later")."""

    def __init__(self, code: str, message: str, *, field: str | None = None,
                 retry: Literal["never", "changed_arguments", "later"] = "changed_arguments",
                 detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.field, self.retry = code, message, field, retry
        self.detail = dict(detail or {})

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message, "retry": self.retry}
        if self.field:
            d["field"] = self.field
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class ToolOutput:
    result: dict[str, Any]
    images: list[Any] = field(default_factory=list)       # (H, W, 3) uint8 arrays, shown to vision models
    image_notes: list[str] = field(default_factory=list)


@dataclass
class ToolReply:
    name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    images: list[Any] = field(default_factory=list)
    image_notes: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        """What the model reads (images travel separately)."""
        body = {"ok": True, **(self.result or {})} if self.ok else {"ok": False, "error": self.error}
        if self.images:
            body["images"] = self.image_notes or [f"image {i + 1}" for i in range(len(self.images))]
        return body

    def text(self) -> str:
        return json.dumps(jsonable(self.payload()), separators=(",", ":"))


@dataclass
class Help:
    purpose: str
    use_when: str = ""
    effect: str = "none: reads only"
    returns: str = ""
    limits: str = ""
    errors: str = ""

    def render(self) -> str:
        parts = [self.purpose.strip()]
        for label, text in (("Use when", self.use_when), ("Effect", self.effect), ("Returns", self.returns),
                            ("Does not", self.limits), ("Errors", self.errors)):
            if text.strip():
                parts.append(f"{label}: {text.strip()}")
        return "\n".join(parts)


@dataclass
class ToolDef:
    name: str
    kind: Kind
    help: Help
    params: dict[str, Any]
    fn: Callable[[dict[str, Any]], Any]

    def __post_init__(self) -> None:
        if not NAME.match(self.name):
            raise ValueError(f"tool name {self.name!r} must be lower_snake_case")
        if self.params.get("type") != "object":
            raise ValueError(f"tool {self.name}: params must be a JSON Schema object")
        self.params = closed(self.params)

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.help.render(), "parameters": self.params}


def obj(properties: Mapping[str, Any] | None = None, required: Sequence[str] = ()) -> dict[str, Any]:
    """A closed object schema."""
    return {"type": "object", "properties": dict(properties or {}), "required": list(required),
            "additionalProperties": False}


def closed(schema: Mapping[str, Any]) -> dict[str, Any]:
    """`schema` with every object closed and every optional property also accepting null."""
    s = dict(schema)
    if s.get("type") == "object" or "properties" in s:
        props = {}
        req = set(s.get("required") or ())
        for k, v in (s.get("properties") or {}).items():
            v = closed(v)
            if "description" not in v:
                raise ValueError(f"property {k!r} needs a description")
            if k not in req:
                v = _nullable(v)
            props[k] = v
        s["properties"] = props
        s.setdefault("additionalProperties", False)
        s.setdefault("required", [])
    if s.get("type") == "array" and isinstance(s.get("items"), Mapping):
        it = dict(s["items"])
        it.setdefault("description", "item")
        s["items"] = closed(it)
    return s


def _nullable(v: dict[str, Any]) -> dict[str, Any]:
    t = v.get("type")
    if isinstance(t, str) and t != "null":
        v = {**v, "type": [t, "null"]}
        if "enum" in v and None not in v["enum"]:
            v["enum"] = [*v["enum"], None]
    return v


# ------------------------------------------------------------------------------------------------ argument checks

_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def check(schema: Mapping[str, Any], value: Any, path: str = "") -> list[tuple[str, str]]:
    """Problems of `value` against a closed schema subset: [(field path, message)]."""
    where = path or "arguments"
    types = schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    if value is None:
        return [] if (not types or "null" in types) else [(where, "is required (got null)")]
    if types:
        ok = False
        for t in types:
            if t == "null":
                continue
            py = _PY.get(t)
            if py is None:
                continue
            if t in ("integer", "number") and isinstance(value, bool):
                continue
            if t == "integer" and isinstance(value, float) and value.is_integer():
                ok = True
                break
            if isinstance(value, py):
                ok = True
                break
        if not ok:
            return [(where, f"must be {' or '.join(t for t in types if t != 'null')}, got {type(value).__name__}")]
    out: list[tuple[str, str]] = []
    if "enum" in schema and value not in schema["enum"]:
        out.append((where, f"must be one of {[e for e in schema['enum'] if e is not None]}, got {value!r}"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            out.append((where, "must be finite"))
        if "minimum" in schema and value < schema["minimum"]:
            out.append((where, f"must be >= {schema['minimum']}, got {value}"))
        if "maximum" in schema and value > schema["maximum"]:
            out.append((where, f"must be <= {schema['maximum']}, got {value}"))
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append((where, f"must be at most {schema['maxLength']} characters"))
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append((where, f"must be at least {schema['minLength']} characters"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append((where, f"needs at least {schema['minItems']} items, got {len(value)}"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append((where, f"takes at most {schema['maxItems']} items, got {len(value)}"))
        if isinstance(schema.get("items"), Mapping):
            for i, item in enumerate(value):
                out += check(schema["items"], item, f"{path}[{i}]" if path else f"[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or ():
            if value.get(k) is None:
                out.append((f"{path}.{k}" if path else k, "is required"))
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    out.append((f"{path}.{k}" if path else k, f"is not an argument; known: {sorted(props)}"))
        for k, sub in props.items():
            if k in value:
                out += check(sub, value[k], f"{path}.{k}" if path else k)
    return out


def jsonable(x: Any) -> Any:
    """Plain JSON values: numpy arrays and scalars become lists and numbers (floats rounded to 6 digits)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if np is not None and isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------------ surface


class Surface:
    def __init__(self, tools: Iterable[ToolDef] = (), name: str = "surface") -> None:
        self.name = name
        self._tools: dict[str, ToolDef] = {}
        for t in tools:
            self.add(t)

    def add(self, tool: ToolDef) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} declared twice")
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> ToolDef:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def kind(self, name: str) -> Kind | None:
        t = self._tools.get(name)
        return t.kind if t else None

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def digest(self) -> str:
        blob = json.dumps({"schema": TOOLS_SCHEMA, "tools": self.specs()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def manual(self) -> str:
        """All tools as text (for documents and for models without native tool calls)."""
        out = []
        for t in self._tools.values():
            out.append(f"## {t.name} ({t.kind})\n{t.help.render()}\nArguments: {json.dumps(t.params)}")
        return "\n\n".join(out)

    def call(self, name: str, args: Mapping[str, Any] | None) -> ToolReply:
        tool = self._tools.get(name)
        if tool is None:
            return ToolReply(name, False, error=ToolError("unknown_tool", f"there is no tool {name!r}; tools: "
                                                          f"{self.names()}", retry="never").as_dict())
        args = dict(args or {})
        problems = check(tool.params, args)
        if problems:
            fld, _ = problems[0]
            msg = "; ".join(f"{p}: {m}" for p, m in problems[:6])
            return ToolReply(name, False, error=ToolError("bad_arguments", msg, field=fld).as_dict())
        args = {k: v for k, v in args.items() if v is not None}
        try:
            out = tool.fn(args)
        except ToolError as exc:
            return ToolReply(name, False, error=exc.as_dict())
        except Exception as exc:  # noqa: BLE001 - a broken tool is reported to the model, never raised into the loop
            return ToolReply(name, False, error=ToolError("tool_failed", f"{type(exc).__name__}: {exc}"[:500],
                                                          retry="later").as_dict())
        if isinstance(out, ToolOutput):
            return ToolReply(name, True, result=jsonable(out.result), images=list(out.images),
                             image_notes=list(out.image_notes))
        return ToolReply(name, True, result=jsonable(dict(out or {})))


__all__ = ["Help", "Kind", "Surface", "TOOLS_SCHEMA", "ToolDef", "ToolError", "ToolOutput", "ToolReply", "check",
           "closed", "jsonable", "obj"]
