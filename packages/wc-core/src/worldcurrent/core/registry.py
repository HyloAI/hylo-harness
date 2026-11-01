"""Plugin registry: explicit registration, `module:attr` references (vla-eval style) and
`worldcurrent.<group>` entry points (inspect-robots style, with autoload switchable off)."""

from __future__ import annotations

import importlib
import os
from importlib.metadata import entry_points
from typing import Any, Callable

GROUPS = ("envs", "agents", "policies", "reasoners", "choosers", "planners", "reflectors", "memory", "scorers",
          "benchmarks", "generators", "learners", "instruments", "datasources", "guards", "gate_tests")


class Registry:
    def __init__(self, *, autoload: bool | None = None) -> None:
        self._items: dict[str, dict[str, Any]] = {g: {} for g in GROUPS}
        if autoload is None:
            autoload = os.environ.get("WORLDCURRENT_NO_PLUGIN_AUTOLOAD") != "1"
        self._autoloaded = not autoload

    def register(self, group: str, name: str) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            self._items[group][name] = obj
            return obj
        return deco

    def register_lazy(self, group: str, name: str, ref: str) -> None:
        """Register a `module:attr` reference without importing it (keeps heavy deps optional)."""
        self._items[group][name] = ref

    def _autoload(self) -> None:
        if self._autoloaded:
            return
        self._autoloaded = True
        for group in GROUPS:
            for ep in entry_points(group=f"worldcurrent.{group}"):
                self._items[group].setdefault(ep.name, ep)  # resolved lazily

    def resolve(self, group: str, name_or_ref: str) -> Any:
        if ":" in name_or_ref:
            module, _, attr = name_or_ref.partition(":")
            obj: Any = importlib.import_module(module)
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        self._autoload()
        try:
            obj = self._items[group][name_or_ref]
        except KeyError:
            known = sorted(self._items[group])
            raise KeyError(f"no {group} plugin named {name_or_ref!r}; known: {known}") from None
        if isinstance(obj, str):
            obj = self.resolve(group, obj)
            self._items[group][name_or_ref] = obj
        elif hasattr(obj, "load") and hasattr(obj, "group"):  # EntryPoint
            from .plugins import check

            check(group, obj)                                 # declared contract version must be compatible
            obj = obj.load()
            self._items[group][name_or_ref] = obj
        return obj

    def reference(self, group: str, name_or_ref: str) -> str:
        """`module:attr` for a plugin without importing it (for components that run in another venv)."""
        if ":" in name_or_ref:
            return name_or_ref
        self._autoload()
        obj = self._items[group].get(name_or_ref)
        if obj is None:
            raise KeyError(f"no {group} plugin named {name_or_ref!r}; known: {sorted(self._items[group])}")
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "value") and hasattr(obj, "group"):  # EntryPoint
            return obj.value
        return f"{obj.__module__}:{obj.__qualname__}"

    def build(self, group: str, spec: dict[str, Any]) -> Any:
        """spec = {"type": "<name or module:attr>", "params": {...}}; store ids in params (artifact:<name>@<digest>,
        run:<id>) become this machine's paths here, after the experiment digest was taken over the ids."""
        from .store import resolve_refs

        factory = self.resolve(group, spec["type"])
        return factory(**resolve_refs(dict(spec.get("params") or {})))


registry = Registry()
