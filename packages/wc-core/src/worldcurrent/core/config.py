"""Versioned experiment configuration and run manifest (design/experiment-schema.md)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from .contract import Contract
from .kernel import Timing

SCHEMA_VERSION = "wc.experiment/v0.1"


class ComponentRef(Contract):
    type: str                                   # plugin name or "module:attr"
    params: dict[str, Any] = Field(default_factory=dict)


class RemoteSpec(Contract):
    """Where a component runs: never part of what is measured. `runtime` names a venv in this machine's profile
    (core.profile); `python` / `env` set or override it directly."""

    runtime: str | None = None
    python: str | None = None                   # interpreter of the component's own venv
    env: dict[str, str] = Field(default_factory=dict)   # extra environment variables for the subprocess


class EnvRef(ComponentRef):
    remote: RemoteSpec | None = None


class BenchmarkRef(ComponentRef):
    split: str = "validation"


class TraceOptions(Contract):
    record_privileged: bool = True
    image_every: int = Field(default=1, ge=0)   # store camera images every N steps (0 = never); state is always stored


class ExperimentConfig(Contract):
    schema_version: Literal["wc.experiment/v0.1"] = SCHEMA_VERSION
    name: str
    benchmark: BenchmarkRef
    env: EnvRef
    agent: ComponentRef
    seed: int = 0
    safety: dict[str, Any] = Field(default_factory=dict)
    trace: TraceOptions = Field(default_factory=TraceOptions)
    timing: Timing = Field(default_factory=Timing)  # sync (default) or realtime: part of what is measured
    output_dir: str = "runs"
    workers: int = Field(default=1, ge=1)        # parallel episode workers; does not change what is measured
    tags: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def measured(self) -> dict[str, Any]:
        """What the experiment measures. Left out: where results go, how many workers, and where components run
        (env.remote, a remote agent's runtime / python / env), so the same experiment has the same digest on every
        machine. Timing is left out while it is the default (sync), so older digests stay valid."""
        d = self.model_dump(mode="json", exclude={"output_dir", "workers"})
        if "remote" in d["env"]:
            d["env"]["remote"] = None                  # hashed as a local environment, as digests always were
        if d.get("timing") == Timing().model_dump(mode="json"):
            d.pop("timing")
        _strip_placement(d["agent"])
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.measured(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def with_overrides(self, patch: dict[str, Any]) -> "ExperimentConfig":
        """Apply dotted-path overrides, e.g. {"agent.params.kp": 2.0}; the result is re-validated."""
        d = self.model_dump(mode="json")
        for path, value in patch.items():
            node = d
            *parents, leaf = path.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return ExperimentConfig.model_validate(d)


REMOTE_AGENTS = ("remote",)                     # agent plugins whose runtime / python / env only say where it runs
PLACEMENT = ("runtime", "python", "env")


def _strip_placement(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") in REMOTE_AGENTS and isinstance(node.get("params"), dict):
            for k in PLACEMENT:
                node["params"].pop(k, None)
        for v in node.values():
            _strip_placement(v)
    elif isinstance(node, list):
        for v in node:
            _strip_placement(v)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


WORLDCURRENT_PACKAGES = ("wc-core", "wc-eval", "wc-agents", "wc-memory", "wc-evolve", "wc-env-mujoco",
                         "worldcurrent-cli")


def build_manifest(cfg: ExperimentConfig, *, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from .events import TRACE_SCHEMA
    from .store import Store, has_refs

    config = json.loads(cfg.canonical_json())
    pinned = {"artifacts": Store().pin(config)} if has_refs(config) else {}   # ids -> full id and path on this machine
    return {
        "run_id": run_id,
        "schema_version": cfg.schema_version,
        "trace_schema": TRACE_SCHEMA,
        "config_digest": cfg.digest(),
        "config": config,
        **pinned,
        "created_at": time.time(),
        "code": {"git_sha": _git_sha()},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "packages": {p: _version(p) for p in ("numpy", "pydantic", "mujoco", *WORLDCURRENT_PACKAGES)}},
        **(extra or {}),
    }
