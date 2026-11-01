"""Execution profiles (wc.profile/v0.1): where components run on this machine, kept out of experiment configs.

An experiment says *what* is measured. Where an interpreter lives, which rendering library to load, where the model
cache is, differ per machine and must not change an experiment's identity. Configs name a runtime instead:

    env:   {type: libero, remote: {runtime: libero}}
    agent: {type: remote, params: {ref: lerobot.steered, runtime: libero, params: {...}}}

and each machine's profile says what that runtime is:

    schema_version: wc.profile/v0.1
    runtimes:
      libero:
        python: /home/me/worldcurrent/bench/envs/libero/.venv/bin/python
        env: {MUJOCO_GL: osmesa, LIBERO_CONFIG_PATH: /home/me/worldcurrent/bench/envs/libero/config, HF_HOME: ...}

The profile is found at $WC_PROFILE, else ~/.config/worldcurrent/profile.yaml. `wc profile` shows and edits it;
bench's envs/libero/setup.sh registers the runtime it builds. Explicit `python` / `env` in a config still work and
take precedence over the runtime's (for a one-off); neither enters the experiment digest. The profile also says
where this machine's run and artifact store is (`store:`, set by `wc store init PATH`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, Sequence

import yaml
from pydantic import Field

from .contract import Contract

PROFILE_SCHEMA = "wc.profile/v0.1"


class Runtime(Contract):
    python: str | None = None                      # interpreter of the runtime's venv
    env: dict[str, str] = Field(default_factory=dict)


class Profile(Contract):
    schema_version: Literal["wc.profile/v0.1"] = PROFILE_SCHEMA
    runtimes: dict[str, Runtime] = Field(default_factory=dict)
    store: str | None = None                       # this machine's run and artifact store (core.store)


def profile_path() -> Path:
    return Path(os.environ.get("WC_PROFILE") or Path.home() / ".config" / "worldcurrent" / "profile.yaml").expanduser()


def load_profile(path: str | Path | None = None) -> Profile:
    p = Path(path) if path else profile_path()
    if not p.exists():
        return Profile()
    return Profile.model_validate(yaml.safe_load(p.read_text()) or {})


def save_profile(profile: Profile, path: str | Path | None = None) -> Path:
    p = Path(path) if path else profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))
    return p


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def resolve(runtime: str | None = None, python: str | None = None,
            env: dict[str, str] | None = None) -> tuple[str | None, dict[str, str]]:
    """The interpreter and environment variables to start a component with: the named runtime from this machine's
    profile, overridden by explicit values."""
    base = Runtime()
    if runtime:
        prof = load_profile()
        if runtime not in prof.runtimes:
            raise KeyError(f"runtime {runtime!r} is not in {profile_path()} (known: {sorted(prof.runtimes) or '-'}); "
                           f"register it with: wc profile set-runtime {runtime} --python PATH [--env KEY=VALUE ...]")
        base = prof.runtimes[runtime]
    py = python or base.python
    merged = {k: _expand(v) for k, v in {**base.env, **(env or {})}.items()}
    return (_expand(py) if py else None), merged


def main(argv: Sequence[str] | None = None) -> int:
    """show or edit this machine's execution profile (runtimes)"""
    ap = argparse.ArgumentParser(prog="wc profile", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("show", help="print the profile and where it is")
    sr = sub.add_parser("set-runtime", help="add or replace a runtime")
    sr.add_argument("name")
    sr.add_argument("--python", required=True)
    sr.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    rm = sub.add_parser("remove-runtime")
    rm.add_argument("name")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    prof = load_profile()
    if a.action == "show":
        print(f"# {profile_path()}")
        print(yaml.safe_dump(prof.model_dump(mode="json"), sort_keys=False), end="")
        return 0
    runtimes = dict(prof.runtimes)
    if a.action == "set-runtime":
        env = dict(kv.split("=", 1) for kv in a.env)
        runtimes[a.name] = Runtime(python=os.path.abspath(os.path.expanduser(a.python)), env=env)   # keep venv symlinks
    else:
        runtimes.pop(a.name, None)
    path = save_profile(prof.model_copy(update={"runtimes": runtimes}))
    print(json.dumps({"profile": str(path), "runtimes": sorted(runtimes)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PROFILE_SCHEMA", "Profile", "Runtime", "load_profile", "profile_path", "resolve", "save_profile"]
