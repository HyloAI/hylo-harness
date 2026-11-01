"""Contract versions and plugin metadata (RFC-0005 part 4).

Every plugin group is a contract with a version (`wc.policy/v0.1`, `wc.agent/v0.1`, ...). A distribution declares
which version of each contract its plugins implement, and what is known about them, in one dictionary exposed under
the `worldcurrent.meta` entry-point group (a module with no heavy imports, so listing plugins never loads them):

    [project.entry-points."worldcurrent.meta"]
    wc-policy-lerobot = "worldcurrent.policies.lerobot:META"

    META = {
        "contracts": {"agents": "wc.agent/v0.1", "policies": "wc.policy/v0.1"},
        "maturity": "implemented",                                  # the default for this distribution's plugins
        "plugins": {"agents/lerobot.steered": {"weights": "per checkpoint: ..."},
                    "datasources/lerobot.dataset": {"maturity": "unverified", "notes": "..."}},
    }

The code licence is the distribution's own (`license` in its pyproject); `weights` names the licence of model
weights a plugin downloads or needs, which is often different. Maturity:

    implemented     run end to end on its intended target (a real simulator or robot), with recorded runs
    unverified      tests pass, but never run end to end on its intended target here
    needs-hardware  needs hardware this lab has not connected yet
    test-double     for tests and conformance only (`testing.*` names)

The registry refuses to load a plugin whose declared contract is incompatible with this core: another major
version, or a newer minor version (a plugin written against features this core does not have). A plugin without a
declaration still loads; `wc plugins` shows it as undeclared and `wc plugin check` warns.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal

from .contract import Contract

CONTRACTS: dict[str, str] = {
    "envs": "wc.env/v0.1", "agents": "wc.agent/v0.1", "policies": "wc.policy/v0.1", "reasoners": "wc.reasoner/v0.1",
    "memory": "wc.memory/v0.1", "scorers": "wc.scorer/v0.1", "benchmarks": "wc.benchmark/v0.1",
    "generators": "wc.generator/v0.1", "learners": "wc.learner/v0.1", "instruments": "wc.instrument/v0.1",
    "datasources": "wc.datasource/v0.1", "guards": "wc.guard/v0.1", "gate_tests": "wc.gate_test/v0.1",
}
MATURITY = ("implemented", "unverified", "needs-hardware", "test-double")
Maturity = Literal["implemented", "unverified", "needs-hardware", "test-double", "undeclared"]
_VERSION = re.compile(r"^(?P<name>wc\.[a-z_]+)/v(?P<major>\d+)\.(?P<minor>\d+)$")


class IncompatiblePlugin(RuntimeError):
    """A plugin declares a contract version this core cannot run."""


class PluginMeta(Contract):
    group: str
    name: str
    target: str                               # module:attr
    distribution: str | None = None
    version: str | None = None                # the distribution's version
    contract: str | None = None               # declared, e.g. wc.policy/v0.1; None = undeclared
    expected: str | None = None               # what this core implements
    compatible: bool | None = None            # None when undeclared
    maturity: Maturity = "undeclared"
    licence: str | None = None                # the code's licence (the distribution's)
    weights: str | None = None                # the licence of weights the plugin needs, when it needs any
    notes: str | None = None


def parse(version: str) -> tuple[str, int, int]:
    m = _VERSION.match(version)
    if m is None:
        raise ValueError(f"{version!r} is not a contract version (wc.<name>/v<major>.<minor>)")
    return m["name"], int(m["major"]), int(m["minor"])


def compatible(declared: str, current: str) -> bool:
    """Same contract and major version, and not newer than this core's minor version."""
    dn, dmaj, dmin = parse(declared)
    cn, cmaj, cmin = parse(current)
    return dn == cn and dmaj == cmaj and dmin <= cmin


@lru_cache(maxsize=None)
def declarations() -> dict[str, dict[str, Any]]:
    """{distribution: META} from the worldcurrent.meta entry points."""
    out: dict[str, dict[str, Any]] = {}
    for ep in entry_points(group="worldcurrent.meta"):
        dist = ep.dist.name if ep.dist else ep.name
        try:
            meta = ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken declaration is reported, not fatal for listing
            meta = {"error": f"{type(exc).__name__}: {exc}"}
        out[dist] = dict(meta) if isinstance(meta, dict) else {"error": "META is not a dict"}
    return out


def _licence(ep: EntryPoint) -> str | None:
    if ep.dist is None:
        return None
    md = ep.dist.metadata
    return md.get("License-Expression") or md.get("License") or None


def describe(group: str, ep: EntryPoint) -> PluginMeta:
    dist = ep.dist.name if ep.dist else None
    decl = declarations().get(dist or "", {})
    per = dict((decl.get("plugins") or {}).get(f"{group}/{ep.name}") or {})
    contract = per.get("contract") or (decl.get("contracts") or {}).get(group)
    expected = CONTRACTS.get(group)
    ok = None
    if contract and expected:
        try:
            ok = compatible(contract, expected)
        except ValueError:
            ok = False
    if ep.name.startswith(("testing.", "testing/")):
        maturity = "test-double"
    else:
        maturity = per.get("maturity") or decl.get("maturity") or "undeclared"
    return PluginMeta(group=group, name=ep.name, target=ep.value, distribution=dist,
                      version=ep.dist.version if ep.dist else None, contract=contract, expected=expected,
                      compatible=ok, maturity=maturity, licence=per.get("licence") or _licence(ep),
                      weights=per.get("weights"), notes=per.get("notes") or decl.get("error"))


def check(group: str, ep: EntryPoint) -> PluginMeta:
    """Raise IncompatiblePlugin when the plugin declares a contract version this core cannot run."""
    meta = describe(group, ep)
    if meta.compatible is False:
        raise IncompatiblePlugin(f"{group} plugin {ep.name!r} ({meta.distribution} {meta.version}) implements "
                                 f"{meta.contract}; this core implements {meta.expected}")
    return meta


def catalogue(groups: tuple[str, ...] | None = None) -> list[PluginMeta]:
    """Every installed plugin with its metadata, without importing any plugin."""
    out = []
    for group in groups or tuple(CONTRACTS):
        for ep in sorted(entry_points(group=f"worldcurrent.{group}"), key=lambda e: e.name):
            out.append(describe(group, ep))
    return out


__all__ = ["CONTRACTS", "MATURITY", "IncompatiblePlugin", "PluginMeta", "catalogue", "check", "compatible",
           "declarations", "describe", "parse"]
