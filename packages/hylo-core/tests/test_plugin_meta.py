"""RFC-0005 part 4: contract versions and plugin metadata."""

from importlib.metadata import entry_points

import pytest

import hylo.core.plugins as plugins
from hylo.core.plugins import CONTRACT_OBJECTS, CONTRACTS, MATURITY, IncompatiblePlugin, catalogue, compatible
from hylo.core.registry import GROUPS, Registry


def test_every_plugin_group_is_a_versioned_contract_defined_in_core():
    import importlib

    assert set(CONTRACTS) == set(GROUPS) == set(CONTRACT_OBJECTS)
    for group, ref in CONTRACT_OBJECTS.items():
        module, _, name = ref.partition(":")
        assert module.startswith("hylo.core.") and hasattr(importlib.import_module(module), name), group
    assert compatible("hylo.policy/v0.1", "hylo.policy/v0.1")
    assert compatible("hylo.policy/v1.0", "hylo.policy/v1.3")              # older minor: still runs
    assert not compatible("hylo.policy/v1.4", "hylo.policy/v1.3")          # needs features this core lacks
    assert not compatible("hylo.policy/v0.1", "hylo.policy/v1.0")          # another major
    assert not compatible("hylo.agent/v0.1", "hylo.policy/v0.1")


def test_installed_plugins_declare_a_compatible_contract_and_a_maturity():
    metas = catalogue()
    assert metas, "no plugins installed"
    for m in metas:
        assert m.contract and m.compatible, f"{m.distribution}: {m.group}/{m.name} declares {m.contract}"
        assert m.maturity in MATURITY, f"{m.group}/{m.name}: {m.maturity}"
        assert m.licence, f"{m.distribution} has no licence"
    assert {m.maturity for m in metas if m.name.startswith("testing.")} == {"test-double"}


def test_the_registry_refuses_an_incompatible_contract_before_importing(monkeypatch):
    ep = next(e for e in entry_points(group="hylo.gate_tests") if e.name == "mcnemar")
    monkeypatch.setattr(plugins, "declarations", lambda: {ep.dist.name: {"contracts": {"gate_tests": "hylo.gate_test/v1.0"}}})
    with pytest.raises(IncompatiblePlugin, match="implements hylo.gate_test/v1.0; this core implements hylo.gate_test/v0.1"):
        Registry().resolve("gate_tests", "mcnemar")
    monkeypatch.setattr(plugins, "declarations", lambda: {})                # undeclared: loads, listed as such
    assert Registry().resolve("gate_tests", "mcnemar").__name__ == "McNemar"
    assert plugins.describe("gate_tests", ep).maturity == "undeclared"
