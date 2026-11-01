"""worldcurrent benchmark layer."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"benchmarks": "wc.benchmark/v0.1", "scorers": "wc.scorer/v0.1", "gate_tests": "wc.gate_test/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "gate_tests/paired_permutation": {"maturity": "unverified", "notes": "not yet run in a campaign"},
        "gate_tests/sign": {"maturity": "unverified", "notes": "not yet run in a campaign"},
    },
}
