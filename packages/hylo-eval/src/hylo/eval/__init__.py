"""Hylo benchmark layer."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"benchmarks": "hylo.benchmark/v0.1", "scorers": "hylo.scorer/v0.1", "gate_tests": "hylo.gate_test/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "gate_tests/paired_permutation": {"maturity": "unverified", "notes": "not yet run in a campaign"},
        "gate_tests/sign": {"maturity": "unverified", "notes": "not yet run in a campaign"},
    },
}
