"""Hylo benchmark layer."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"benchmarks": "hylo.benchmark/v0.1", "scorers": "hylo.scorer/v0.1", "gate_tests": "hylo.gate_test/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "gate_tests/paired_permutation": {"notes": "gated on steps in a MuJoCo planar reach campaign (maturity-20261101)"},
        "gate_tests/sign": {"notes": "gated on steps in a MuJoCo planar reach campaign (maturity-20261101)"},
    },
}
