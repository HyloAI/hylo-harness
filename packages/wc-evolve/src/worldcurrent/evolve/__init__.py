"""worldcurrent evolution layer."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"generators": "wc.generator/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "generators/critic.chunk_value": {"maturity": "unverified", "notes": "not yet run in a campaign"},
        "generators/experience": {"maturity": "unverified", "notes": "not yet run in a campaign"},
        "generators/learn": {"maturity": "unverified", "notes": "not yet run in a campaign"},
        "generators/selftrain.expert_iteration": {"maturity": "unverified", "notes": "smoke run only"},
    },
}
