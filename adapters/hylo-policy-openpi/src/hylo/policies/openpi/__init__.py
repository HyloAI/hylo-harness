# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"agents": "hylo.agent/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "agents/openpi": {
            "maturity": "unverified",
            "notes": "needs an openpi policy server and its checkpoint",
            "weights": "PaliGemma-based checkpoints: Gemma Terms of Use (openpi ships LICENSE_GEMMA.txt)",
        },
    },
}
