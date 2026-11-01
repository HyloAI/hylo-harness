"""Rule-based reasoners and choosers (the Reasoner / Chooser contracts are hylo.core.reasoning).

`hylo.reasoners` is a namespace shared with hylo-reasoner-llm (hylo.reasoners.llm); this distribution owns
hylo.reasoners.rules only.
"""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "reasoners": "hylo.reasoner/v0.1",
        "choosers": "hylo.chooser/v0.1",
    },
    "maturity": "implemented",
    "plugins": {},
}
