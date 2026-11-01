"""worldcurrent intelligence layer: agents, reasoners, policies."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {
        "envs": "wc.env/v0.1",
        "agents": "wc.agent/v0.1",
        "reasoners": "wc.reasoner/v0.1",
        "choosers": "wc.chooser/v0.1",
        "planners": "wc.planner/v0.1",
        "instruments": "wc.instrument/v0.1",
    },
    "maturity": "implemented",
    "plugins": {
        "agents/planner.subgoal": {"maturity": "unverified"},
        "agents/shadow": {"maturity": "unverified"},
        "agents/skills": {"maturity": "unverified"},
        "agents/memory.steered": {
            "maturity": "unverified",
            "notes": "the LeRobot variant (lerobot.memory_steered) has run",
        },
        "agents/noise.steered": {
            "maturity": "unverified",
            "notes": "the LeRobot variant (lerobot.noise_steered) has run",
        },
        "planners/planner.identity": {"maturity": "unverified"},
        "envs/world_model": {
            "maturity": "unverified",
            "notes": "world-model pre-screening not yet used in a campaign",
        },
    },
}
