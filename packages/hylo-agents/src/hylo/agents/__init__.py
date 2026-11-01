"""Hylo intelligence layer: agents, reasoners, policies."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "envs": "hylo.env/v0.1",
        "agents": "hylo.agent/v0.1",
        "reasoners": "hylo.reasoner/v0.1",
        "choosers": "hylo.chooser/v0.1",
        "planners": "hylo.planner/v0.1",
        "instruments": "hylo.instrument/v0.1",
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
