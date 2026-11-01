"""LeRobot policies as worldcurrent Agents."""

WEIGHTS = "per checkpoint (its model card); PaliGemma-based pi0 / pi0.5: Gemma Terms of Use"

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {
        "agents": "wc.agent/v0.1",
        "policies": "wc.policy/v0.1",
        "datasources": "wc.datasource/v0.1",
        "learners": "wc.learner/v0.1",
    },
    "maturity": "implemented",
    "plugins": {
        "agents/lerobot": {"weights": WEIGHTS},
        "policies/lerobot": {"weights": WEIGHTS},
        "agents/lerobot.steered": {"weights": WEIGHTS},
        "agents/lerobot.critic_steered": {"weights": WEIGHTS},
        "agents/lerobot.memory_steered": {"weights": WEIGHTS},
        "agents/lerobot.noise_steered": {"weights": WEIGHTS},
        "datasources/lerobot.dataset": {
            "maturity": "unverified",
            "notes": "dataset downloads not yet approved here; each dataset has its own licence",
        },
        "learners/lerobot.expert_iteration": {
            "maturity": "unverified",
            "notes": "smoke run only",
            "weights": WEIGHTS,
        },
    },
}
