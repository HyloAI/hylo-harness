"""LeRobot policies as Hylo Agents."""

WEIGHTS = "per checkpoint (its model card); PaliGemma-based pi0 / pi0.5: Gemma Terms of Use"

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "agents": "hylo.agent/v0.1",
        "policies": "hylo.policy/v0.1",
        "datasources": "hylo.datasource/v0.1",
        "learners": "hylo.learner/v0.1",
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
            "notes": "ingested a local LeRobotDataset v3 (external-selftrain-smoke-demos); hub downloads not yet approved "
                     "here; each dataset has its own licence",
        },
        "learners/lerobot.expert_iteration": {
            "maturity": "unverified",
            "notes": "smoke run only",
            "weights": WEIGHTS,
        },
    },
}
