"""Hylo decision-making: agents (composition, steering, planning, skills, baselines).

Instruments (success judges, value models, world models) are in hylo-instruments; rule-based reasoners in
hylo-reasoners. Both are optional extras of this distribution (`critic`, `rules`).
"""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "agents": "hylo.agent/v0.1",
        "planners": "hylo.planner/v0.1",
    },
    "maturity": "implemented",
    "plugins": {
        "agents/planner.subgoal": {
            "notes": "LIBERO-10 with planner.llm (maturity-20261101; 10 episodes) and MuJoCo with planner.identity",
        },
        "agents/shadow": {"notes": "hylo evolve shadow on MuJoCo planar reach (maturity-20261101; 2858 decisions)"},
        "agents/skills": {
            "notes": "applied a campaign-verified steering skill on MuJoCo planar reach (maturity-20261101); "
                     "with a remote SteeredAgent the skill agent must run in the same process (inside the remote)",
        },
        "agents/memory.steered": {
            "maturity": "unverified",
            "notes": "the class runs inside lerobot.memory_steered (memory-ab runs); by name it needs a sampler with "
                     "the prior protocol, and only PriorLeRobotSampler implements one",
        },
        "agents/noise.steered": {
            "maturity": "unverified",
            "notes": "the class runs inside lerobot.noise_steered (noise-ab runs); by name it needs a sampler with "
                     "the prior protocol, and only PriorLeRobotSampler implements one",
        },
    },
}
