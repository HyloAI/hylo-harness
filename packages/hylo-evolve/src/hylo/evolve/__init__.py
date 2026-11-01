"""Hylo evolution layer."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"generators": "hylo.generator/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "generators/critic.chunk_value": {
            "maturity": "unverified",
            "notes": "not yet run in a campaign; it steers a LeRobot policy (lerobot.critic_steered), so its target is "
                     "LIBERO, where the offline pair accuracy (0.54) was below its proposal floor (0.55)",
        },
        "generators/experience": {"notes": "two campaigns on MuJoCo planar reach with lessons, skills and llm.reflector "
                                           "(maturity-20261101)"},
        "generators/learn": {"notes": "retrained a flow policy inside a MuJoCo planar reach campaign and gated it "
                                      "(maturity-20261101)"},
        "generators/selftrain.expert_iteration": {"maturity": "unverified", "notes": "smoke run only"},
    },
}
