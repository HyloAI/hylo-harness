"""Hylo instruments: learned models that measure or predict but never act.

Success judges (judges), chunk value models (critic) and world models (world_model). They are trained by
hylo-learn, used by the improvement loop (labels, pre-screening) and by agents that consult them
(hylo.agents.critic.CriticSteeredAgent).
"""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {
        "envs": "hylo.env/v0.1",
        "instruments": "hylo.instrument/v0.1",
    },
    "maturity": "implemented",
    "plugins": {
        "envs/world_model": {
            "notes": "pre-screened candidates in a MuJoCo planar reach campaign (maturity-20261101); state models only",
        },
    },
}
