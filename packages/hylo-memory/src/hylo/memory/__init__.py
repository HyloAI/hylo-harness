"""Hylo memory backends."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"memory": "hylo.memory/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "memory/lesson_book": {"notes": "written by llm.reflector and read and credited by two experience campaigns "
                                        "on MuJoCo planar reach (maturity-20261101)"},
        "memory/skill_library": {"notes": "skills derived from a campaign, one verified by its gate and then applied by "
                                          "the skills agent on MuJoCo planar reach (maturity-20261101)"},
        "memory/sqlite": {"notes": "spatial memory of composable.object_reach on the MuJoCo table (maturity-20261101)"},
    },
}
