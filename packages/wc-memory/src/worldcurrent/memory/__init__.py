"""worldcurrent memory backends."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"memory": "wc.memory/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "memory/lesson_book": {"maturity": "unverified", "notes": "experience campaigns not yet run"},
        "memory/skill_library": {"maturity": "unverified", "notes": "experience campaigns not yet run"},
        "memory/sqlite": {"maturity": "unverified"},
    },
}
