"""Hylo memory backends."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"memory": "hylo.memory/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "memory/lesson_book": {"maturity": "unverified", "notes": "experience campaigns not yet run"},
        "memory/skill_library": {"maturity": "unverified", "notes": "experience campaigns not yet run"},
        "memory/sqlite": {"maturity": "unverified"},
    },
}
