"""Optional SimpleMemVLA integration. Model dependencies are imported only at runtime."""

META = {
    "contracts": {"agents": "hylo.agent/v0.1"},
    "maturity": "unverified",
    "plugins": {
        "agents/simplememvla": {
            "notes": "Independent video-memory option; memory bridge GPU-verified, native agent lifecycle CPU-tested; requires pinned upstream SimpleMemVLA and a compatible frozen checkpoint",
            "weights": "Released yinchenghust/simplememvla_robomme checkpoint: MIT",
        },
    },
}
