"""LLM/VLM-backed reasoner components."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"choosers": "wc.chooser/v0.1", "planners": "wc.planner/v0.1", "instruments": "wc.instrument/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "choosers/chooser.llm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "choosers/chooser.vlm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "planners/planner.llm": {
            "maturity": "unverified",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "instruments/judge.vlm": {
            "maturity": "unverified",
            "notes": "not yet calibrated against ground truth",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
    },
}
