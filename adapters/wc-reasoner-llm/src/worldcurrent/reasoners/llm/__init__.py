"""LLM/VLM-backed reasoner components."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"reasoners": "wc.reasoner/v0.1", "instruments": "wc.instrument/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "reasoners/chooser.llm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "reasoners/chooser.vlm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "reasoners/planner.llm": {
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
