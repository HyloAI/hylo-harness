"""LLM/VLM-backed reasoner components."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"choosers": "hylo.chooser/v0.1", "planners": "hylo.planner/v0.1", "instruments": "hylo.instrument/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "choosers/chooser.llm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "choosers/chooser.vlm": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "planners/planner.llm": {
            "notes": "LIBERO-10 in front of SmolVLA, 10 episodes (maturity-20261101): it ran, 1 of 10 succeeded; no "
                     "same-state comparison without the planner yet",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "instruments/judge.vlm": {
            "notes": "calibrated on 300 LIBERO-10 episodes (judges/calib-vlm-300): AUC 0.61, recall 0.13 at precision "
                     "0.95, not ready to label",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
    },
}
