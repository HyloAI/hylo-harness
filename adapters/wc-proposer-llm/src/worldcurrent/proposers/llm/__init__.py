# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"generators": "wc.generator/v0.1", "reasoners": "wc.reasoner/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "generators/llm.steering": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "reasoners/llm.reflector": {
            "maturity": "unverified",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
    },
}
