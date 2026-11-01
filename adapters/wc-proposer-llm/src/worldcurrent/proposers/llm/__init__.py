# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"generators": "wc.generator/v0.1", "reflectors": "wc.reflector/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "generators/llm.steering": {
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
        "reflectors/llm.reflector": {
            "maturity": "unverified",
            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)",
        },
    },
}
