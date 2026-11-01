# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"generators": "hylo.generator/v0.1", "reflectors": "hylo.reflector/v0.1"},
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
