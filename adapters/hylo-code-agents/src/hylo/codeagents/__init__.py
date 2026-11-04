"""Tool-using, code-writing agents (RFC-0006 parts 4-6).

- sandbox: AST allow-list validation and a restricted, persistent namespace for model-written Python
- bridge: the toolkit's Executor on top of the kernel (robot.run -> one ActionChunk through the SafetyShield)
- agents: `code.llm` (a chat model writes one block per turn) and `code.program` (a frozen solve(robot), no model)
- research: research sessions on development cases, frozen into program candidates (`hylo research`)
- generator: `research.llm`, a research session as an improvement-campaign generator
- skills: promoted programs as skills of the skill library
- scene_check, scenes: the scene builder (`hylo scene`): a model writes a hylo.scene/v0.1 file, the checker
  validates it, code.llm solves it with privileged state
"""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"agents": "hylo.agent/v0.1", "generators": "hylo.generator/v0.1"},
    "maturity": "unverified",
    "plugins": {
        "generators/harness.program": {"notes": "Independent HarnessPAI-inspired phase planning, program repair and evidence-linked repair recall; benchmark effectiveness unverified; no upstream code included",
                                       "weights": "the configured reasoning model's own licence"},
        "agents/code.llm": {"notes": "run on mujoco/manip-v0 (examples/RESULTS.md): gpt-oss:20b zero-shot 14/30 "
                                     "(pick_place, put_in_bowl, stack solved; push, press, sort not), qwen3.5:35b-a3b "
                                     "2/6, Claude Sonnet 5.5 via the CLI 5/6",
                            "weights": "the served model's own licence (e.g. a Qwen or GLM model run by Ollama)"},
        "agents/code.program": {"notes": "tested with the real toolkit on hylo-manip's kinematic test kit and on a toy double"},
        "generators/research.llm": {"notes": "a gpt-oss:20b session improved a weak stack program, gate 0.95 vs 0.50 "
                                             "(p=0.002), held-out 1.00 vs 0.60, promoted (examples/campaign)",
                                    "weights": "the served model's own licence"},
    },
}
