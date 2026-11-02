---
name: hylo-run
description: Run a Hylo experiment config (benchmark + env + agent) with `hylo run`, then read the episodes with `hylo trace`. Use when asked to evaluate an agent or policy, reproduce a number, or debug why episodes fail.
---

# Run an experiment and read it

An experiment config (`schema_version: hylo.experiment/v0.1`) names a benchmark and split, an environment, an
agent and an output directory. Examples live in `bench/examples/configs/`.

## Steps

1. Check that the config's environment and agent meet the contracts:
   `uv run hylo conformance CONFIG`
2. Run it, overriding fields instead of editing the file:
   ```bash
   uv run hylo run bench/examples/configs/mujoco_panda_stack.yaml --split validation --set workers=8
   ```
   `--set` takes dotted paths (`--set benchmark.params.tasks=[stack]`, `--set agent.params.model.model=gpt-oss:20b`).
3. The last lines print the run directory and the success rate. Summarise the run, then one episode:
   ```bash
   uv run hylo trace RUN_DIR
   uv run hylo trace RUN_DIR EPISODE_ID
   ```
   An episode summary lists its chunks, safety interventions, the end reason (`success`, `timeout`,
   `agent_done: ...`, `error: ...`) and the timings.

## Reading failures

- `agent_done` without success means the agent declared itself finished: look at its last chunk's provenance.
- Many safety interventions mean actions hit the joint or velocity limits; the shield clipped them.
- For code agents (`code.llm`, `code.program`) each episode keeps the program's stdout and any traceback in its
  trace events. Read them before changing anything.

## Rules

- Report the split and the number of episodes with every success rate; a rate on 10 episodes is a hint, not a result.
- Do not tune on `gate` or `heldout`. Use `search` and `search_val` while developing.
- Camera-based configs need `MUJOCO_GL=egl` or `osmesa`.
