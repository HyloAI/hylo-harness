---
name: hylo-agent-runs
description: Run the tool-calling robot agent (`tools.llm`) on a benchmark, check a model and a config before a run with `hylo preflight`, watch or steer a running agent with `hylo watch`, and seal a finished run with `hylo seal`. Use when asked to evaluate a chat model on robot tasks (MuJoCo manipulation, LIBERO, perturbed LIBERO), to follow what an agent is doing, or to make a run's evidence verifiable.
---

# Tool-calling agent runs

`tools.llm` gives a chat model the robot as typed tools (locate, grasp options, grasp, place, move_tcp, ...). It
reads and measures freely; every action it takes is a proposal that runs through the kernel and the safety shield
and comes back as a measured receipt with fresh images. The task's own check decides success.

## Commands

```bash
uv run hylo preflight runs/configs/tools-llm.yaml --out runs/tools-llm-run     # config, storage, live model probe
uv run hylo preflight --model ollama:qwen3.5:35b-a3b --vision                   # just the model

MUJOCO_GL=osmesa uv run hylo run runs/configs/tools-llm.yaml
uv run hylo watch runs/<run-dir>                    # follow the newest episode; --replay --speed 4 for a finished one
uv run hylo watch --say "the target is the small cup" --inbox runs/steer/inbox.jsonl
uv run hylo seal runs/<run-dir>                     # verify hash chains, write seal.json, make files read-only
uv run hylo seal --verify runs/<run-dir>

# all six mujoco/manip-v0 tasks, or LIBERO with perturbations
MUJOCO_GL=osmesa uv run python harness/adapters/hylo-code-agents/examples/tools_llm/run_tools_llm.py run \
    --out runs/tools-llm --episodes 3
uv run python .../run_tools_llm.py run --suite libero --tasks libero_goal:8 --perturb position language camera \
    --episodes 2 --out runs/tools-llm-libero
```

Config: `agent: {type: tools.llm, params: {model: {backend: ollama, model: qwen3.5:35b-a3b, vision: true,
reasoning_effort: none}, max_decisions: 20, max_calls: 60, inbox: runs/steer/inbox.jsonl}}` with an environment
that publishes a RobotSpec (`mujoco.panda_table`, `mujoco.aloha_table`, `mujoco.g1_table`, `libero` with
`control: joint_position, tool_obs: true, segmentation: true`).

LIBERO perturbations (`benchmark params: perturb`): position, task (a rewritten goal), language (a paraphrase),
object (recoloured), camera, noise, robot_init; combine with a list of lists.

## Reading the result

- `agent.tool.summary` per episode: calls, decisions (refusals that moved nothing are not counted), tokens,
  outcome the agent stated, `human_assisted` when anyone sent a message (do not count such runs as autonomous).
- An agent that says `finish(success)` while the task check failed is common: the verdict is the episode's status.
- `results.md` from the runner: success per task, decisions, model calls, tokens, wall time.
