---
name: hylo-research
description: Run a Hylo research session, in which a chat model writes and refines a `solve(robot)` manipulation program on development cases and freezes the best one as a code candidate. Use when asked to have an LLM solve a manipulation task, write a policy as code, or produce a program skill.
---

# Research sessions

In a research session a chat model works on a few development cases from the `search` split. It can reset a
case, run code against the robot toolkit (`hylo.manip`: look, find, grasps, plan, run, why, ...), see the images
and printed output, submit a program for scoring, and finish. The best submission is frozen as a candidate of
kind `code`.

## Steps

1. Start from an experiment whose agent is `code.llm` (it supplies the model), or pass `--model`:
   ```bash
   MUJOCO_GL=osmesa uv run hylo research runs/research/configs/stack.yaml \
       --dev 3 --budget turns=30,resets=6,submits=3 --model ollama:qwen3.5:35b-a3b --name stack-qwen
   ```
   Budget keys: `turns`, `tokens`, `cost` (USD), `resets`, `submits`. The session stops at the first one used up.
2. The session directory (`<output_dir>/research/<name>/`) holds:
   - `log.jsonl`: every turn (the model's text, the code it ran, the output, the images it was shown).
   - `submissions/`: each submitted program with its development score.
   - `summary.json`: the frozen program, its score, tokens and cost.
3. To measure the frozen program on splits the model never saw, add `--evaluate` to the session command:
   ```bash
   ... --evaluate search_val gate heldout --with-parent
   ```
   The model never sees these results. `--with-parent` also runs the experiment's own agent on the same cases, so the two can be compared pairwise.
4. Put the program through the gate with a campaign (see the `hylo-evolve` skill, proposer `research.llm`).
   After promotion, turn promoted programs into reusable program skills:
   ```bash
   uv run hylo research skills CAMPAIGN_DIR --library runs/skills --family stack
   ```

## Writing programs by hand

A program is a Python file that defines `solve(robot)`. `robot` is the toolkit facade; `print(robot.api_reference())`
lists every verb. See `harness/adapters/hylo-code-agents/examples/programs/` for six worked programs that use the
cameras only, never privileged state. Test one with the `code.program` agent, which takes the source inline
(`agent: {type: code.program, params: {source: ...}}`) or a sealed program (`params: {artifact: artifact:NAME@DIGEST}`).
`run_programs.py` in that folder runs all six.

## Rules

- Development scores are optimistic because the model saw those cases. Quote `search_val`, `gate` or `heldout`.
- Report tokens and cost from `summary.json` next to the score.
- If the model keeps failing the same way, read `log.jsonl` around the failure before raising the budget.
