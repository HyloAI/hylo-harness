# Stack campaign with a program written by a local model

One complete improvement campaign on the `stack` task of `mujoco/manip-v0` (Franka Panda, MuJoCo, CPU only). The
parent is a deliberately weak, hand-written program. The generator `research.llm` runs a research session in which a
local language model (gpt-oss:20b through Ollama) writes a better program. The campaign's gate decides whether that
program replaces the parent, held-out episodes confirm the decision, and the program is promoted.

| | |
|---|---|
| Task | `stack`: two or three blocks, stacked in the order the instruction names, bottom block on the table |
| Parent | [`parent_solve.py`](parent_solve.py), hand-written and **deliberately weak** (see below) |
| Generator | `research.llm`, model `ollama:gpt-oss:20b` (text only, reasoning effort low), 3 development cases |
| Result | promoted. Gate 0.95 vs 0.50 (9 wins, 0 losses, p = 0.002); held-out 1.00 vs 0.60 (8 wins, 0 losses, p = 0.0039) |
| Promoted program | [`promoted_solve.py`](promoted_solve.py), sealed artifact digest `0a1ee96b2f810ab6` |
| Cost | $0 (local model), 30 model calls, 406,698 input and 7,082 output tokens, about 17 min of model latency |

## The parent is weak on purpose

The parent is the reference `stack` program (`examples/programs/stack.py`) with three things removed: it places only
the first block of the named order (so it can finish two-block stacks but never three-block ones), it does not retry a
missed grasp, and it does not retry a level. It is a real program that works on part of the task family: on the
splits below it succeeds on 30 to 60 percent of the episodes, which are the two-block ones. It is not a result to beat;
it gives the gate a real program to compare against.

## What happens

1. **Parent runs.** The campaign runs the parent (`agent.type: code.program`, no model) on the `search` split (40
   episodes, success 0.55) and on `search_val` (20 episodes, success 0.30).
2. **Research session** (`research.llm`, the only place a model is used). The model works on 3 development cases from
   the `search` split (seeds 30000, 30001 and 30002, all three of them three-block stacks). It sees the robot description,
   the toolkit's API, the parent's source, and the instructions of its development cases. The parent's functions are
   already defined in its namespace. It can run code on a live episode, `reset(case)` to another case,
   and `submit()` a program, which runs without the model on a fresh episode of every development case. Budget: 30
   turns, 6 resets, 3 submissions.

   In the first turn the model rewrote `solve` so it goes through every level of the order, and restored grasp retries.
   It ran `solve(robot)` once on the live case, then submitted: 3 of 3 development cases succeeded. Its two later
   submissions also scored 3 of 3, so the first one was frozen into a sealed program artifact (`kind: program`,
   provenance: the `search` split only) and proposed as a code candidate. The candidate is a patch that sets
   `agent.type: code.program` and the artifact.
3. **Campaign cascade.** The candidate goes through the same steps as any other candidate (RFC-0002):
   - stage 1, early rejection on the first 10 pairs of `search_val`;
   - stage 2, all of `search_val`;
   - stage 3, the gate on unseen initial states, a paired McNemar test at alpha 0.05;
   - the held-out report, which is never used for selection.

   The model never sees any of these outcomes.
4. **Promotion.** The gate passed, held-out confirmed it, and approval is `auto`, so the candidate became the active
   version (`active.json`). It can be stored as a program skill (last command below).

## Numbers

From `runs/campaign/campaign/cycle-01/cycle_report.json` and `hylo evolve status runs/campaign/campaign`:

| Step | Split (episodes) | Candidate | Parent | Wins / losses (pairs) | p |
|---|---|---|---|---|---|
| Research session | search, 3 dev cases | 3/3 (each of 3 submissions) | | | |
| Stage 1: early rejection | first 10 of search_val | | | 6 / 0 (10) | |
| Stage 2 | search_val (20) | 1.00 | 0.30 | 14 / 0 (20) | 6.1e-5 |
| Stage 3: gate | gate (20) | 0.95 | 0.50 | 9 / 0 (20) | 0.002 |
| Held-out report | heldout (20) | 1.00 | 0.60 | 8 / 0 (20) | 0.0039 |

Session: 30 turns (the model did not call `finish()`, so the turn budget ended it), 0 resets, 3 submissions,
406,698 input tokens and 7,082 output tokens. Model latency was 1,031 s, about 17 min, and the session took 22 min of
wall time. The whole campaign took 44 min, including about 22 min of parent and candidate rollouts on the CPU with 10
episode workers. The cost is $0 because the model runs locally. The one gate failure out of 20 was the candidate's
only loss in any split, and it was not a paired loss: the parent failed that episode too.

## The promoted program

[`promoted_solve.py`](promoted_solve.py) is the frozen `solve.py` exactly as sealed. Compared with the parent, it:

- goes through every block of the order instead of only the first pair;
- uses three grasp attempts, and tries again when no grasp plan is found;
- after each placement, looks again and checks with `stacked()` that the block is on the one below, and stops if not.

One quirk, visible in the videos: its loop also picks up the bottom block ("first block: just lift and keep"). That
block drops back near its old place when the hand opens for the next grasp. The program then stacks on where that
block was measured before it was picked up. It is a wasted motion that the gate did not penalise (19 of 20 gate
episodes succeeded). It is left as the model wrote it.

Sealed artifact: `runs/campaign/research/base-r0/frozen/campaign-stack-base-r0-sub01`, digest
**`0a1ee96b2f810ab6`**, builder `hylo.codeagents.research@0.1`, splits `[search]`. `hylo artifact` verifies it: it
recomputes the content digest and compares it with the manifest. A copy with one comment line added is reported as
`INVALID: content digest a45ffaaaf7c8fbc4 != manifest 0a1ee96b2f810ab6 (modified after sealing)`.

## Videos

All three are re-simulated bit for bit from the recorded actions in the traces, with captions taken from the traces.

| Video | What it shows | Renderer |
|---|---|---|
| [`stack_campaign_before_after.mp4`](../videos/stack_campaign_before_after.mp4) | Held-out seed 30080 ("build a tower: the blue block at the bottom, then the green block, then the orange block on top"), side by side. Left, "before: hand-written parent": stops after one level (failure, 177 steps). Right, "after: program written by gpt-oss:20b, promoted by the gate": builds the three-block tower (success, 424 steps). The shorter clip holds its last frame. | Blender Cycles (CPU) |
| [`stack_campaign_promoted_program_blender.mp4`](../videos/stack_campaign_promoted_program_blender.mp4) | The promoted program alone on the same seed, with its printed output | Blender Cycles (CPU) |
| [`stack_campaign_parent_weak.mp4`](../videos/stack_campaign_parent_weak.mp4) | The parent alone on the same seed | MuJoCo |

## Reproduce

Requirements: the hyloai workspace (one `uv` environment), an Ollama server at `http://localhost:11434` with
`gpt-oss:20b`, and OSMesa for the cameras. The Blender videos also need Blender (`$HYLO_BLENDER`, by default
`~/.local/opt/blender/blender`). Run from the workspace root:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu

# the campaign: parent rollouts, research session, cascade, gate, held-out report, promotion
uv run hylo evolve run harness/adapters/hylo-code-agents/examples/campaign/campaign.yaml
uv run hylo evolve status runs/campaign/campaign

# check the sealed program, then store the promoted program as a program skill
uv run hylo artifact runs/campaign/research/base-r0/frozen/campaign-stack-base-r0-sub01
uv run hylo research skills runs/campaign/campaign --library runs/skills/manip --family stack

# videos (RUNS=runs/campaign/campaign/cycle-01/runs; episode ids from the evallog.json of each run)
RUNS=runs/campaign/campaign/cycle-01/runs
uv run python harness/adapters/hylo-code-agents/examples/llm/compare_videos.py \
    $RUNS/campaign-stack-parent-heldout-* mujoco.manip_stack-s30080-27feaa05 "before: hand-written parent" \
    $RUNS/campaign-stack-cand-*-heldout-* mujoco.manip_stack-s30080-0d64c3e1 \
    "after: program written by gpt-oss:20b, promoted by the gate" \
    harness/adapters/hylo-code-agents/examples/videos/stack_campaign_before_after.mp4
uv run python harness/adapters/hylo-code-agents/examples/llm/make_videos.py \
    $RUNS/campaign-stack-cand-*-heldout-* mujoco.manip_stack-s30080-0d64c3e1 \
    harness/adapters/hylo-code-agents/examples/videos/stack_campaign_promoted_program_blender.mp4 \
    --workflow "research.llm campaign: promoted program (held-out)" --model "written by gpt-oss:20b" --blender
```

The configs are [`campaign.yaml`](campaign.yaml) and its experiment [`experiment.yaml`](experiment.yaml). The
experiment holds the parent's source inline, and its config digest is `24318af74137ae70`. The recorded run used
`runs/campaign/campaign.yaml`, which has the same settings and absolute output paths. Settings in brief:

- benchmark `mujoco/manip-v0`, task `stack`, `rsi_states: 100` (search 40, search_val 20, gate 20, heldout 20 seeds);
- cameras front, side and wrist with the environment's segmentation, and no privileged state;
- protocol: alpha 0.05, McNemar, 10 stage-1 pairs, one gate look, at most one candidate, `auto` approval.

The program runs (parent and candidate) are deterministic. The model's replies are not guaranteed to repeat, even
at temperature 0, so another session can write a different program.

## How this run came about

This is the fourth launch of this campaign, and the only one whose numbers are reported here. The first three were
stopped while fixing general problems that they exposed; none of the fixes is specific to this task:

- gpt-oss sometimes writes its code on Ollama's tool channel, which made the HTTP call fail. The code is now
  recovered as the reply (`hylo-reasoner-llm`).
- The development cases were listed with the task template ("stack the <blocks> in the named order") instead of
  each episode's own instruction.
- A session that improves a parent now starts with the parent's program defined in its namespace. Before, the model
  rewrote the program from scratch and lost its working parts.
- `submit()` without an argument now submits the functions the blocks defined.
