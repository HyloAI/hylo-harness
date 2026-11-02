# Tool-using agents with real language models on mujoco/manip-v0

Recorded runs of `code.llm`, research sessions (`hylo research`), a `research.llm` improvement campaign and
program-skill reuse, on the six Panda tasks of `mujoco/manip-v0`.

Common setup for every run:

- environment: Franka Panda in MuJoCo, cameras front, side and wrist with the environment's segmentation, no
  privileged state;
- hardware: one shared 128-thread CPU, no GPU;
- models: local models through Ollama, `qwen3.5:35b-a3b` (vision, thinking off) and `gpt-oss:20b` (text, reasoning
  effort low), plus one small comparison with Claude Sonnet 5.5 through the Claude Code CLI (text only);
- the task's score is never shown to a model.

Drivers are in `examples/llm/` (`run_code_llm.py`, `make_videos.py`, `compare_videos.py`). Videos are in
[`videos/`](videos/README.md), and the campaign has its own page, [`campaign/README.md`](campaign/README.md).

**Shared model server.** Ollama ran with one request slot per model and served four agents at once. The CPU load
average was 130 to 200 for most of the day. In those hours a qwen3.5 turn took 20 to 30 min, almost all of it in the
queue: a one-token request waited 19 min. Late at night the same turn took 30 to 60 s. Latencies below therefore
include queueing. The qwen3.5 rows have fewer episodes than planned, and gpt-oss:20b, whose runner was nearly idle,
carried the 5-episode rows. From 00:25 to about 01:10 Ollama was stopped. Episodes cut by that stop are not counted;
they are listed at the end and were rerun.

## 1. Code policy (`code.llm`), validation split

Each episode: one model conversation, at most 12 turns, a 60 s compute limit per block, and the same validation seeds
(10000 to 10004) for every row. One-shot: the prompt holds the reference program of a **different** task family.
The pairs were pick_place ← put_in_bowl, put_in_bowl ← pick_place, stack ← pick_place, sort_by_colour ← pick_place,
push_to_zone ← press_by_number and press_by_number ← push_to_zone. Columns: successes over episodes; mean turns; mean
model latency per turn (queueing included); mean input/output tokens per episode; mean wall time per episode.

### gpt-oss:20b (5 episodes per task)

| Task | Zero-shot | Turns | s/turn | Tokens in / out | Wall min | One-shot | Turns | s/turn | Tokens in / out | Wall min |
|---|---|---|---|---|---|---|---|---|---|---|
| pick_place | **5/5** | 1.2 | 73 | 5,296 / 652 | 1.6 | **5/5** | 1.6 | 192 | 10,184 / 1,169 | 4.2 |
| stack | **4/5** | 3.2 | 43 | 17,197 / 930 | 2.4 | 3/5 | 3.0 | 36 | 19,968 / 1,640 | 1.7 |
| put_in_bowl | **5/5** | 1.2 | 74 | 5,251 / 714 | 1.8 | 3/5 | 2.8 | 19 | 17,795 / 931 | 1.1 |
| push_to_zone | 0/5 | 10.2 | 46 | 64,922 / 3,037 | 7.6 | 0/5 | 9.0 | 22 | 70,455 / 3,416 | 3.6 |
| press_by_number | 0/5 | 11.0 | 65 | 69,523 / 4,279 | 12.2 | 0/5 | 3.6 | 15 | 22,394 / 1,791 | 1.1 |
| sort_by_colour | 0/5 | 4.0 | 162 | 19,026 / 1,032 | 10.5 | **1/5** | 2.8 | 11 | 19,866 / 1,612 | 1.2 |
| **all** | **14/30** | | | | | **12/30** | | | | |

Totals for these 60 episodes: 268 model calls, 1.71 M input and 0.11 M output tokens, 4.1 h of wall time, $0.

An earlier run of the zero-shot row (`runs/code-llm-gptoss-v1`, before most of the fixes in section 6) gave the same
picture: pick_place 5/5, put_in_bowl 5/5, stack 4/5, push 0/5, press 0/5, sort 0/5. The only change was that its
failures came sooner.

### qwen3.5:35b-a3b (vision), 1 episode per task (the shared queue allowed no more)

| Task | Zero-shot | Turns | s/turn | Tokens in / out | Wall min |
|---|---|---|---|---|---|
| pick_place | **1/1** | 2 | 1,460 | 10,340 / 1,034 | 48.8 |
| stack | **1/1** | 1 | 1,732 | 4,540 / 776 | 29.0 |
| put_in_bowl | 0/1 | 7 | 746 | 56,912 / 3,397 | 87.5 |
| push_to_zone | 0/1 | 12 | 39 | 150,937 / 11,488 | 7.9 |
| press_by_number | 0/1 | 11 | 58 | 148,704 / 16,719 | 11.1 |
| sort_by_colour | 0/1 | 12 | 36 | 134,974 / 11,069 | 7.3 |

The first three ran under heavy contention (12 to 29 min per turn) and the last three in the quiet hours (36 to
58 s per turn). qwen3.5 writes longer replies than gpt-oss: about 1,000 output tokens per turn against about 400.
One-shot runs with qwen3.5 were not done; at the daytime latency they would have taken about 15 h.

### Comparison: Claude Sonnet 5.5 through the Claude Code CLI (6 episodes, text only)

| Task | Zero-shot | Turns | s/turn | Cost |
|---|---|---|---|---|
| pick_place | **1/1** | 2 | 5 | $0.08 |
| stack | **1/1** | 1 | 6 | $0.04 |
| put_in_bowl | **1/1** | 2 | 7 | $0.08 |
| push_to_zone | **1/1** | 4 | 10 | $0.19 |
| press_by_number | **1/1** | 4 | 11 | $0.19 |
| sort_by_colour | 0/1 | 7 | 12 | $0.40 |

5 of 6 episodes succeeded, in 9 min of wall time for all six, for a total of $0.98. Each turn is one stateless
`claude -p` call with the conversation flattened onto stdin, and with `--tools ""` so that the CLI's own tools are
off. The sort_by_colour failure ended with 2 of 3 blocks in their trays when it ran out of control steps.

### Typical failure modes (from the `agent.code.turn` events)

- **Straight-line motions (push_to_zone, press_by_number): 0 of 30 gpt-oss episodes.** The models build `push` and
  `press` goals from box poses whose TCP +x axis points sideways, not down. They then retry variations of
  "pre-pose in collision (table)", "approach line ... unreachable", and "ik residual 0.17 m" until the 12 turns
  run out. Planning failures appear in 13 of the 20 zero-shot and one-shot push/press failures. The demonstration
  from the other straight-line task did not help (0/5 both ways). Claude solved both tasks in 4 turns.
- **Ending too early (sort_by_colour).** In all 5 zero-shot gpt-oss failures the model placed every block, printed
  small `tcp_error`s and called `finish()` without looking again. A replay with the task's own check shows 0, 2, 2,
  2 and 2 of 3 or 4 blocks in the right tray. Likely causes, not checked per episode: a block resting on a tray wall,
  or the last block still touching the open fingers when `finish()` came.
- **Ambiguous perception results.** `robot.find(view, ["blue block", "red block"])` also returns other colours by
  word overlap (score 0.5); models that do not filter by score act on the wrong object.
- **Python errors** (wrong signatures such as `robot.grasps(points, ...)` without the arm, keys that do not exist,
  `import time`, `locals()`) appear in about half of the failures. They cost turns but are rarely the final cause.
- **qwen3.5** fails the same way as gpt-oss on push and press, more slowly and at greater length (150 k input
  tokens in 12 turns). On put_in_bowl it spent 7 turns of 12 min each and ran out of control steps.

## 2. Research sessions (`hylo research`)

Each session: development cases from the `search` split, a budget of 30 turns, 6 resets and 3 submissions, and the
cameras with segmentation. The frozen program would then be evaluated on search_val, gate and heldout with
`--evaluate` (10 episodes each). The model never sees those results.

| Session | Model | Dev cases | Submissions (dev successes) | Frozen | Turns, resets | Tokens in / out | Model latency |
|---|---|---|---|---|---|---|---|
| stack, from scratch | gpt-oss:20b | 2 | 0/2, 0/2, 0/2 | none | 30, 0 | 358,544 / 10,872 | 4.7 min |
| press_by_number, from scratch | gpt-oss:20b | 2 | none | none | 30, 1 | 304,949 / 4,444 | 2.2 min |
| sort_by_colour, from scratch | gpt-oss:20b | 2 | none | none | 30, 0 | 261,601 / 1,529 | 0.9 min |
| stack, improving a weak parent (campaign) | gpt-oss:20b | 3 | **3/3**, 3/3, 3/3 | sub 1 | 30, 0 | 406,698 / 7,082 | 17 min |
| press_by_number, from scratch | qwen3.5 | 2 | none | none | 30, 0 | 589,370 / 40,498 | 61 min |
| stack, from scratch | qwen3.5 | 2 | 0/2 | none | 30, 6 | 462,277 / 30,722 | 32 min |
| sort_by_colour, from scratch | qwen3.5 | 2 | none | none | 30, 0 | 532,714 / 47,923 | 44 min |

Earlier sessions, recorded but superseded:

- gpt-oss with earlier code: stack 0/2 with 1 submission; sort and press with no submission.
- qwen3.5 stack and press: 15 model calls each over 3.7 h of queueing, each with one submission at 0/2. Both were
  cut by the Ollama stop.

All of these are under `runs/research/aborted-*` and `runs/research/interrupted/`.

No from-scratch session froze a program, so there was nothing to evaluate on search_val, gate or heldout for these
six sessions. The only program that went through those splits is the campaign's (section 3).

**What works and what does not.** With a working parent defined in its namespace, gpt-oss turned a two-block-only
program into a program that does every level, in one turn, and then submitted it. From scratch, neither local model
produced a program that solved a development case:

- gpt-oss writes one long program early (hard-coded labels, placing at `(0, 0)`), then shrinks to 20 to 50 tokens
  per turn. It redefines `solve` repeatedly or prints "task done", and twice it resubmitted an identical program.
- In 30 turns it never ran its program on a live case to look at the result.
- qwen3.5 experiments for longer, at 1,000 to 3,000 tokens per reply, but loops. In press_by_number it sent the
  same 1,334-token reply six times in a row after "approach line from the pre-pose unreachable"; at temperature 0
  the same feedback produces the same reply. That led to the repeated-block notice in section 6, which only the
  sort session had. Its one stack submission hard-coded the colours of case 0 and scored 0/2.

## 3. Campaign: `research.llm` through the gate

The campaign ran on stack, with a deliberately weak hand-written parent that does two-block stacks only, without
retries. All details are on [`campaign/README.md`](campaign/README.md).

| Step | Candidate | Parent | Wins / losses (pairs) | p |
|---|---|---|---|---|
| Research session (3 dev cases) | 3/3 | | | |
| Stage 1 (first 10 pairs) | | | 6 / 0 (10) | |
| search_val (20) | 1.00 | 0.30 | 14 / 0 (20) | 6.1e-5 |
| **Gate** (20, McNemar, alpha 0.05) | **0.95** | 0.50 | 9 / 0 (20) | **0.002**, passed |
| **Held-out** (20, report only) | **1.00** | 0.60 | 8 / 0 (20) | 0.0039 |

The candidate was promoted (approval `auto`), sealed artifact `0a1ee96b2f810ab6` (verified by `hylo artifact`). The
campaign took 44 min: a 22 min research session and 22 min of rollouts. It cost $0.

## 4. Program skill reuse

`hylo research skills runs/campaign/campaign --library runs/skills/manip --family stack` turned the promoted program
into the verified program skill `stack@1`, with its gate evidence and digest. `code.llm` then ran with that library
(at most 8 turns) on validation seeds 10000 to 10009.

| Run | Skill retrieved | Episodes in which the model called the skill | Success | Turns | Output tokens per episode |
|---|---|---|---|---|---|
| gpt-oss, before the retrieval fix | no (similarity 0.14 < 0.2) | 0 of 10 | 9/10 | 1.8 | 724 |
| gpt-oss, skill listed in the system message | yes (0.58 to 0.72) | 0 of 10 | 7/10 | 4.0 | 739 |
| qwen3.5, the same | yes | 0 of 2 | 1/2 | 4.5 | 4,880 |
| **gpt-oss, skill also named in the first message** | yes | **10 of 10** (turn 1) | **10/10** | **1.0** | **49** |
| qwen3.5, the same | yes | 3 of 3 tried `from skills import stack`, which the sandbox rejected | 2/3 | 4.0 | 4,210 |
| qwen3.5, after `skills` became importable | yes | 2 of 3 (turn 1) | 2/3 (both episodes that called it) | 3.3 | 3,747 |

For comparison, gpt-oss zero-shot stack without skills scored 4/5 on seeds 10000 to 10004, in 3.2 turns and with
930 output tokens per episode. Calling the gated program makes the episode cheap: one 4 s model call that writes
`skills.stack(robot)`. It is also more reliable, at 10/10. The same model ignored the skill as long as the skill was
described only at the end of a 14,000-character system message. Two general fixes made the difference (section 6):
naming the retrieved skills in the first message next to the instruction, and saying that they are already in the
namespace. Video: [`stack_code_llm_gptoss_calls_program_skill.mp4`](videos/stack_code_llm_gptoss_calls_program_skill.mp4).

## 5. Costs and wall time

| Workflow | Episodes or sessions | Model calls | Tokens in / out | Wall time | Cost |
|---|---|---|---|---|---|
| code.llm, gpt-oss (zero + one-shot) | 60 episodes | 268 | 1.71 M / 0.11 M | 4.1 h | $0 |
| code.llm, qwen3.5 | 6 episodes | 45 | 0.51 M / 0.04 M | 3.2 h | $0 |
| code.llm, Claude Sonnet 5.5 (CLI) | 6 episodes | 20 | 0.20 M / 0.02 M | 0.15 h | $0.98 |
| Campaign (research session + cascade) | 1 | 30 | 0.41 M / 0.007 M | 0.74 h | $0 |
| Skill reuse, all runs | 38 episodes | about 110 | 0.5 M / 0.04 M | 0.6 h | $0 |
| Program rollouts (code.program, no model) | 160 episodes in the campaign | 0 | | 22 min on 10 workers | $0 |

A promoted program needs no model at run time: about 20 to 60 s of wall time per episode (simulation on the CPU),
against 1 to 90 min per `code.llm` episode with a local model.

## 6. Bugs fixed (general fixes; none is task-specific)

| Where | Problem | Fix |
|---|---|---|
| `turns.py` | The history was capped only by turn count; Ollama's 32 k context with `--context-shift` silently drops the start of the prompt (the system message) | Older turns are dropped to keep the prompt under `max_context_chars` (80,000), and the model is told |
| `hylo-manip` `datatypes.py` | `print(masks)` dumped (240, 320) boolean arrays into the model's context (8 k characters of feedback for one print) | Compact `__repr__`s for View, CameraView, Mask, Detection, Box, Goal and Plan |
| `hylo-manip` `tools.py` | `run()` omitted `tcp_error` for joint goals, although the docstring promises it (`KeyError` in model code) | The key is always present (`None` for joint goals) |
| `prompt.py` / `agents.py` | Every model spent its first turn only looking | The first message lists the segment labels and attaches the camera images (vision models) |
| `prompt.py` | gpt-oss sent code as tool calls | The rules say there are no tools to call; code goes in the reply |
| `prompt.py` | Models called `finish()` in the same block as their motions, so a failed step could not be retried; top-down TCP poses were built wrong | The rules ask for `finish()` in a block of its own after a check; the frame note says a top-down TCP has +x along world -z |
| `reasoner-llm` `chat.py` | gpt-oss on Ollama: a reply on the tool channel came back as HTTP 500 "error parsing tool call: raw='<code>'", was retried, and after three turns ended sessions (it killed one campaign session) | The raw code is recovered as the reply (`stop_reason: tool_call_recovered`), with no retry |
| `turns.py` | Replies cut at the token limit looked like "no python block"; hidden reasoning and stop reasons were not traced | Truncation is reported to the model; `stop_reason` and `reasoning` are recorded in `agent.code.turn` |
| `research.py` | `submit(source)` needed the program as a string, and gpt-oss defined `solve` without ever submitting (10 turns lost) | `submit()` with no argument submits the imports, functions and constants the blocks defined (`Sandbox.program_source`) |
| `research.py` | The development cases were listed with the task template ("stack the <blocks> in the named order"), not their own instructions; the frozen program's metadata (used for skill retrieval) had the template too | Each case's instruction is read from its episode at session start |
| `research.py` / `generator.py` | A campaign session was not told what it should improve | `research.llm` shows the parent's program and defines it in the namespace, so the session can run it and change only what fails |
| `research.py` | The live episode timed out after 3,600 s without a motion while a slow model was thinking (`agent_timeout`, a case lost); a kernel start error surfaced as "episode ended before its first observation" | `act_wait_s` is 24 h for research and 6 h for `code.llm`; start errors are reported |
| `research.py` | Models kept acting on a finished live episode | The status says "the live episode has ended: call reset(case)" |
| `agents.py` | Skills were retrieved by the task template, so the gated `stack` skill (similarity 0.14) was never offered | Retrieval uses the episode's instruction (0.58 to 0.72) |
| `turns.py` | At temperature 0 a model looped on one block (qwen3.5 sent one reply six times) | A block repeated verbatim is pointed out in the feedback |
| `prompt.py` / `agents.py` | A retrieved skill was described only at the end of a 14 k-character system message, and no model called it (0 of 12) | The skills section says to try a fitting skill first, and the first message names the retrieved skills (then 10 of 10 calls) |
| `sandbox.py` | qwen3.5 wrote `from skills import stack`, which was rejected as a disallowed import | `import skills` and `from skills import name` resolve to the session's `skills` namespace |
| `skills.py` | `--family stack` did not name the skill (it was `grasp_manipulation_place_precision`) | The family given names the skill (`stack@1`) |
| `hylo-tools` render worker | Blender renders of an episode needed the episode's colours and fixture poses | Accepts a compiled `.mjb` (runtime edits kept) |

Tests: `uv run pytest harness/packages/hylo-manip harness/adapters/hylo-code-agents harness/adapters/hylo-reasoner-llm`
gives 295 passed, 5 skipped (the live Ollama test deselected; the count includes other agents' new tests in hylo-code-agents). New tests are in `tests/test_prompt_context.py` (context cap, scene preview, parent program,
`program_source`, a research session starting from a parent), plus one each for the reprs and for the recovery of
tool-channel replies.

## 7. What the local models can and cannot do

- **Can:**
  - gpt-oss:20b solves the pick-and-place family (pick_place, put_in_bowl, stack) zero-shot in one or two turns,
    14 of 15 episodes. qwen3.5 does the same (2 of 3), when its turns get through the queue.
  - Given a working parent, gpt-oss can extend a program and get it through a statistical gate (0.50 to 0.95 on
    unseen seeds, 1.00 on held-out).
  - Both models reuse a gated program skill once it is put in front of them: gpt-oss 10 of 10 episodes in one turn,
    qwen3.5 2 of 3.
- **Cannot (yet):**
  - Push and press, which need reasoning about TCP frames: 0 of 30 for gpt-oss, 0 of 2 for qwen3.5, against 2 of 2
    for Claude Sonnet 5.5.
  - Checking their own work before `finish()` (sort_by_colour).
  - Developing a program from scratch over a 30-turn research session.
  - Use an offered skill unless it is put in front of them: once it is (first message, no import needed), gpt-oss
    calls it in turn 1 of every episode.
- **Transfer.** A one-shot demonstration from another family did not help overall: 12/30 one-shot against 14/30
  zero-shot. The demonstration did give the only sort_by_colour success, 1 of 5, with the pick_place program as the
  demonstration.

## Interrupted episodes (not counted, rerun)

The Ollama stop (00:25 to about 01:10) cut these runs; the rows above use their reruns:

- gpt-oss zero-shot sort_by_colour, 5 episodes;
- gpt-oss one-shot pick_place, 5 episodes, and stack, 2 episodes;
- qwen3.5 zero-shot push_to_zone (after 6 turns), press_by_number and sort_by_colour;
- research sessions: qwen3.5 stack and press (15 calls each), and gpt-oss press, sort and stack.

Their traces are kept under `runs/interrupted/` and `runs/research/interrupted/`.
