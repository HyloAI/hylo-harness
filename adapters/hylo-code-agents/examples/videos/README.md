# Videos

Every video is an episode re-simulated bit for bit from its trace. A fresh simulator is reset with the recorded task
and seed and stepped with the recorded actions only, and the replayed outcome matched the recorded one for every
video here. The front camera is shown at 960x720 (the side-by-side video uses 640x480 panels).

Captions come from the same trace:

- **`code.llm`**: the turn number, the model's thought and the key lines of its code, the toolkit verbs it called,
  what it printed, and the model's reply time (queueing on the shared server included). A turn with no motion is
  held for 1.6 s so it can be read.
- **Programs**: the motions and the program's printed output.

Each video ends on its outcome, as judged by the task and never shown to the model. Renderers: MuJoCo offscreen, or
Blender Cycles on the CPU through the `render` tool (episode model as `.mjb`). Tools: `examples/llm/make_videos.py`
and `examples/llm/compare_videos.py`. Results: [`../RESULTS.md`](../RESULTS.md).

| Video | Task | Workflow | Model | Outcome | Turns | Duration | Renderer |
|---|---|---|---|---|---|---|---|
| [stack_campaign_before_after.mp4](stack_campaign_before_after.mp4) | stack, held-out seed 30080 | Campaign side by side: weak hand-written parent (left) and promoted program (right) | program written by gpt-oss:20b | left failure, right success | 0 | 23.8 s | Blender |
| [stack_campaign_promoted_program_blender.mp4](stack_campaign_promoted_program_blender.mp4) | stack, held-out seed 30080 | Research-produced program, promoted by the gate (`code.program`, no model) | written by gpt-oss:20b | success, 424 steps | 0 | 24.6 s | Blender |
| [stack_campaign_parent_weak.mp4](stack_campaign_parent_weak.mp4) | stack, held-out seed 30080 | Campaign parent, deliberately weak, hand-written | none | failure, stops after one level | 0 | 12.1 s | MuJoCo |
| [stack_code_llm_gptoss_calls_program_skill.mp4](stack_code_llm_gptoss_calls_program_skill.mp4) | stack, seed 10000 | `code.llm` with the promoted program as skill `stack@1` | gpt-oss:20b | success; its one block is `skills.stack(robot)` | 1 | 14.7 s | MuJoCo |
| [stack_code_llm_qwen_zero.mp4](stack_code_llm_qwen_zero.mp4) | stack, seed 10000 | `code.llm` zero-shot | qwen3.5:35b-a3b (vision) | success | 1 | 10.3 s | MuJoCo |
| [pick_place_code_llm_qwen_zero.mp4](pick_place_code_llm_qwen_zero.mp4) | pick_place, seed 10000 | `code.llm` zero-shot | qwen3.5:35b-a3b (vision) | success | 2 | 12.7 s | MuJoCo |
| [pick_place_code_llm_gptoss_zero.mp4](pick_place_code_llm_gptoss_zero.mp4) | pick_place, seed 10000 | `code.llm` zero-shot | gpt-oss:20b | success | 1 | 12.3 s | MuJoCo |
| [put_in_bowl_code_llm_gptoss_zero_blender.mp4](put_in_bowl_code_llm_gptoss_zero_blender.mp4) | put_in_bowl, seed 10000 | `code.llm` zero-shot | gpt-oss:20b | success | 1 | 11.7 s | Blender |
| [sort_by_colour_code_llm_gptoss_one_shot_blender.mp4](sort_by_colour_code_llm_gptoss_one_shot_blender.mp4) | sort_by_colour, seed 10002 | `code.llm` one-shot (pick_place program as demonstration) | gpt-oss:20b | success (the only sort success by any model) | 1 | 32.4 s | Blender |
| [sort_by_colour_code_llm_claude_zero_failure.mp4](sort_by_colour_code_llm_claude_zero_failure.mp4) | sort_by_colour, seed 10000 | `code.llm` zero-shot | Claude Sonnet 5.5 (CLI, text only) | failure: 2 of 3 blocks sorted when the step budget ran out | 7 | 69.2 s | MuJoCo |
| [push_to_zone_code_llm_claude_zero_blender.mp4](push_to_zone_code_llm_claude_zero_blender.mp4) | push_to_zone, seed 10000 | `code.llm` zero-shot | Claude Sonnet 5.5 (CLI, text only) | success | 4 | 23.9 s | Blender |
| [press_by_number_code_llm_claude_zero.mp4](press_by_number_code_llm_claude_zero.mp4) | press_by_number, seed 10000 | `code.llm` zero-shot | Claude Sonnet 5.5 (CLI, text only) | success | 4 | 25.4 s | MuJoCo |

Coverage, and what is missing:

- **Successful `code.llm` episodes.** Every task has one. pick_place, stack, put_in_bowl and sort_by_colour were
  solved by a local model. push_to_zone and press_by_number were solved by none of the local models (0 of 30
  episodes with gpt-oss, 0 of 2 with qwen3.5), so those two videos show the Claude comparison runs.
- **Research-produced program episodes.** These exist for stack only (the campaign). None of the six from-scratch
  research sessions (gpt-oss and qwen3.5 on stack, press_by_number and sort_by_colour) produced a program that
  solved a development case, so there is no frozen program to show for the other tasks (`RESULTS.md`, section 2).

Every file is at most 2.6 MB (H.264, CRF 23 or a capped bitrate).
