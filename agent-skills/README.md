# Skills for coding agents

These folders teach a coding agent (Claude Code, Codex, or any agent that reads `SKILL.md` files) to drive Hylo
from a terminal. Each one covers a single command family: when to use it, the exact commands, what it writes and
how to read the result.

| Skill | Command | Use it to |
|---|---|---|
| [hylo-run](hylo-run/SKILL.md) | `hylo run`, `hylo trace` | run an experiment config and read the episodes |
| [hylo-research](hylo-research/SKILL.md) | `hylo research` | have a chat model develop a `solve(robot)` program on development cases |
| [hylo-evolve](hylo-evolve/SKILL.md) | `hylo evolve` | put candidates through the gate and promote or roll back |
| [hylo-tools](hylo-tools/SKILL.md) | `hylo tool` | segmentation, depth, gravity, meshes, convex parts, hands, human motion, text-to-motion, rendering |
| [hylo-scene](hylo-scene/SKILL.md) | `hylo scene` | write a MuJoCo scene from a task sentence, check it, solve it |
| [hylo-replay](hylo-replay/SKILL.md) | `hylo replay` | turn a recorded episode or a video into a simulated replay |
| [hylo-motion](hylo-motion/SKILL.md) | `hylo motion` | text, a pose image or a video into a checked humanoid motion |
| [hylo-world](hylo-world/SKILL.md) | `hylo world`, `hylo stage`, `hylo mcp`, `hylo roles` | build a scene, run robots of 24 packs in it, check goals, record videos and datasets |
| [hylo-agent-runs](hylo-agent-runs/SKILL.md) | `hylo preflight`, `hylo watch`, `hylo seal` | run the tool-calling agent, follow and steer it, seal the evidence |

## Install

Claude Code reads skills from `.claude/skills/` in a project, or from `~/.claude/skills/` for every project:

```bash
mkdir -p .claude/skills && cp -r harness/agent-skills/hylo-* .claude/skills/
```

Other agents can be pointed at the folder directly; each `SKILL.md` stands on its own.

## Conventions shared by all skills

- Commands run from the workspace root with `uv run hylo ...` (or `.venv/bin/hylo` after `uv sync`).
- MuJoCo cameras need an off-screen GL: `MUJOCO_GL=egl`, or `MUJOCO_GL=osmesa` on machines without a GPU.
- Everything runs on the CPU. Tool workers create their own environments on first use (`uv run --script`), so
  the first call to a tool is slow.
- A chat model is given as a spec: JSON (`{"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": true}`) or
  a shorthand (`ollama:gpt-oss:20b`, `anthropic:<model>`, `openai:<model>`, `claude_code`, `codex`).
- Never read the gate or held-out results back into a model's prompt: those splits exist to measure what the
  development cases cannot.
