---
name: hylo-motion
description: Make a humanoid motion from text, an image of a pose, or a reference video with `hylo motion`. The motion is retargeted to the Unitree G1 (or kept as a human character), checked in MuJoCo and rendered. Use when asked to animate a humanoid, generate a robot motion from a description, or copy a motion from a video.
---

# Motion studio

A chat model plans the motion from the request, calls the tools (text-to-motion, pose or human-motion
estimation), retargets the result to the G1 within its joint limits, checks it in MuJoCo (balance, foot sliding,
self-collision, limit violations) and revises until the checks pass or the budget runs out.

## Commands

```bash
MUJOCO_GL=osmesa uv run hylo motion "wave with the right hand, then bow" --robot g1 --out runs/motion/wave-bow
MUJOCO_GL=osmesa uv run hylo motion "strike this pose" --image pose.jpg --out runs/motion/pose
MUJOCO_GL=osmesa uv run hylo motion "copy the dance" --video dance.mp4 --side-by-side --out runs/motion/dance
```

Useful flags: `--style blender` for a Cycles render, `--budget turns=8`, `--steps` for the text-to-motion
denoising steps (fewer is faster), and `--threads` for the CPU threads per tool.

## Output

`--out` holds `motion.npz` (joint trajectories), `check.json` (retargeting, kinematic metrics, physics rollouts),
`log.jsonl` (turns and tool calls), `motion.mp4` (plus `split.mp4` with `--side-by-side`) and `summary.json`
(turns, cost, wall time, time per tool).

## Rules

- Text-to-motion on the CPU takes minutes per clip; run long requests in the background.
- Report the check results with the video, especially any balance or limit failures.
