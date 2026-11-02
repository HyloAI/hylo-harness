---
name: hylo-scene
description: Build a MuJoCo manipulation scene from a sentence with `hylo scene build`, check it with `hylo scene check`, and have a code agent solve it with `hylo scene solve`. Use when asked to create a new simulated task, a scene for a robot, or a benchmark case that does not exist yet.
---

# Scene builder

`build` asks a chat model to write a scene file (`hylo.scene/v0.1`: objects, their sizes, colours and poses, the
success condition) for a Panda at a table. The checker loads it in MuJoCo, settles it over several seeds and
rejects scenes where objects overlap, fall off, start in the goal or cannot be reached. The model gets the
checker's report and retries until the scene passes or the attempts run out.

## Commands

```bash
MUJOCO_GL=osmesa uv run hylo scene build "put the two red cubes into the blue tray" --out runs/scenes/red-tray
uv run hylo scene check runs/scenes/red-tray/scene.json
MUJOCO_GL=osmesa uv run hylo scene solve runs/scenes/red-tray --turns 20
```

`solve` runs a code agent with privileged state (exact object poses) on the scene. It writes the program, the
turns (`solve/turns.jsonl`), the outcome (`solve.json`) and a front-camera video `front.mp4` (skip it with `--no-video`).

## Reading the result

- `build` prints each attempt's checker verdict. `--out` gets the final `scene.json`, every attempt under
  `attempts/` and `build.json` (per attempt: problems, tokens, latency; totals; built or not).
- `solve` prints whether the success condition held and the number of turns used. A scene that the privileged agent
  cannot solve usually has an unreachable object or an ambiguous goal: re-run `hylo scene check`, then read the turns.

## Rules

- Keep scene text concrete: name the objects, colours and the goal relation (in, on, left of, stacked).
- A scene that passes the checker is physically plausible, not necessarily solvable. Run `solve` before using it
  as a benchmark case.
