# Free-form scenes

The scene builder's free mode. Instead of filling in the fixed `hylo.scene/v0.1` format (primitives on a Panda's
table and a small predicate language), the model writes the task as a Python **scene program**
(`hylo.freescene/v0.1`). The program:

- picks the robot: Panda, the two ALOHA arms, or the G1 humanoid;
- builds the world with MuJoCo's model editor or MJCF, using any feature MuJoCo compiles except actuators and
  keyframes (cables, cloth, tendons, equality constraints, hinged and sliding parts);
- defines `success(sim)` over the simulation state;
- may define `demo(sim)`, which proves the task can be done using invisible hands.

The environment `mujoco.free_scene` runs the program around the existing robot environments, unchanged.

```bash
uv run hylo scene build --free "drape the cloth over the box" --out runs/scenes_free/cloth --attempts 6
uv run hylo scene check runs/scenes_free/cloth
uv run hylo scene solve runs/scenes_free/cloth --turns 15
```

## Checker

1. **program**: the source passes the scene-program allow-list. It defines `ROBOT`, `INSTRUCTION`, `build` and
   `success`.
2. **load**: the scene builds and compiles with the robot. Build errors and MuJoCo compiler errors come back with
   the program's own line numbers.
3. **physics**: checked on seeds 0, 1 and 2, with the scene left alone for 2 s:
   - no deep overlaps at the start (between scene objects, with the table, or with the robot at its home
     posture);
   - the simulation does not diverge;
   - nothing falls off the table;
   - rigid bodies stay within 2 cm and 10 degrees, and cable nodes and cloth vertices within 5 cm;
   - joints do not drift, and nothing is still moving at the end;
   - `success(sim)` runs, returns a bool, and stays False.
4. **reach**: for each movable thing (free body, cable, cloth, jointed part), and for each point `targets(sim)`
   names, IK finds a top-down or side approach for at least one arm, with a pre-pose 8 cm back.
5. **satisfiable**: `demo(sim)` drives invisible hands with the physics running. `success(sim)` must become True
   during the demo or within 1 s after the hands let go. Every point the hands grabbed must be reachable by an
   arm. A program without a demo gets an "unverified" warning.

**What the checker cannot verify:**

- that the success function means what the instruction says;
- that a real gripper can hold a thin cable or a cloth corner (the demo's hands are ideal);
- that two arms can act at the same time where a task needs both hands together (each point is checked on its own).

## Results

**Nothing has been run yet.** Every cell below is "not run yet". The numbers will come from
[`results.json`](results.json), which `run_all.py` writes. One command builds, checks, solves and records every
task, with one chat-model session at a time:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu
uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py
```

| Task (text given to the model) | Robot | Build attempts (problems) | Solved by qwen3.5 | Turns | Wall time | Scripted solution | Newton cross-check | Video | Reproduce |
|---|---|---|---|---|---|---|---|---|---|
| cable_lift: "lift the cable off the table by both of its ends, using two arms" | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | `run_all.py --tasks cable_lift` |
| cloth_drape: "drape the cloth over the box" | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | `run_all.py --tasks cloth_drape` |
| rope_line: "lay the rope straight along the white line on the table" | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | `run_all.py --tasks rope_line` |
| rope_ring: "thread the end of the rope through the ring" | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | `run_all.py --tasks rope_ring` |
| hinged_lid: "open the hinged lid of the box" | not run yet | not run yet | not run yet | not run yet | not run yet | not run yet | not applicable | not run yet | `run_all.py --tasks hinged_lid` |

The tasks are in [`tasks.json`](tasks.json). Stages: `build`, `solve`, `scripted`, `newton`, `blender`, `collect`.

- `--stages` runs some of them, and `--force` redoes stages whose output already exists.
- A **scripted solution** is a `solve(robot)` program written by hand, after the build, for the scene the model
  built (save it as `<task>/scripted.py`). `run_all.py` runs it with `code.program` and the same privileged
  toolkit. It is reported separately from the model's own solve and labelled as a scripted solution.
- The **Newton cross-check** (`newton_check.py`, run through `uv run --with newton --with warp-lang`) replays the
  program's demo and success function in Newton on the CPU. It covers the task's physics only: cloth and cables as
  particles, rigid bodies as static boxes. The robot stack stays on MuJoCo.
- The **Blender video** re-simulates the successful episode from its recorded actions, then renders it with
  `hylo tool render`.

`run_all.py --smoke` runs the same code path end to end with scripted chat models. It uses no Ollama and renders
nothing, so it checks only that the scripts work, not any result.
