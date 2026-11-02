---
name: hylo-world
description: Build a robot scene as a stage (revisioned entities), run robots from 24 packs in a MuJoCo world (arms, grippers, hands, quadrupeds, humanoids with walking policies, drones, a mobile manipulator), check goals, record videos and camera datasets. Use when asked to set up a simulated workspace, try a robot, make a demo video or collect RGB-D data — through `hylo world`, `hylo stage`, `hylo mcp` or `hylo roles`.
---

# Stage worlds

A stage is a list of entities (shapes, meshes, robots from packs, cameras, lights, people) edited by patches with
revisions. A world is a MuJoCo simulation built from one revision. Every robot action names the world generation it
was decided on and comes back with a receipt that says what the simulation measured.

## Commands

```bash
# a model builds and runs the scene (vision models also see a view after every action)
MUJOCO_GL=osmesa uv run hylo world chat "a table with a red cube, a Panda that picks it up, a Go2 walking past" \
    --stage lab --root runs/stages --vision --out runs/world-chat/lab
uv run hylo watch runs/world-chat/lab                    # follow the session in the terminal
uv run hylo watch --say "use a blue cube instead" --inbox runs/world-chat/lab/inbox.jsonl   # needs --inbox at start

uv run hylo world tools                                  # every tool with its arguments
uv run hylo stage show lab --root runs/stages            # the stage as JSON; history, restore, patch as well
uv run hylo mcp --surface mujoco.world --param root=runs/stages --param stage=lab   # the same tools over MCP
uv run hylo roles run campaign.yaml                      # worker / verifier roles over dev and held-out tasks
uv run hylo tool physicalize chair.obj --out runs/meshes/chair   # collision geometry for a mesh entity
```

## The tools (also over MCP)

| Tool | What it does |
|---|---|
| `stage_show`, `stage_history` | the entities and the revision a `stage_edit` needs |
| `stage_edit` | add / remove / update / reparent entities against `expected_revision`; `stage_restore` |
| `packs`, `policies` | robots by family with their controls; pretrained walking policies and whether they match a robot |
| `world_build`, `world_reset`, `world_state` | compile the newest revision (new generation), start over, read everything |
| `act` | robot actions (several robots at once): joints, gripper, tcp, walk, fly, base, policy, hold |
| `advance` | let time pass (people walk, objects settle); `record` on `act` / `advance` writes an mp4 |
| `capture`, `view`, `pixel_to_world` | camera images with depth and segmentation by entity; a free view; pixels to points |
| `check`, `gate`, `overlaps` | goal predicates with reason codes; refuse impossible goals; penetrations |
| `export` | RGB png, depth npy, segmentation png, camera pose and world state per frame (`samples.jsonl`) |

## Getting a scene right

- Sizes are half extents for boxes; a table top at height h with half thickness t has its centre at h - t; an
  object on it has its centre at h + its half height.
- A shape or mesh moves only with `"body": {"dynamic": true}`; `world_build` lists `fixed_in_place` entities.
- Free-base robots (quadrupeds, humanoids, drones) take x, y and yaw from the pose; their height comes from the
  pack's keyframe.
- `tcp` keeps the current tool orientation unless `quat_wxyz` is given (an arm at home points its hand down).
  Grasp from above: above the object, down, close the gripper, up; `holding` in the gripper receipt says whether
  the fingers stopped on something.
- A humanoid stands and walks only under a policy: pack `g1_legs` with `{"kind": "policy", "policy": "g1_walk",
  "command": [vx, vy, yaw_rate]}` (or `h1_legs` / `h1_walk`).
- Verify with `check` on the goal you were asked for. Do not weaken a goal until it passes: report what failed.

## Reading the result

- Receipts: `status` (completed or refused with a reason: stale, unsupported, unreachable, mismatch, busy) and
  `measured` (joint error, opening and holding, distance walked and turn, flight error, policy speed and upright).
- A chat session's trace is in `--out/episodes/*.jsonl` (hash-chained: `hylo seal --chain DIR`).
