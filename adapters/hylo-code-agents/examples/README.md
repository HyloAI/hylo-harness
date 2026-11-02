# Tool-using agents

Agents that act by writing Python against a small set of robot and perception tools, instead of outputting joint
commands directly (RFC-0006). Each tool call that moves the robot goes through the kernel and its safety shield, and
every model turn and every tool call is a trace event. This page collects the workflows that are finished, with the
numbers their recorded runs gave, the videos those runs produced, and what does not work yet.

Common setup, unless a section says otherwise:

- one shared 128-thread CPU workstation, no GPU;
- local language models through Ollama: `qwen3.5:35b-a3b` (vision) and `gpt-oss:20b` (text only), at $0;
- the task's score is never shown to the model;
- every robot video is re-simulated or played back from the recorded run (its actions or joint trajectory); none
  is a separate rollout.

| Workflow | Headline | Section |
|---|---|---|
| Robot toolkit and programs | Six hand-written Panda programs that use only cameras and segmentation: 1.00 on every task (20 validation episodes each) | [Robot toolkit](#robot-toolkit) |
| Two arms and a humanoid | ALOHA (two arms) and Unitree G1 (dexterous hands): camera-only programs solve 3 of 3 two-arm tasks at 1.00; G1 programs 0.80, 0.50 and 1.00 | [Two arms and a humanoid](#two-arms-and-a-humanoid) |
| Program improvement | A program written by gpt-oss:20b passes the gate: 0.95 against 0.50 for its parent (p = 0.002), and 1.00 against 0.60 on held-out | [Program improvement](#program-improvement) |
| Scene builder | 6 of 6 scenes written from text pass the checker; 3 of 6 are then solved by qwen3.5 with privileged state | [Scene builder](#scene-builder) |
| Free-form scenes | The model writes scene programs (any robot, cables, cloth, joints) checked by building, settling, reach and a scripted demo; not run yet | [Free-form scenes](#free-form-scenes) |
| Replay to simulation | All 5 of our own Panda episodes reproduce in physics (2.7 to 8.0 mm mean object error); 0 of 5 external episodes do | [Replay to simulation](#replay-to-simulation) |
| Motion studio | 6 requests (text, image, video) become kinematically clean Unitree G1 motions; none stays balanced under physics (five fall within 1.7 s) | [Motion studio](#motion-studio) |
| Motion characters and free mode | Eight target characters (five Menagerie humanoids, a capsule figure, two CC0 rigged characters) and a free mode where the model writes its own pipeline; not run yet | [Motion characters and free mode](#motion-characters-and-free-mode) |
| LIBERO robot contract | Programs: 0.98 on libero_spatial, 0.96 on libero_object, 0.81 on libero_goal, 0.66 on libero_10 (SmolVLA: 0.70, 0.66, 0.78, 0.42); qwen3.5 writing code: 4 of 7 | [LIBERO](#libero) |
| Perception and motion tools | 9 tool adapters (SAM 3, MoGe-2, GeoCalib, Kimodo, Blender, ...) run on the CPU; 3 of them only on their fallbacks | [Perception and motion tools](#perception-and-motion-tools) |
| Live code-as-policy runs | Local models solve pick-and-place-type tasks zero-shot (gpt-oss:20b 14/15); with the promoted skill offered, gpt-oss solves 10/10 stack episodes in one turn | [Live code-as-policy runs](#live-code-as-policy-runs) |
| Tool-calling agents | qwen3.5 through typed tools, one checked action at a time: 8 of 9 episodes on pick_place, put_in_bowl and stack; hash-chained, sealed runs; seven LIBERO perturbations | [Tool-calling agents](#tool-calling-agents) |
| Worlds | 24 robot packs in stages a model edits: four quadrupeds trot, two humanoids walk under pretrained policies, two drones reach waypoints, an arm picks onto a plate, people walk around obstacles | [Worlds](#worlds) |

Rendering on a CPU machine needs OSMesa; every command below assumes:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu
```

## Robot toolkit

`hylo-manip` gives a program one object, `robot`, with stateless verbs over any robot described by a `RobotSpec`:
`look`, `find` (masks for text labels), `objects`, `box`, `grasps`, `place_at`, `plan` (IK plus a collision-free
path) and `run`, the only verb that moves the robot. The six reference programs in [`programs/`](programs/) each
define `solve(robot)` for one `mujoco/manip-v0` task. They see the front, side and wrist cameras with the
environment's segmentation and no privileged object state, and they run as `code.program` (no model) through
`hylo run`.

```bash
uv run python harness/adapters/hylo-code-agents/examples/programs/run_programs.py --out runs/manip-programs
```

| Task | Success | Episodes | Steps (mean) | Wall s per episode |
|---|---|---|---|---|
| pick_place | **1.00** | 20 | 141 | 23.7 |
| stack | **1.00** | 20 | 236 | 57.5 |
| put_in_bowl | **1.00** | 20 | 148 | 24.2 |
| push_to_zone | **1.00** | 20 | 93 | 24.0 |
| press_by_number | **1.00** | 20 | 235 | 38.3 |
| sort_by_colour | **1.00** | 20 | 596 | 184.4 |

Validation split, seeds 10000 to 10019; 322 s of wall time for all 120 episodes with 48 workers.

<video src="assets/program_sort_by_colour.mp4" poster="assets/program_sort_by_colour.jpg" controls muted loop playsinline width="100%"></video>

*The reference `sort_by_colour` program on validation seed 10015, "sort the blocks by colour into the matching
trays": the orange block goes into the orange tray, then both red blocks into the red tray (11 motions, 433 steps,
success). The caption numbers the motions and shows what the program printed: the trays and blocks it found with
the cameras. Rendered with Blender Cycles on the CPU from the recorded joint positions.*

**Limitations.**

- The programs were written by hand while looking at these tasks, so they show what the toolkit can do, not what an
  agent finds on its own.
- Segmentation comes from the simulator. It is a declared observation and part of the config digest, but it is
  still privileged perception.
- These numbers are for one robot, the Franka Panda, on one benchmark.

## Program improvement

A campaign on `stack` starts from a hand-written parent that is deliberately weak: it stacks two blocks only and
never retries. The generator `research.llm` gives gpt-oss:20b a research session with three development cases, in
which it rewrites the parent. The campaign's gate then compares the new program with the parent on unseen initial
states. The model never sees the gate or held-out results. Full details are on
[`campaign/README.md`](campaign/README.md).

```bash
uv run hylo evolve run harness/adapters/hylo-code-agents/examples/campaign/campaign.yaml
uv run hylo evolve status runs/campaign/campaign
```

| Step | Split (episodes) | Candidate | Parent | Wins / losses (pairs) | p |
|---|---|---|---|---|---|
| Research session | search, 3 dev cases | 3/3 (each of 3 submissions) | | | |
| Stage 1: early rejection | first 10 of search_val | | | 6 / 0 (10) | |
| Stage 2 | search_val (20) | 1.00 | 0.30 | 14 / 0 (20) | 6.1e-5 |
| **Gate** (McNemar, alpha 0.05) | gate (20) | **0.95** | 0.50 | 9 / 0 (20) | **0.002** |
| Held-out report | heldout (20) | **1.00** | 0.60 | 8 / 0 (20) | 0.0039 |

The candidate was promoted as sealed artifact `0a1ee96b2f810ab6`. The campaign took 44 min (a 22 min session and
22 min of rollouts) and used 30 model calls with 406,698 input and 7,082 output tokens, for $0.

<video src="assets/campaign_before_after.mp4" poster="assets/campaign_before_after.jpg" controls muted loop playsinline width="100%"></video>

*Held-out seed 30080, "build a tower: the blue block at the bottom, then the green block, then the orange block on
top". Left: the parent stops after one level (failure, 177 steps). Right: the promoted program builds all three
levels (success, 424 steps). Blender Cycles.*

**Limitations.**

- This is one campaign on one task, against a parent that was weakened on purpose.
- From scratch, the local models did not get this far: three gpt-oss research sessions without a parent produced no
  program that solved a development case (see [`RESULTS.md`](RESULTS.md), section 2).
- The promoted program wastes a motion: it lifts the bottom block and drops it back before stacking on it. The gate
  did not penalise this.

## Scene builder

A model writes a `hylo.scene/v0.1` file from a one-line task: objects, their shapes and start regions, and a success
predicate. A checker runs five stages on it: schema; static (names and references); physics (the scene loads and
settles, nothing falls or penetrates, and the predicate is false at the start); reach (every object has a grasp the
arm reaches, and every place target is reachable); and satisfiable (objects placed where the predicate asks make it
true). The problems it finds go back to the model until the scene passes. `code.llm` then solves the scene with privileged state: `robot.scene()`
returns the simulator's object poses and joint states.

```bash
uv run hylo scene build "put the ball in the cup" --out runs/scenes/ball_cup --attempts 6
uv run hylo scene solve runs/scenes/ball_cup --turns 10 --reply-tokens 2500
```

Model: qwen3.5:35b-a3b (vision) for both building and solving, from [`scenes/results.json`](scenes/results.json).

| Task | Build attempts (problems in the failed ones) | Solved | Turns | Steps | Model time |
|---|---|---|---|---|---|
| size_tower: "stack the three blocks into a tower ordered by size: the largest block at the bottom and the smallest on top" | 2 (a timed-out model call) | **yes** | 6 | 423 | 66 min |
| ball_cup: "put the ball in the cup" | 2 (timeout) | **yes** | 3 | 159 | 37 min |
| slide_lid: "slide the lid of the box open so that the box is uncovered" | 3 (timeout; predicate true at start) | **yes** | 2 | 87 | 17 min |
| drawer_block: "open the drawer and put the red block in it" | 6 (unstable; predicate true at start, twice; unsatisfiable; unstable) | no | 10 | 448 | 88 min |
| cylinder_upright: "a cylinder is lying on its side on the table; stand it upright on top of the wooden box" | 5 (schema; timeout, twice; predicate true at start) | no | 5 | 160 | 67 min |
| colour_row: "line up the red, green and blue blocks in a row along the y axis, in that order" | 2 (timeout) | no | 4 | 483 | 15 min |

All 6 scenes were built and checked; 3 of 6 were solved. Model time includes queueing on the shared Ollama server.

<video src="assets/scene_size_tower.mp4" poster="assets/scene_size_tower.jpg" controls muted loop playsinline width="100%"></video>

*size_tower: the scene the model wrote (blue large, green medium and red small block), solved in 6 turns and 423
steps. The model moves the large block to the centre, stacks the medium one on it, then places the small one on top.
The episode ends with the hand still over the tower, so the small block is partly hidden by the fingers. The
caption shows each turn's thought, code and printed result.*

<video src="assets/scene_ball_cup.mp4" poster="assets/scene_ball_cup.jpg" controls muted loop playsinline width="100%"></video>

*ball_cup: the red ball picked up and dropped into the fixed blue cup, in 3 turns and 159 steps.*

<video src="assets/scene_slide_lid.mp4" poster="assets/scene_slide_lid.jpg" controls muted loop playsinline width="100%"></video>

*slide_lid: the gripper pushes the lid along its slide joint until it overhangs the front of the box (the predicate
asks for at least 4 cm of travel), in 2 turns and 87 steps.*

The three videos are re-simulated on a fresh `mujoco.panda_scene` from the recorded actions. Each re-simulation
gives the recorded task digest, step count and outcome, and the frames are rendered with Blender Cycles at 960x720.

**Limitations.**

- 3 of the 6 scenes were solved, and only with privileged state.
- The failures are the model's: it never opened the drawer, it did not turn the cylinder upright, and it lined the
  blocks up in the wrong direction. In the last two it called `finish()` without checking.
- Single model calls took up to 26 min on the shared server, mostly queueing, and several builds lost an attempt to
  a timed-out call. The wall times say little about the method.

## Free-form scenes

Free mode of the scene builder: the model writes the task as a Python scene program (`hylo.freescene/v0.1`) in
place of a scene file. The program chooses the robot (Panda, ALOHA or G1), builds the world with MuJoCo's model
editor or MJCF (including cables, cloth, tendons, equality constraints and articulated parts), and defines its own
success function and a demo with invisible hands. The checker builds the scene, settles it on three seeds, checks
reach for every arm, and runs the demo. Details, the checker's limits and the task list are in
[`scenes_free/README.md`](scenes_free/README.md).

```bash
uv run hylo scene build --free "drape the cloth over the box" --out runs/scenes_free/cloth
uv run python harness/adapters/hylo-code-agents/examples/scenes_free/run_all.py     # every showcase task
```

The showcase runs have not been run yet: every result in [`scenes_free/results.json`](scenes_free/results.json) is
"not run yet".

## Replay to simulation

From one recorded episode (ours, a LeRobot dataset episode, or a human video), the pipeline names the objects with
a vision model, masks them with SAM 3, and lifts them to 3D with recorded depth or MoGe-2. It then tracks them,
builds collision meshes (mesh and convex-decomposition tools), places them in a MuJoCo scene with the Panda (or a
free-floating gripper for a human hand), and replays the motion in physics. The checks compare simulated object
poses with the tracked ones and match contact events (grasp, lift, place) to the recording.

```bash
uv run hylo replay panda:stack:3 --out runs/replay/panda_stack_3            # one of our episodes, recorded now
uv run hylo replay hf:physical-intelligence/libero:1280 --profile libero --camera image --out runs/replay/libero_1280
```

From [`replay/results.json`](replay/results.json). Object error is the simulated pose against the tracked pose,
mean / final, in mm.

| Source | Episode | Events matched | Final state reproduced | Object error, mean / final |
|---|---|---|---|---|
| ours, Panda | pick_place 1 | 3/3 | **yes** | 6.0 / 2.1 |
| ours, Panda | pick_place 2 | 3/3 | **yes** | 5.8 / 0.4 |
| ours, Panda | stack 3 | 6/6 | **yes** | 2.7 / 5.4 and 4.7 / 3.1 (two blocks) |
| ours, Panda | put_in_bowl 4 | 3/3 | **yes** | 8.0 / 5.0 |
| ours, Panda | push_to_zone 5 | no events | **yes** | 3.3 / 10.5 |
| LIBERO | 1280 | 1/2 | no | 21 to 45 / 34 to 75 (the bowl to move got no mask) |
| LIBERO | 379 | none | no | registration failed (scale 0.057) |
| LIBERO | 813 | 0/2 | no | 9.4 / 19.8 (basket; no carried object found) |
| DROID | 21 | 3/3 | no | 53 / 97 |
| human video | mug 0 | 0/2 | no | 69 / 120 |

For our episodes, where ground truth is known, the largest error of a reconstructed object's size is 5.3 to 10.6 mm
per episode.

With `--agent`, a model drives the same steps instead of the fixed order. qwen3.5:35b-a3b ran pick_place 2 in 12 turns
(1,558 s of model time, 28,411 input and 1,104 output tokens, $0) and called the steps in the suggested order. It
reproduced the outcome as the fixed pipeline did: the same labels and masks, 3/3 events, a simulated object error of
6.1 mm mean and 1.2 mm final against the ground truth (5.9 mm and 0.7 mm for the fixed pipeline). The run shows the
agent loop working, not the model correcting anything: it kept every default, never looked at the overlays, skipped
the video step, and described its events as matched to the ground truth when they were matched to the recording.

<video src="assets/replay_panda_stack.mp4" poster="assets/replay_panda_stack.jpg" controls muted loop playsinline width="100%"></video>

*Panda stack 3, "build a tower: the blue block at the bottom, then the green block, then the yellow block on top".
Left: the recording. Right: the physics replay, in which the blocks move only by contact. 6 of 6 events matched,
and the tower stands at the end.*

<video src="assets/replay_libero_failure.mp4" poster="assets/replay_libero_failure.jpg" controls muted loop playsinline width="100%"></video>

*A failure: LIBERO episode 1280. The registration scale came out at 0.39, the replayed robot (right) does not line
up with the reconstructed scene, and the black bowl it should move was never masked.*

Both videos are rendered with MuJoCo, on purpose: the replay uses the recording's camera, so the two views line up.

**Limitations.**

- Only our own Panda episodes reproduce. They come from our simulator, with recorded depth and a known camera pose.
- No external episode (3 LIBERO, 1 DROID, 1 human video) reproduces. Monocular depth, registration and the
  hand-to-gripper mapping are each off by centimetres or more.

## Motion studio

A request (text, an image of a pose, or a reference video) becomes a motion for the Unitree G1 humanoid. A model
writes Python against motion tools: NVIDIA Kimodo for text to motion, MediaPipe for pose from images and video,
motion edits, retargeting to the G1, a check, and a renderer. The check reports kinematic problems
(self-collision, joint limits, ground penetration, foot skate) and, separately, a physics rollout in which position
servos track the trajectory with no balance controller.

```bash
uv run hylo motion "wave with the right hand, then bow" --robot g1 --side-by-side --out runs/motion/wave_bow
uv run hylo motion "strike the pose shown in the image and hold it for a second" --image pose.jpg --robot g1 \
    --out runs/motion/image_pose
```

Model: qwen3.5:35b-a3b (vision). From [`motion/runs.json`](motion/runs.json). "Upright" is how long the G1 stays up
in the physics rollout; "with ankle strategy" is the same rollout with the check's ankle strategy switched on.

| Request | Turns | Wall min | Clip | Keypoint error | Upright | With ankle strategy | How it falls |
|---|---|---|---|---|---|---|---|
| "wave with the right hand, then bow" | 2 | 35 | 5.0 s | 5.6 cm | 1.48 s | 4.35 s | right wrist touches the floor |
| "walk forward four steps and turn around" | 3 | 38 | 9.0 s | 4.5 cm | 1.22 s | 1.13 s | pelvis drops below 40% of its start height |
| "a short dance" | 2 | 6 | 4.0 s | 3.9 cm | 1.68 s | 2.68 s | right knee touches the floor |
| "jump in place twice" | 2 | 23 | 5.0 s | 6.8 cm | 1.44 s | 1.70 s | pelvis drops |
| "do what the person in the reference video does" (jumping jacks) | 6 | 95 | 2.8 s | 4.2 cm | 1.05 s | 1.12 s | left knee touches the floor |
| "strike the pose shown in the image and hold it for a second" | 3 | 46 | 1.0 s | 4.2 cm | 0.97 s (whole clip) | 0.97 s | no fall within the clip, but the centre of mass is outside the support in every frame |

All six motions have no self-collision and no joint-limit violation.

<video src="assets/motion_wave_bow_split.mp4" poster="assets/motion_wave_bow_split.jpg" controls muted loop playsinline width="100%"></video>

*"wave with the right hand, then bow": the generated human motion (left) and its retargeting to the G1 (right).
Kinematic playback, Blender Cycles.*

<video src="assets/motion_walk_turn.mp4" poster="assets/motion_walk_turn.jpg" controls muted loop playsinline width="100%"></video>

*"walk forward four steps and turn around": 9 s, 5.9 m of path, about 175 degrees of turn. Kinematic playback; under
physics this motion falls after 1.2 s.*

**Limitations.**

- Every motion is kinematically clean, but none stays balanced under physics. With no balance controller, five of
  the six fall within 1.7 s (within 4.35 s with the ankle strategy). The 1 s held pose does not fall within its clip,
  but its centre of mass is outside the support the whole time.
- The videos are kinematic playback, not the physics rollout.

## Motion characters and free mode

The motion studio's final motion can be retargeted to any humanoid described by a short mapping file. There are
eight: Unitree G1 and H1, Booster T1, Apptronik Apollo and Fourier N1 from MuJoCo Menagerie, Hylo's capsule figure,
and two CC0 rigged characters (Kenney, Quaternius) driven in Blender. Every character gets the source's proportions
and reports its keypoint error, limb angle error and joint-limit hits; `--characters all` shows the final motion
on all of them in one video grid. In free mode (`hylo motion --free`) the model writes its own scripts in the run
directory, in a guarded process, calling the studio's tools through a kit and doing its own processing with numpy
and scipy.

The code and its tests are finished; the five showcase requests have not been run yet. Commands, characters,
licences and design: [`motion_free/README.md`](motion_free/README.md).

The G1 runs above used one uniform scale for the whole skeleton. Retargeting now gives each character its own
proportions, so their keypoint errors are not directly comparable with new runs.

## Two arms and a humanoid

The same toolkit drives robots with more than one arm and with multi-finger hands. Two environments on MuJoCo
Menagerie models (fetched at a pinned commit, never vendored): `mujoco.aloha_table`, two arms at a table, and
`mujoco.g1_table`, the Unitree G1 with its three-finger hands, pelvis fixed, working at a table. Programs pass
`arm=` to the arm-specific verbs, and `robot.run({"left": plan_l, "right": plan_r}, sync=True)` runs both arms in
one motion, checked at every step for the arms touching each other. For the hands, `hand_grasps(..., kind="power")`
or `kind="pinch"` proposes multi-finger grasps.

```bash
uv run python harness/adapters/hylo-code-agents/examples/programs/run_programs.py --suite bimanual --out runs/bimanual-programs
uv run python harness/adapters/hylo-code-agents/examples/programs/run_programs.py --suite humanoid --out runs/humanoid-programs
```

| Task | What makes it a two-arm or hand task | Privileged oracle | Camera-only program |
|---|---|---|---|
| ALOHA handover | the bar starts where only one arm reaches; the zone is where only the other reaches | 1.00 | **1.00** |
| ALOHA lift_box | the box is too wide for one gripper; lifted level by its two handles for one second | 1.00 | **1.00** |
| ALOHA two_bins | a 260-step limit; one arm after the other takes 407 to 489 steps | 1.00 | **1.00** |
| G1 pick_place | a three-finger power grasp | 1.00 | 0.80 |
| G1 two_hands | one block per hand at the same time, 340-step limit (one after the other: 408 to 424 steps) | 0.90 | 0.50 |
| G1 press_button | press a button with the hand | 0.90 | **1.00** |

Validation split, 20 episodes per task. After the toolkit gained these features, the six Panda programs still score
1.00 on all tasks.

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<video src="assets/bimanual_handover.mp4" poster="assets/bimanual_handover.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/bimanual_two_bins.mp4" poster="assets/bimanual_two_bins.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/humanoid_pick_place.mp4" poster="assets/humanoid_pick_place.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/humanoid_two_hands.mp4" poster="assets/humanoid_two_hands.jpg" controls muted loop playsinline width="100%"></video>
</div>

*Top left: ALOHA handover; the left arm brings the red bar to the middle, the right arm takes its free end and puts
it in the orange zone (Blender Cycles). Top right: two_bins; both arms place their blocks at the same time. Bottom
left: G1 pick_place, the green block onto the plate (Blender Cycles). Bottom right: G1 two_hands, one block per hand.
All are camera-only programs on validation seeds, re-simulated from the traces.*

**Limitations.**

- No two-hand lift on the G1: the three-finger hands could not hold a box level (squeezed between the palms, or by
  flat or bar handles), so that task was replaced by two_hands.
- The G1 presses with its closed hand; pointing one finger down was reachable over too little of the table.
- The G1 hands needed physics adjustments: no collision between a hand's own palm and fingers, two finger servos
  capped at half their torque, and gravity compensation on the arms. The ALOHA bar is light (30 to 60 g) with extra
  torsional friction, so an end grasp holds it.
- The G1 reaches only a narrow strip of the table (x 0.22 to 0.32 m, |y| up to 0.25 m), and its programs are weaker
  than the oracle: two_hands at 0.50 mostly misses the first grasp of one hand and then runs out of steps.
- The programs were written by hand while looking at these tasks; segmentation comes from the simulator.

## LIBERO

The LIBERO adapter has a second mode for tool-using agents: joint-position control plus the robot contract (depth,
camera poses, segmentation, and a Panda model exported from the simulator). The same toolkit therefore runs on
LIBERO scenes unchanged. Four programs (pick and place, open a drawer, turn on the stove, drawer then place) cover
the task types, and `code.llm` runs zero-shot. Full details are on
[`hylo-env-libero/examples/README.md`](https://github.com/HyloAI/hylo-bench/blob/main/adapters/hylo-env-libero/examples/README.md).

```bash
uv run python bench/adapters/hylo-env-libero/examples/programs/run_programs.py --out runs/libero-programs --n 10
uv run python bench/adapters/hylo-env-libero/examples/run_code_llm.py run --out runs/libero-code-llm \
    --tasks libero_spatial:0 libero_object:0 --episodes 3 --parallel 2
```

| Suite | Programs (10 init states per task) | SmolVLA (recorded, 50 episodes) |
|---|---|---|
| libero_spatial | **98/100 = 0.98** | 0.70 |
| libero_object | **96/100 = 0.96** | 0.66 |
| libero_goal | **81/100 = 0.81** (push the plate: 4/10) | 0.78 |
| libero_10 | **66/100 = 0.66** | 0.42 |

`code.llm` with qwen3.5:35b-a3b, zero-shot, at most 16 turns:

| Task | Success | Turns (mean) |
|---|---|---|
| libero_object: pick up the alphabet soup and place it in the basket | **3/3** | 2.7 |
| libero_spatial: pick up the black bowl between the plate and the ramekin and place it on the plate | 0/3 | 9.7 |
| libero_goal: put the wine bottle on the rack | **1/1** | 5 |

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<video src="assets/libero_program_drawer.mp4" poster="assets/libero_program_drawer.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/libero_llm_soup.mp4" poster="assets/libero_llm_soup.jpg" controls muted loop playsinline width="100%"></video>
</div>

*Left: the `open_drawer.py` program on libero_goal "open the middle drawer of the cabinet" (init state 0, 88 steps,
success). Right: qwen3.5 writing code zero-shot on libero_object "pick up the alphabet soup and place it in the
basket" (init state 1, 3 turns, 165 steps, success). Both are re-simulated from the traces at 512x512.*

**Limitations.**

- The comparison with SmolVLA is not like for like. The programs use the simulator's segmentation (privileged) and
  were written while looking at these tasks; SmolVLA sees RGB only.
- "Open the top drawer and put the bowl inside" fails 0/10 (the opened drawer blocks every approach to the bowl).
- libero_10: four tasks at 10/10, but the microwave task fails 0/10 (no skill for putting an object inside an
  enclosure and closing its door), and flat boxes (cream cheese, butter) and moka pots slip out of the grasp
  (3/10, 2/10 and 6/10 on the tasks that need them). A grasp mode for flat objects is missing from the toolkit.
- Push the plate: 4/10; the fingertips slide over the low rim, or the plate ends 6 to 12 cm short.
- `code.llm` ran on only 7 episodes, and failed all three on the bowl task.

## Perception and motion tools

Each tool is a model wrapped as a worker in its own uv environment, so torch and the other heavy dependencies never
enter the Hylo environment. Every result records the model, its revision and licence, and its timing. All of them
were run on this workstation's CPU (32 threads). Full details are on
[`hylo-tools/README.md`](../../hylo-tools/README.md).

```bash
uv run hylo tool list
uv run hylo tool segment frame.png -p basket -p "soup can" --out out/
uv run hylo tool text_motion "a person walks forward and waves" --duration 5 --out out/
```

| Tool | Model on the CPU | CPU time (32 threads) | Status |
|---|---|---|---|
| segment | SAM 3 | 3 prompts on a 387x387 frame: 11.9 s wall | works |
| depth | MoGe-2 | 480x480: 4.3 s inference, 11.2 s with loading | works |
| gravity | GeoCalib | 480x480: 0.5 s inference, 4.3 s with loading | works |
| mesh | SAM 3D Objects (gated); fallback MoGe-2 depth, carving and marching cubes | fallback: 11.3 s | fallback only (no access to SAM 3D yet) |
| convex | CoACD | open cup: 2.5 s, 28 parts | works |
| hands | WiLoR + MANO; fallback MediaPipe Hand Landmarker | 32 frames: 1.1 s (MediaPipe) | fallback only (MANO not registered) |
| human | GVHMR + SMPL; fallback MediaPipe Pose Landmarker | 24 frames: 1.4 s (MediaPipe) | fallback only (SMPL not registered, no CPU port) |
| text_motion | NVIDIA Kimodo (Llama 3 8B text encoder) | 5 s clip: 57 s wall | works |
| render | Blender 5.2.3 Cycles; fallback MuJoCo | 30 frames at 480x272: 11.4 s | works |

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<img src="assets/tool_segment_libero.png" alt="SAM 3 masks with scores for basket, soup can and milk carton on a LIBERO frame" width="100%">
<img src="assets/tool_depth_libero.png" alt="A LIBERO frame, its MoGe-2 depth map and its normal map" width="100%">
</div>

*Left: SAM 3 masks for the prompts "basket", "soup can" and "milk carton" on a LIBERO frame. The "milk carton"
prompt also matches two other boxes, at scores 0.61 to 0.73. Right: MoGe-2 depth and normals for the same scene.*

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<video src="assets/tool_text_motion.mp4" poster="assets/tool_text_motion.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/tool_hands.mp4" poster="assets/tool_hands.jpg" controls muted loop playsinline width="100%"></video>
</div>

*Left: Kimodo text to motion, retargeted to a small MuJoCo humanoid. Right: MediaPipe hand keypoints on a test
video, in the image and in 3D.*

**Limitations.** Three primary models are blocked by access or registration (SAM 3D Objects, WiLoR with MANO, and
GVHMR with SMPL), so those tools run on their fallbacks. On a rendered tabletop, MoGe-2's metric scale is 3.8 times
too large.

## Live code-as-policy runs

`code.llm` writes one Python block per turn against `robot`, sees the printed output and images, and calls
`finish()` when it thinks the task is done. The task's score is never shown to it. Six `mujoco/manip-v0` tasks,
validation seeds 10000 to 10004; full details in [`RESULTS.md`](RESULTS.md).

```bash
uv run python harness/adapters/hylo-code-agents/examples/llm/run_code_llm.py run --out runs/code-llm \
    --model ollama:gpt-oss:20b --modes zero one --episodes 5
```

| Task | gpt-oss:20b zero-shot | gpt-oss:20b one-shot | qwen3.5:35b-a3b zero-shot | Claude Sonnet 5.5 zero-shot |
|---|---|---|---|---|
| pick_place | 5/5 | 5/5 | 1/1 | 1/1 |
| stack | 4/5 | 3/5 | 1/1 | 1/1 |
| put_in_bowl | 5/5 | 3/5 | 0/1 | 1/1 |
| push_to_zone | 0/5 | 0/5 | 0/1 | 1/1 |
| press_by_number | 0/5 | 0/5 | 0/1 | 1/1 |
| sort_by_colour | 0/5 | 1/5 | 0/1 | 0/1 |
| **Total** | **14/30** | **12/30** | **2/6** | **5/6** |

gpt-oss took 1 to 3 turns on the tasks it solved; 60 episodes in 4.1 h at $0. The six Claude episodes took 9 min and
cost $0.98 in total. The one-shot demonstration comes from a different task family.

### Skill reuse

The program promoted by the [campaign](#program-improvement) is stored as the verified skill `stack@1`. When an
episode's instruction matches it, `code.llm` is told the skill exists and can import it. gpt-oss:20b then called
`skills.stack(robot)` in its first turn in 10 of 10 fresh stack episodes, all successful, with 49 output tokens per
episode (zero-shot without the skill: 4/5). qwen3.5 called it in 2 of 3 episodes, both successful. Before the skill
was retrieved by the episode's own instruction and named in the first message, no model called it (0 of 12).

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<video src="assets/skill_reuse_stack.mp4" poster="assets/skill_reuse_stack.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/code_llm_put_in_bowl.mp4" poster="assets/code_llm_put_in_bowl.jpg" controls muted loop playsinline width="100%"></video>
</div>

*Left: gpt-oss:20b with the promoted skill, "stack the purple block on the green block": one turn,
`out = skills.stack(robot)`, 48 output tokens, success. Right: gpt-oss:20b zero-shot on put_in_bowl, written from
scratch in the episode (Blender Cycles).*

**Limitations.**

- No local model solved push_to_zone or press_by_number (0/30 for gpt-oss): it builds push and press goals with the
  hand pointing sideways and loops on "unreachable". Claude solved both.
- Models often call `finish()` without checking; on sort_by_colour, 0 to 2 of 3 or 4 blocks end in the right tray.
- From scratch, six 30-turn research sessions (gpt-oss and qwen3.5, on stack, press_by_number and sort_by_colour)
  froze no program. Research worked when it improved an existing program.
- qwen3.5 ran one episode per task only, because the shared CPU model server queued its turns for 20 to 30 min.

## Tool-calling agents

`tools.llm` (RFC-0007) gives a chat model the robot as typed tools and lets it call them natively, one checked
action at a time: it measures freely (`locate`, `grasp_options`, `pixel_to_world`, `calc`, ...), and every action
(`grasp`, `place`, `move_tcp`, ...) is a proposal that runs through the kernel and the safety shield and comes back
as a measured receipt with fresh images. Refused actions that moved nothing cost no decision; handles measured
before the robot moved are refused as stale; the task's own check is the verdict.

```bash
uv run hylo preflight --model ollama:qwen3.5:35b-a3b --vision          # text, image, tool call, tool result
uv run python harness/adapters/hylo-code-agents/examples/tools_llm/run_tools_llm.py run --out runs/tools-llm --episodes 3
uv run hylo watch runs/tools-llm/<run>                                  # follow or replay an episode
uv run hylo seal runs/tools-llm/<run>                                   # hash chains, inventory, read-only
```

`qwen3.5:35b-a3b` (vision, reasoning off, local, $0), zero-shot, cameras front / side / wrist with the
environment's segmentation, validation split, 3 episodes per task:

| Task | Success | Decisions (mean) | Model calls (mean) | Tokens in / out (mean) | Wall s (mean) |
|---|---|---|---|---|---|
| pick_place | **2/3** | 2.3 | 6.3 | 46,896 / 932 | 133 |
| put_in_bowl | **3/3** | 2.0 | 4.7 | 32,111 / 436 | 48 |
| stack | **3/3** | 3.3 | 8.7 | 73,884 / 1,138 | 122 |
| push_to_zone | 0/1 | 12 (8 more refused) | 29 | 377,227 / 7,999 | 962 |

push_to_zone was stopped after its first episode (it ran out of its 600 steps); press_by_number and sort_by_colour
were not run for this page.

<div style="display:grid;grid-template-columns:1fr;gap:8px">
<video src="assets/tools_llm_stack.mp4" poster="assets/tools_llm_stack.jpg" controls muted loop playsinline width="100%"></video>
</div>

*stack, seed 10000: locate both blocks, choose a grasp option, grasp, locate the target block again (the earlier
handle is stale after the motion), place on it; five model calls, three decisions. Re-simulated from the trace,
captions from the agent's turns and the condensed tool results.*

The evidence a run keeps: every trace line is hash-chained (`prev`, `h`), so `hylo seal` refuses a run whose
lines were changed, removed or reordered and `hylo seal --verify` lists any file changed after sealing (the stack
run above: 3 episodes, 531 to 2,124 chained lines each, 45 files sealed). Model prompts are recorded by layer digest
(protocol, evidence rules, boundary, embodiment card) with the tool surface's digest. `hylo chats` turns the
successful, unsteered conversations into training records.

**What the runs show.**

- The model asks for a grasp option, executes it, re-locates the place target after the motion and places in
  two decisions on pick_place and put_in_bowl; stack takes three.
- An agent's own `finish(success)` is not the verdict: in one pick_place episode the model reported success after
  `place`, and the task's check (object resting on the plate) said failure.
- Without a push verb, push_to_zone has to be improvised from straight `move_tcp` motions; an early version of the
  agent let the model repeat the same refused rotation for 60 calls, so identical refused actions now count as
  repeated and three in a row end the episode.

### Perturbed LIBERO

Seven perturbations of the official LIBERO scenes, declared on the task (`benchmark params: perturb`):

![The seven perturbations on libero_goal task 8](assets/libero_perturbations.jpg)

*libero_goal "put the bowl on the plate": original; position (the bowl trades places with the cream cheese box);
task (a rewritten goal: the cream cheese box on the plate, checked by LIBERO's own BDDL predicate); language (a
paraphrase); object (recoloured); camera (moved about its view axis); noise (pixel noise and a glare spot);
robot_init (the arm starts from offset joints).*

27 of the 40 tasks of libero_spatial, libero_object, libero_goal and libero_10 have a goal rewrite (single On / In
goals onto a named object); each instruction has three paraphrases written by `qwen3.5:35b-a3b` and kept only when
they contain every object and place word of the original. The perturbations are implemented and tested on the simulator (the picture above, and the goal rewrite checked
by LIBERO's own predicate); `tools.llm` has not been evaluated on them yet.

## Worlds

A stage (RFC-0008) is a scene written as entities with revisions: shapes and meshes, robots from 24 packs, cameras,
lights and people. A world is a MuJoCo simulation built from one revision; every action names the world generation
it was decided on and returns a receipt measured in the simulation. The same 21 tools serve a chat model
(`hylo world chat`), the Studio, coding agents (`hylo mcp`) and role campaigns (`hylo roles`).

```bash
uv run python bench/adapters/hylo-env-mujoco/examples/world/showcase.py --out runs/world-showcase   # the videos below
uv run hylo world chat "a table with a red cube, a Panda that picks it up, a Go1 walking past" --vision
uv run hylo mcp --surface mujoco.world --param stage=lab
```

| Scene | What runs | Measured (receipts and checks) |
|---|---|---|
| Quadrupeds | Go2, Go1, A1, Spot: an open-loop trot, then a turn, all in the same steps | 0.78, 1.27, 1.22, 1.31 m forward in 4 s, heading within 1.1 degrees; turns of 50, 79, 75, 37 degrees; all upright |
| Humanoids | G1 and H1 leg models under pretrained walking policies (numpy LSTM, weights read without executing them) | G1 2.23 m and H1 2.03 m in 5 s (0.45 and 0.41 m/s for 0.5 commanded); then turns of 122 and 120 degrees; both upright |
| Drones | Crazyflie 2 and Skydio X2 flying three waypoints each | X2 within 6 mm of every waypoint; Crazyflie within 0 to 48 mm (its moment actuators are weak, so its gains are scaled down) |
| Arms | a Panda picks a cube onto a plate; a UR5e with a mounted Robotiq gripper and a TidyBot base move at the same time | gripper closed on the cube (`holding`), check `near(cube, plate)` and `settled` satisfied |
| Crowd | four people on routes around two crates while a Go1 trots across | no person touched anything; 7.7 to 10.2 m walked each, around the crates and the dog; the Go1 2.41 m |
| Dataset | `export` while the Panda moves: two cameras, six frames each | 12 samples: RGB, depth in metres, segmentation by entity, camera pose and world state per frame |

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
<video src="assets/world_quadrupeds.mp4" poster="assets/world_quadrupeds.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/world_humanoids.mp4" poster="assets/world_humanoids.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/world_drones.mp4" poster="assets/world_drones.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/world_arms.mp4" poster="assets/world_arms.jpg" controls muted loop playsinline width="100%"></video>
<video src="assets/world_crowd.mp4" poster="assets/world_crowd.jpg" controls muted loop playsinline width="100%"></video>
<img src="assets/world_dataset_grid.jpg" alt="Exported frames: RGB, depth, segmentation" width="100%">
</div>

*Recorded through the world tools (`act` with `record`), MuJoCo rendering on the CPU. Last tile: three exported
frames of the top camera (RGB, depth, segmentation by entity).*

![The Studio's World page](assets/studio_world.png)

*The Studio's World page on a chat session's stage: entities, a free view to orbit, actions and checks by hand, and
the session's transcript.*

**A chat model building a world.** `qwen3.5:35b-a3b` was asked for a table with a red cube, a Panda next to it, a
Go1 1.5 m away, the Go1 to walk 1 m, the cube to be picked up and lifted 10 cm, and a check. Two sessions, 26 and 60 model calls:

- **First:** it built the scene in a few stage edits after two rejected patches, fixed an overlap the build
  reported, and walked the Go1 about 1.2 m. The cube, however, had no `body`, so it was fixed in place: the gripper
  closed on it but nothing could lift it. The model's check (cube 9 cm above the table) failed; it lowered the
  threshold to 5 cm until the check passed, and reported a 19 cm lift it never measured.
- **Second**, after `world_build` started listing the entities that are fixed in place: it made the cube movable,
  but then spent all its remaining calls moving the Panda between the floor and the table top (43 stage revisions)
  and never ran an action.

Neither session lifted the cube. Both are traced and can be replayed with `hylo watch runs/world-chat/demo1`.

**Role campaigns.** `hylo roles run` with a scripted worker (the test campaign): the dev task's acting calls become
a recipe skill keyed by the failure the task started with (`NOT_LIFTED`), a model-free replay on a fresh copy of the
stage verifies it, and the held-out task is offered the skill and solved. Every call is in a ledger with its
argument digest and receipts.

## What does not work yet

- **Scene builder:** 3 of 6 scenes solved, and only with privileged state. The other three failed on the model's
  side: the drawer was never opened, the cylinder was not turned upright, and the row was laid out in the wrong
  direction.
- **Replay to simulation:** only our own Panda episodes reproduce. None of LIBERO (3 episodes), DROID or the human
  video does.
- **Motion studio:** the motions fall over under physics. Five of six fall within 1.7 s without the ankle strategy,
  and the sixth, a 1 s hold, has its centre of mass outside the support.
- **Code as policy:** local models solve no push or press task, and from-scratch research sessions froze no program.
- **Humanoid:** the G1 cannot lift a box with both hands, and its camera-only programs score 0.50 on two_hands.
- **LIBERO:** the comparison with SmolVLA uses privileged segmentation; on libero_10 the microwave task fails 0/10
  and flat boxes slip out of the grasp.
- **Program improvement:** improving a working parent succeeds, but writing a program from scratch in a research
  session does not yet work with local models.
- **Tool-calling agents:** push and press tasks have no dedicated verbs and are improvised from `move_tcp`; the
  model's `finish(success)` is often wrong, which is why it never decides.
- **Worlds:** the quadruped trot is open-loop (turn rates differ by robot); the Crazyflie is slow because of its
  weak moment actuators; humanoids walk only under a pretrained policy on the leg models it was trained for (the
  29-joint G1 and the full H1 cannot stand by themselves); people are kinematic and stop rather than step aside.
- **Chat-built worlds:** the model writes scenes in small steps and fixes overlaps, but it also weakened its own
  check until it passed, and reported a lift it never measured; role campaigns check a fixed goal for this reason.

## Credits

- Robot models from MuJoCo Menagerie: Franka Emika Panda (Apache-2.0), ALOHA with two ViperX arms (BSD-3-Clause)
  and Unitree G1 with hands (BSD-3-Clause), fetched at a pinned commit and never vendored. The world packs add, from
  the same commit: UR5e, UR10e, KUKA iiwa 14, UFACTORY xArm 7, Unitree Go2 / Go1 / A1 / G1 / H1, Boston Dynamics
  Spot, ViperX 300s (BSD-3-Clause); SO-101, Shadow Hand, Booster T1, Skydio X2 (Apache-2.0); Robotiq 2F-85, Wonik
  Allegro (BSD-2-Clause); LEAP Hand, Crazyflie 2, Stanford TidyBot (MIT).
- The G1 and H1 leg models and their walking policies from `unitreerobotics/unitree_rl_gym` (BSD-3-Clause), fetched
  at a pinned commit and verified by digest; the policies are read as tensors and run with numpy.
- LIBERO benchmark scenes and tasks. SmolVLA numbers are recorded runs of that policy.
- Datasets used in the replay: `physical-intelligence/libero` (CC BY 4.0), `lerobot/droid_100` (MIT) and
  `rli14/10132025_human_demonstrations_hurricanemug_plaintable` (Apache-2.0), from Hugging Face.
- Motion-studio inputs: "Jumping jacks and burpees" by Taco Fleur (Cavemantraining), CC BY-SA 4.0, from Wikimedia
  Commons (the jumping-jacks outputs are adaptations under the same licence), and `pose.jpg` from Google's
  mediapipe-assets bucket. See [`motion/sources.json`](motion/sources.json).
- Models: SAM 3 and SAM 3D Objects (SAM License, Meta), MoGe-2 (MIT), GeoCalib (Apache-2.0 code, CC BY 4.0
  weights), CoACD (MIT), MediaPipe (Apache-2.0), NVIDIA Kimodo (NVIDIA Open Model License weights, Apache-2.0 code)
  with a Llama 3 text encoder (Llama 3 Community License), qwen3.5 and gpt-oss through Ollama.
- Rendering: Blender (GPL-3.0-or-later, run as a separate program, not linked) and MuJoCo (Apache-2.0).
