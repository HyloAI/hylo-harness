# Dexterous hands

Multi-finger grasp planning, human-to-robot hand retargeting and replay-to-sim with a floating multi-finger hand.
**Status: code complete, tested with fast unit tests and smoke runs; none of the experiments below has been run
yet.** Every result cell says "not run yet" and lists the one command that produces it. `collect.py` copies the
outputs into [`results.json`](results.json) after the runs.

## Pieces

| Module | Package | Responsibility |
|---|---|---|
| `hylo.manip.handmodel` | hylo-manip | A hand cut out of any MJCF/URDF (the ee body's subtree under a mocap root): finger chains, tips, collision samples, Jacobians, the wrist point |
| `hylo.manip.grasp_quality` | hylo-manip | Friction cones, contact wrenches, force closure, the epsilon (largest-ball) measure, holding effort against gravity |
| `hylo.manip.dexgrasp` | hylo-manip | Objects as convex parts (exact ray entry, signed distance); sample hand poses, approach to contact, close each finger to contact, score, refine; ranked `DexGrasp`s and toolkit `Goal`s |
| `hylo.manip.retarget` | hylo-manip | 21 MediaPipe keypoints to finger joints and palm pose (fingertip and thumb-to-finger vectors, joint limits, smoothness) |
| `hylo.manip.handpath` | hylo-manip | Bend a retargeted hand motion onto a planned grasp (pre-pose, approach, hold, release) |
| `Tools.dex_grasps`, `Tools.retarget` | hylo-manip | Toolkit verbs for programs (`hand_grasps` is unchanged) |
| `hylo.envs.mujoco.hands` | hylo-env-mujoco | Menagerie hands (G1 left/right, Allegro left/right): HandSpecs, the floating-hand robot model |
| `mujoco.dex_table`, `mujoco/dexterous-v0` | hylo-env-mujoco | A floating hand on a six-joint base above a box, cylinder, sphere, mug and rock; lift tasks |
| `replay.py --hand`, `replay_sim.py` | hylo-code-agents | Replay-to-sim with a floating multi-finger hand driven by the retargeted motion, grasps refined by the planner |

Robot models come from MuJoCo Menagerie at commit b0655057a9df2abe39ea7fc989914ba6f248301c, fetched and verified
file by file (`assets.py`), never vendored: the Unitree G1 hands (`unitree_g1`, BSD-3-Clause) and the Wonik Robotics
Allegro Hand (`wonik_allegro`, BSD-2-Clause). Shadow Hand and LEAP Hand are also at that commit; they are not wired up.

## Grasp benchmark

`dex_lift.py` (camera-only: look, find, objects, `dex_grasps`, plan, run, lift) on `mujoco/dexterous-v0` through
`hylo run`, 10 validation episodes per hand and object.

| hand | box | cylinder | sphere | mug | rock |
|---|---|---|---|---|---|
| G1 right | not run yet | not run yet | not run yet | not run yet | not run yet |
| Allegro right | not run yet | not run yet | not run yet | not run yet | not run yet |

```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \
  uv run python harness/adapters/hylo-code-agents/examples/dexterous/grasp_benchmark.py --out runs/dex-grasps --episodes 10 --videos
```

Videos (`runs/dex-grasps/grasps_<hand>.mp4`, one lift per object, at most 4 MB): not run yet.

## Retargeting

The mug video's MediaPipe keypoints onto each hand: fingertip error per frame (robot scale, palm frame) and time
per frame.

| hand | tip error mean (mm) | median | p95 | ms per frame |
|---|---|---|---|---|
| G1 | not run yet | not run yet | not run yet | not run yet |
| Allegro | not run yet | not run yet | not run yet | not run yet |

```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \
  uv run python harness/adapters/hylo-code-agents/examples/dexterous/retarget_eval.py \
  hf:rli14/10132025_human_demonstrations_hurricanemug_plaintable:0 --profile video \
  --camera observation.image.ego_global --out runs/dex-retarget --hands g1 allegro --video
```

Side-by-side video (recording | human hand in its palm frame | each robot hand): not run yet.

## Hand replay

`hylo replay ... --hand allegro` (or `g1`; the side follows the detected hand) replaces the floating parallel
gripper with a floating multi-finger hand. Before: the same command without `--hand`.

| recording | hand | events matched | final object error | success reproduced |
|---|---|---|---|---|
| mug (`rli14/10132025_human_demonstrations_hurricanemug_plaintable`, episode 0, Apache-2.0) | gripper (earlier run, [`../replay/results.json`](../replay/results.json)) | 0/2 | 0.120 m | no |
| mug | gripper (rerun) | not run yet | not run yet | not run yet |
| mug | Allegro | not run yet | not run yet | not run yet |
| Hydro Flask bottle (`rli14/10132025_human_demonstrations_hydroflask_plaintable_reorient`, episode 0, Apache-2.0) | gripper | not run yet | not run yet | not run yet |
| Hydro Flask bottle | Allegro | not run yet | not run yet | not run yet |

```
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu
E=harness/adapters/hylo-code-agents/examples/dexterous
MUG=hf:rli14/10132025_human_demonstrations_hurricanemug_plaintable:0
FLASK=hf:rli14/10132025_human_demonstrations_hydroflask_plaintable_reorient:0
uv run hylo replay $MUG --profile video --camera observation.image.ego_global --labels "wooden spatula,white mug" --hand allegro --out runs/dex-replay-mug --video-path $E/human_mug_allegro.mp4
uv run hylo replay $MUG --profile video --camera observation.image.ego_global --labels "wooden spatula,white mug" --out runs/dex-replay-mug-gripper
uv run hylo replay $FLASK --profile video --camera observation.image.ego_global --labels "water bottle" --hand allegro --out runs/dex-replay-flask --video-path $E/human_flask_allegro.mp4
uv run hylo replay $FLASK --profile video --camera observation.image.ego_global --labels "water bottle" --out runs/dex-replay-flask-gripper
uv run python $E/collect.py
```

The labels skip the vision chat model; drop `--labels` to let it name the objects. The second recording's labels
are a guess until its first frame has been looked at.

## Tests

- `harness/packages/hylo-manip/tests/test_manip_dexgrasp.py`: grasp-quality math on analytic cases, object
  geometry, the hand model, planning on a box, trajectory bending, retargeting with known answers, the toolkit verbs
  (a synthetic three-finger hand; no downloads, no rendering).
- `bench/adapters/hylo-env-mujoco/tests/test_dex_table.py`: the hand library against the Menagerie models, shapes,
  tasks, and a physics lift of a box planned with `dex_grasps` and run through `plan` / `run`.
- `harness/adapters/hylo-code-agents/tests/test_replay.py`: the floating-hand scene, kinematic and physics replay,
  and the replay steps with stub tools and a multi-finger hand.
