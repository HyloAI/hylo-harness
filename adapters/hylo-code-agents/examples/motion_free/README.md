# Motion characters and free mode

The motion studio's final motion can be shown on many target characters, and in free mode the model composes its
own pipeline. **Status: code complete and tested; the showcase requests below have not been run yet**, so this page
has no result numbers or videos. [`results.json`](results.json) lists every request as "not run yet" with its
command; after a run, `showcase.py collect` fills it from the run directories.

## Characters

Every character is a short mapping file in
[`motion_characters/`](../../src/hylo/codeagents/motion_characters/): which body (robots) or bone end (rigged
characters) is the pelvis, chest, head and, per side, hip, knee, ankle, toe, shoulder, elbow and wrist. Assets are
fetched from their original source on first use and verified by digest; nothing is vendored.

| Name | Character | Kind | Source | Licence |
|---|---|---|---|---|
| `g1` | Unitree G1 | robot (MJCF) | MuJoCo Menagerie `unitree_g1` | BSD-3-Clause |
| `h1` | Unitree H1 | robot (MJCF) | MuJoCo Menagerie `unitree_h1` | BSD-3-Clause |
| `t1` | Booster T1 | robot (MJCF) | MuJoCo Menagerie `booster_t1` | Apache-2.0 |
| `apollo` | Apptronik Apollo | robot (MJCF) | MuJoCo Menagerie `apptronik_apollo` | Apache-2.0 |
| `n1` | Fourier N1 | robot (MJCF) | MuJoCo Menagerie `fourier_n1` | Apache-2.0 |
| `human` | Hylo capsule figure | simulated humanoid (MJCF) | hylo-tools | Apache-2.0 |
| `kenney_survivor` | Kenney, Animated Characters 1 (skin survivorMaleB) | rigged character (FBX) | OpenGameArt upload by Kenney | CC0 1.0 |
| `quaternius_casual` | Quaternius, Ultimate Animated Character Pack (Casual_Female) | rigged character (FBX) | OpenGameArt upload by Quaternius | CC0 1.0 |

The Menagerie models come from commit `b0655057a9df2abe39ea7fc989914ba6f248301c`, with a sha256 per file
(`menagerie_*.json`). The Berkeley Humanoid at that commit has no arms, so it is not a target. Every input, asset
and model is listed with its licence in [`sources.json`](sources.json).

## Retargeting

- **Proportions.** The source keypoints are given each character's proportions: the root path is scaled by the
  leg-length ratio; every limb segment keeps the source's direction with the character's own length; hips and
  shoulders sit as on the character, turned with the source's pelvis and chest.
- **Robots.** Damped least-squares IK in MuJoCo fits the targets within the joint limits. It keeps planted feet flat
  in their rest orientation and holds hands clear of the body. A height correction then keeps the lowest foot point
  on the floor while the source's feet are down.
- **Rigged characters.** Blender drives the armature. The pelvis bone takes the hips' position and turn, spine bones
  share the turn up to the chest, and each limb bone is swung so its segment points like the source's. The same
  floor correction follows. Characters render with the hylo-tools render tool's look.
- **Reported per character.** Keypoint error (metres and percent of height), limb angle error, and joint-limit hits.
  A robot hit is a frame where the solver still pushed past a limit. A rigged character has no joint limits, so a
  hit there is a knee bent forwards. The report adds floor penetration, foot sliding and, for robots, the physics
  rollout.

The motions are kinematic: a physics rollout with position servos and no balance controller is reported for
robots, but the videos play the kinematic motion.

## Free mode

```bash
uv run hylo motion --free "REQUEST" --out DIR [--video V] [--image I] [--budget turns=12] [--max-tool-calls 40]
```

Each turn the model writes one script. It is checked (allowed imports, no eval or dunder access) and saved as
`scripts/turnNN.py`. It then runs in a separate Python process in the run directory, with resource limits and an
audit hook that refuses files outside the directory, processes, sockets and ctypes. Scripts import the studio kit
(`from studio import *`): text_motion, video_motion, pose_from_image, hands, depth, segment, human, retarget, check,
render, lineup, look and finish. They may write their own processing with numpy and scipy, and their own modules
in `lib/`. Every tool call is a request to the studio, which checks it (an allowed tool, paths inside the run
directory, the tool-call budget). Each script and request is logged in `log.jsonl`.

## Showcase requests

One command per request, run one at a time (the Ollama server is shared). It needs OSMesa for MuJoCo and Blender
for the Cycles renders:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py run text_all
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py run video_jacks
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py run image_warrior
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py run free_loop
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py run free_keyframe
uv run python harness/adapters/hylo-code-agents/examples/motion_free/showcase.py lineup text_all
```

`run all` runs the five in turn and `collect` rewrites `results.json`.

| Request | Mode | Input | Characters | Status |
|---|---|---|---|---|
| `text_all`: "wave with the right hand, then squat down and stand up again" | guided | text | all eight | not run yet |
| `video_jacks`: copy the jumping jacks (10.6 to 13.4 s), next to the source | guided | jumping-jacks clip (CC BY-SA 4.0) | g1, h1, kenney_survivor | not run yet |
| `image_warrior`: move into the photo's pose, hold it, return | guided | Warrior II photo (CC BY 2.0) | g1, apollo, quaternius_casual | not run yet |
| `free_loop`: loop and mirror the clip into a 12 s routine | free | jumping-jacks clip | chosen by the model | not run yet |
| `free_keyframe`: walk four steps and end in the photo's pose | free | Warrior II photo | chosen by the model | not run yet |
| All characters: `text_all`'s motion on every character | lineup | | all eight | not run yet |

`showcase.py smoke --out DIR` runs all five requests and the lineup with a scripted model. It uses synthetic motion
sources instead of Kimodo and the pose model, MuJoCo renders, and keypoint stand-ins for the rigged characters (no
Blender). It checks the plumbing (run directories, reports, video grids and the source next to the characters),
not the motions.

The jumping-jacks clip is CC BY-SA 4.0 (Taco Fleur, Cavemantraining), so videos derived from it are shared under
the same licence. The Warrior II photo is CC BY 2.0 (lululemon athletica).
