---
name: hylo-replay
description: Turn a recorded episode (LIBERO, a DROID-style dataset, or a plain video of a hand) into a MuJoCo scene and replay it kinematically and with physics using `hylo replay`, producing checks and a side-by-side video. Use when asked to bring a demonstration or a video into simulation, check whether a recorded motion is physically consistent, or make a sim twin of a clip.
---

# Replay to simulation

The pipeline: pick the frames, name the objects (from labels, the instruction, or a vision model), segment and
track them with depth, estimate the hand or gripper path, build a MuJoCo scene, replay the path kinematically,
then with physics, and check that the objects end where the recording says they did.

## Commands

```bash
# a LIBERO episode
MUJOCO_GL=osmesa uv run hylo replay EPISODE --profile libero --out runs/replay/libero-0
# a DROID-style dataset on the Hugging Face hub, choosing the camera
MUJOCO_GL=osmesa uv run hylo replay hf:DATASET/EPISODE --profile droid --camera exterior_image_1_left --out runs/replay/droid-0
# a phone video of a hand moving objects, labels given so no vision model is needed
MUJOCO_GL=osmesa uv run hylo replay clip.mp4 --profile video --labels "mug,plate" --out runs/replay/clip
```

Useful flags: `--step K` keeps every k-th frame, `--max-frames N` caps the frames, `--agent` lets a chat model
drive the steps, `--render-backend blender` gives a better-looking video, and `--no-video` skips the video.

## Output

`--out` holds `scene.xml` (with `meshes/`), the tracks (`tracks.npz`), `report.json` (the checks: per-object
position error, contact consistency, whether the physics replay follows the kinematic one), segmentation overlays
under `overlays/`, and `replay.mp4`, the side-by-side video (recording | sim).

## Rules

- Report the checks in `report.json` with the video; a good-looking video with failed checks is not a successful replay.
- Monocular depth sets the scale of a plain video, so expect centimetre-level errors.
