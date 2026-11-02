---
name: hylo-tools
description: Run Hylo's perception and motion tools on the CPU with `hylo tool`. They cover text-prompted segmentation, metric depth, camera gravity, image-to-mesh, convex decomposition, hand and human motion from video, text-to-motion and offline rendering. Use when a task needs masks, depth, meshes, collision geometry, poses from video, a generated motion, or a good-looking video of a MuJoCo scene.
---

# Perception and motion tools

Each tool is a script with its own environment, created by `uv run --script` on first use (slow the first time,
cached afterwards). Tools run on the CPU. `hylo tool list` shows each tool's model and licence; `hylo tool NAME --help`
shows its arguments.

| Tool | In | Out |
|---|---|---|
| `segment` | image + text prompts | instance masks, boxes, scores |
| `depth` | image | metric depth, camera-frame points, normals, intrinsics |
| `gravity` | image | camera roll and pitch, focal length |
| `mesh` | image + mask | object mesh (.glb, .obj) |
| `convex` | mesh | convex collision parts and an MJCF snippet |
| `hands` | video | per-frame 3D hand keypoints |
| `human` | video | per-frame 3D body keypoints |
| `text_motion` | text | humanoid motion, with retargeting for the G1 |
| `render` | MuJoCo scene or trajectory | MP4 and a still (Blender Cycles, or MuJoCo off-screen) |

## Examples

```bash
uv run hylo tool segment frame.png -p "red block" -p "bowl" --out out/seg       # masks.npz + instances
uv run hylo tool depth frame.png --out out/depth                                 # depth.npz (depth, K, points)
uv run hylo tool mesh frame.png out/seg/masks.npz --instance 0 --depth out/depth/depth.npz --out out/mesh
uv run hylo tool convex out/mesh/mesh.obj --mass 0.2 --out out/convex
uv run hylo tool render scene.xml --trajectory traj.npz --backend blender --out out/video
```

Every tool also takes `--threads` (default `$HYLO_TOOLS_THREADS` or 32), `--workdir` and `--verbose`.

## Notes

- Gated weights (SAM 3 for `segment`) need a Hugging Face login with access granted (`huggingface-cli login`).
- `hands` and `human` fall back to keypoint models when the licensed body models (MANO, SMPL-X) are not installed;
  the output says which model ran.
- Blender rendering uses `$HYLO_BLENDER` (default `~/.local/opt/blender/blender`; extra shared libraries from
  `$HYLO_BLENDER_LIBS`); without Blender, `render` falls back to MuJoCo.
- Each run writes its files plus a `result.json` describing them: read that file rather than guessing names.
