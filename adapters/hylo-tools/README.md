# hylo-tools

Perception and motion tool adapters for Hylo (RFC-0006 part 8): segmentation, monocular depth, gravity, image to
mesh, convex decomposition, hand and human motion from video, text to motion, and offline rendering. Every tool runs
on the CPU of this workstation; a GPU would only make them faster.

Each model runs as a **worker** in its own uv environment, a subprocess. The workers are scripts under
`src/hylo/tools/workers/` with inline PEP 723 dependencies, so torch, transformers, mediapipe and coacd never enter
the Hylo virtualenv. The worker and Hylo exchange files: a request JSON, input PNG / .npy / .npz / meshes, and
`result.json` plus output .npz / .png / .glb / .obj / .mp4 / .xml. Every result carries timing and provenance:
model id, revision, licence, backend, device and package versions.

```
hylo tool list                                  # tools, models, licences
hylo tool segment frame.png -p basket -p "soup can" --out out/
hylo tool depth frame.png --out out/
hylo tool gravity frame.png
hylo tool mesh frame.png out/masks.npz --instance 0 --out out/          # auto: SAM 3D Objects, else depth fallback
hylo tool convex out/mesh.obj --name basket --mass 0.3 --out out/
hylo tool hands clip.mp4 --out out/          (hylo tool hands --status)
hylo tool human clip.mp4 --out out/          (hylo tool human --status)
hylo tool text_motion "a person walks forward and waves" --duration 5 --out out/
hylo tool render scene.xml --simulate 2 --out out/                       # Blender Cycles, else MuJoCo offscreen
```

In Python, each module has a function that returns a dataclass together with the raw `ToolResult`:
`segment.segment`, `depth.depth`, `gravity.gravity`, `mesh.mesh`, `convex.convex`, `hands.hands`, `human.human`,
`text_motion.text_motion` (plus `retarget` and the `Retargeter` protocol), and `render.render`.

- `segment.Sam3Segmenter` implements the hylo.manip segmenter interface: `segment(view, labels) -> list[Mask]`. Use
  it as `Tools(spec, executor, segmenter=Sam3Segmenter())`. It also accepts `__call__(rgb, labels) -> [(label, mask,
  score)]`, and it keeps SAM 3 loaded in a persistent worker.
- `runner.ToolServer(name)` keeps any worker and its model loaded across calls.

## Status (CPU, recorded runs)

Recorded by `HYLO_TOOLS_RUN=1 HYLO_TOOLS_THREADS=32 pytest harness/adapters/hylo-tools/tests` on this workstation
(torch 2.15 CPU build). CPU time is the wall time of the recorded example with 32 threads once the environment and
weights are cached. "Load" is model loading and "infer" is the model call. The first use of a tool also builds its uv
environment and downloads its weights, which takes from a minute to about 20 min for the 16 GB Llama text encoder.
The per-run details are in `examples/runs/<tool>.json`.

| Tool | Model run on the CPU | Licence | CPU time, 32 threads | Status |
|---|---|---|---|---|
| segment | SAM 3, `facebook/sam3` (transformers `Sam3Model`, fp32) | SAM License (Meta, custom; gated, access granted) | 3 prompts on a 387x387 frame: 6.5 s infer (image encoder once, ~2 s per prompt), 4.5 s load, 11.9 s wall | works |
| depth | MoGe-2, `Ruicheng/moge-2-vitl-normal` | MIT (code and weights) | 480x480: 4.3 s infer; 11.2 s wall with load | works |
| gravity | GeoCalib, pinhole weights | Apache-2.0 code; CC-BY-4.0 weights | 480x480: 0.5 s infer; 4.3 s wall with load | works (roll -0.8 deg and pitch -2.6 deg off the true camera) |
| mesh | SAM 3D Objects, primary (gated); fallback: MoGe-2 depth, frustum carving, marching cubes | SAM License (gated); fallback MIT / BSD-3 / Apache-2.0 | fallback: 11.3 s (depth 10.3 s + mesh 1.0 s) | works with fallback; SAM 3D blocked on Hugging Face access (403), see 1. below |
| convex | CoACD | MIT (MuJoCo Apache-2.0) | open cup: 2.5 s, 28 parts, MJCF dropped and settled | works |
| hands | WiLoR + MANO, primary; fallback MediaPipe Hand Landmarker 0.10.21 | WiLoR CC BY-NC-ND 4.0 + MANO (non-commercial); fallback Apache-2.0 | 32 frames at 320x320: 1.1 s wall (MediaPipe); WiLoR detector stage 13.7 s | works with fallback; WiLoR blocked on the MANO registration, see 2. below |
| human | GVHMR + SMPL/SMPL-X, primary; fallback MediaPipe Pose Landmarker (heavy) | GVHMR non-commercial + SMPL/SMPL-X licences; fallback Apache-2.0 | 24 frames at 384x256: 1.4 s wall (MediaPipe) | works with fallback; GVHMR blocked on the SMPL/SMPL-X registration and checkpoints (and a CPU port), see 3. below |
| text_motion | NVIDIA Kimodo, `nvidia/Kimodo-SOMA-RP-v1.2` and `Kimodo-G1-RP-v1`; text encoder LLM2Vec on Llama 3 8B, fp32 | NVIDIA Open Model License (weights), Apache-2.0 (code); Llama 3 Community License (gated, access granted) | 5 s clip, 150 frames, 100 steps: 28 s generation + 15 s load + 0.4 s retarget, 57 s wall | works |
| render | Blender 5.2.3 LTS Cycles (CPU, external program); fallback MuJoCo offscreen (OSMesa) | Blender GPL-3.0-or-later (run as a program, not linked); MuJoCo Apache-2.0 | 30 frames at 480x272, 16 spp + denoise: 0.36 s/frame, 11.4 s wall; MuJoCo 0.04 s/frame | works |

Examples (each at most 1 MB) are in `examples/`:

- segment: `segment_libero.png`
- depth: `depth_libero.png`, `depth_scene.png`
- gravity: `gravity_libero.png`, `gravity_scene.png`
- mesh: `mesh_libero_basket.png`
- convex: `convex_cup.png`
- hands: `hands_mediapipe.{png,mp4}`, `hands_wilor_detect.png`
- human: `human_mediapipe.{png,mp4}`
- text_motion: `text_motion_{skeleton,robot}.{png,mp4}`, `text_motion_g1.png`
- render: `render_{blender,mujoco}.{png,mp4}`

## What the user must do for the blocked primary paths

The fallbacks run now. Each primary path switches on automatically once its files or access are in place:
`hylo tool hands --status` and `hylo tool human --status` report what is still missing, and `mesh` checks access on
every `auto` call.

1. **SAM 3D Objects (mesh).** Request access at https://huggingface.co/facebook/sam-3d-objects as the logged-in user
   (`csbingao`) and wait for approval. On 2026-11-01 the request was still pending: the README is readable but
   `checkpoints/*` returns 403. The upstream pipeline needs CUDA builds of pytorch3d, kaolin and gsplat and a 32 GB
   GPU. After access is granted, porting its inference to the CPU is the remaining step in this package; until
   then `auto` reports the 403 and uses the depth fallback.
2. **MANO (hands, WiLoR).** Register at https://mano.is.tue.mpg.de and download `mano_v1_2.zip` (Downloads > Models
   & Code). Copy `mano_v1_2/models/MANO_RIGHT.pkl` to `$HYLO_MANO_DIR/MANO_RIGHT.pkl`; the default location is
   `~/.cache/hylo/tools/body_models/mano/MANO_RIGHT.pkl`. The WiLoR weights (`rolpotamias/WiLoR`, 2.6 GB) are
   already cached, and the WiLoR hand detector already runs. Do not use MANO files redistributed by third parties:
   WiLoR-mini would fetch one from its own HF repo, and the worker blocks that download.
3. **SMPL / SMPL-X and GVHMR checkpoints (human).** The default root is `~/.cache/hylo/tools/`; override it with
   `$HYLO_BODY_MODELS` and `$HYLO_GVHMR_DIR`.
   - From https://smpl-x.is.tue.mpg.de, download "SMPL-X v1.1 (NPZ+PKL)" and copy `models/smplx/SMPLX_NEUTRAL.npz`
     to `body_models/smplx/SMPLX_NEUTRAL.npz`.
   - From https://smpl.is.tue.mpg.de, download "SMPL for Python" (`SMPL_python_v.1.1.0.zip`) and copy
     `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `body_models/smpl/SMPL_NEUTRAL.pkl`.
   - From the GVHMR checkpoint folder (https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD),
     place these under `gvhmr/checkpoints/`: `gvhmr/gvhmr_siga24_release.ckpt`, `hmr2/epoch=10-step=25000.ckpt`,
     `vitpose/vitpose-h-multi-coco.pth` and `yolo/yolov8x.pt`.

   GVHMR's own code calls `.cuda()` throughout and needs a CUDA-only pytorch3d wheel, so a CPU port is still
   needed after the files are in place. `--backend gvhmr` currently only checks the files.
4. **Llama 3 (text_motion), on other machines only.** Accept the licence at
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and run `hf auth login`. Access is already granted on
   this workstation.

## Notes per tool

- **segment.** SAM 3 runs through transformers `Sam3Model` in fp32. The image encoder runs once per image, and
  each text prompt reuses it.
- **depth.** MoGe is pinned to the last MoGe-2 commit (`b942f00`), because MoGe-3 added the CUDA-only FlexGEMM
  extension.
  - On the rendered tabletop scene the shape is good (abs-rel 4% after scale alignment), but the metric scale is
    3.8x too large: 8 cm toy objects with no real-world size cues.
  - The `mesh` tool's `scale` corrects a known scale error.
- **gravity.** Uses GeoCalib's conventions: roll = atan2(g_x, g_y), and pitch = -asin(g_z), positive when the camera
  looks up, with g the gravity (down) in the OpenCV camera frame. `R_level_cam` levels the camera.
- **mesh (fallback).** The pipeline is MoGe-2 depth, then masked back-projection and outlier removal. The hidden
  back is the visible surface reflected across the plane of the silhouette points.
  - The solid is carved inside the mask's viewing frustum, between the two surfaces, then turned into a mesh with
    marching cubes and Taubin smoothing. The result is watertight and has vertex colours.
  - Two other methods are available: `--method hull` and `--method relief` (the visible height field only).
  - Open3D's Poisson was not used, because the open3d wheel needs libEGL, which this headless machine lacks.
  - Accuracy with true depth on the rendered can: extents within 1.5 cm and centre within 1 cm.
  - The completed volume comes out smaller than the true one (about 40 to 60%), because the reflection
    underestimates the hidden depth of round objects.
- **convex.** CoACD with hull decimation (at most 64 vertices per part).
  - The MJCF snippet is a body with a free joint, the visual mesh (group 2) and one collision geom per part
    (group 3). It can be included with `<include file="name.xml"/>`, with `meshes/` next to it.
  - Every run compiles the snippet with a floor and drops the object for 3 s.
- **hands / human (fallbacks).** MediaPipe 0.10.21 with headless OpenCV; the 1.x wheels need libEGL / libGLESv2.
  - Hands: 21 keypoints per hand in 2D, metric hand-centred 3D, and camera-frame 3D (approximate, fitted under the
    intrinsics).
  - Human: 33 joints per frame in image, hip-centred, camera and an approximate world frame. The world frame
    assumes a static camera with a level or given pitch and roll, and puts the ground at the feet.
  - Test videos are MediaPipe's Apache-2.0 test images animated with zoom, rotation and pan. They are fetched into
    the tools cache and are not stored in the repo.
- **text_motion.** NVIDIA Kimodo with code pinned at `e1c219c`; the worker uses a uv-managed Python because Kimodo
  builds a C++ extension.
  - SOMA (77 joints) gives joint positions directly, so no SMPL is needed. `model="g1"` gives Unitree G1 MuJoCo
    qpos directly.
  - `KeypointIKRetargeter` maps any skeleton to any MuJoCo humanoid; a SOMA-to-humanoid map and a small humanoid
    model are included. The text encoder (LLM2Vec on Llama 3 8B) needs about 32 GB of RAM in fp32.
- **render.** Blender 5.2.3 LTS runs as an external program: `$HYLO_BLENDER`, default
  `~/.local/opt/blender/blender`, with its libraries from `$HYLO_BLENDER_LIBS`.
  - Blender gets principled materials, area lights, a sky gradient, denoising and AgX colour.
  - The fallback is MuJoCo offscreen (OSMesa) with extra shadowed lights, a 4096 shadow map, MSAA, 2x
    supersampling and an orbit or tracking camera.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HYLO_TOOLS_THREADS` | 32 | CPU threads per worker |
| `HYLO_TOOLS_CACHE` | `~/.cache/hylo/tools` | work directories, small model files, test media |
| `HYLO_TOOLS_UV` | `uv` on PATH | the uv executable |
| `HYLO_MANO_DIR`, `HYLO_BODY_MODELS`, `HYLO_GVHMR_DIR` | under the tools cache | user-provided body models and checkpoints |
| `HYLO_BLENDER`, `HYLO_BLENDER_LIBS` | `~/.local/opt/blender/blender`, its user-space libs | Blender for `render` |

Model weights go to the default Hugging Face and torch caches.

## Tests

`pytest harness/adapters/hylo-tools/tests` runs the runner tests and the mesh geometry check with true depth; the
model tests are skipped. `HYLO_TOOLS_RUN=1` runs every tool on its example and refreshes `examples/`, keeping each
file at most 1 MB. The inputs are:

- the first frame of `docs/docs/assets/figures/libero-episode-strip.png`;
- a tabletop scene rendered with MuJoCo (`tests/scene_frame.py`), with true depth, intrinsics, camera pose and
  segmentation;
- synthesised gesture and person videos;
- a procedural cup mesh.
