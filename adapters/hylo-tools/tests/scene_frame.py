"""Render a small tabletop MuJoCo scene (RGB, metric depth, intrinsics, camera pose) for the tool tests.

    python scene_frame.py OUT_DIR [--size 480]

Writes OUT_DIR/scene.png, OUT_DIR/scene.npz (depth (H, W) metres, K (3, 3), T_world_cam (4, 4) OpenCV optical frame,
seg (H, W) geom ids, names), OUT_DIR/scene.xml. Needs MuJoCo with an offscreen GL backend (MUJOCO_GL=osmesa or egl).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCENE = """
<mujoco model="tool_test_scene">
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><quality shadowsize="4096"/>
          <global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.78 0.70" rgb2="0.74 0.70 0.62" width="256" height="256"/>
    <material name="table" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.82 0.92" rgb2="0.35 0.40 0.50" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.6" dir="-0.2 0.4 -1" castshadow="true" diffuse="0.7 0.7 0.7"/>
    <geom name="table" type="box" size="0.6 0.45 0.02" pos="0 0 -0.02" material="table"/>
    <body name="red_box" pos="-0.12 0.05 0.04"><freejoint/>
      <geom name="red_box" type="box" size="0.04 0.03 0.04" rgba="0.85 0.15 0.12 1" euler="0 0 25"/></body>
    <body name="blue_can" pos="0.10 0.10 0.06"><freejoint/>
      <geom name="blue_can" type="cylinder" size="0.035 0.06" rgba="0.15 0.35 0.85 1"/></body>
    <body name="green_ball" pos="0.02 -0.12 0.035"><freejoint/>
      <geom name="green_ball" type="sphere" size="0.035" rgba="0.2 0.7 0.25 1"/></body>
    <body name="yellow_mug" pos="0.18 -0.08 0.0">
      <geom name="mug_body" type="cylinder" size="0.04 0.045" pos="0 0 0.045" rgba="0.95 0.8 0.15 1"/>
      <geom name="mug_handle" type="capsule" fromto="0.04 0 0.025 0.06 0 0.065" size="0.01" rgba="0.95 0.8 0.15 1"/></body>
    <camera name="front" pos="0 -0.55 0.45" xyaxes="1 0 0 0 0.62 0.78" fovy="55"/>
  </worldbody>
</mujoco>
"""


def render(out_dir: Path, size: int = 480) -> None:
    import mujoco
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.xml").write_text(SCENE)
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    H = W = size
    r = mujoco.Renderer(m, H, W)
    cam = m.camera("front").id
    r.update_scene(d, camera=cam)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    depth = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render()[..., 0].copy()
    r.close()
    fovy = np.radians(m.cam_fovy[cam])
    f = 0.5 * H / np.tan(fovy / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = d.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])   # MuJoCo camera (-z forward, +y up) -> OpenCV
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d.cam_xpos[cam]
    names = np.array([m.geom(i).name for i in range(m.ngeom)])
    from PIL import Image
    Image.fromarray(rgb).save(out_dir / "scene.png")
    np.savez_compressed(out_dir / "scene.npz", depth=depth.astype(np.float32), K=K, T_world_cam=T, seg=seg, names=names)


if __name__ == "__main__":
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 480
    render(Path(sys.argv[1]), size)
