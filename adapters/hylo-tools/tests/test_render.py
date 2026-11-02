"""render: a dropped-objects MuJoCo scene to MP4 on the CPU with Blender Cycles and with the MuJoCo offscreen
fallback (from a recorded trajectory, tracking camera); auto falls back when Blender is missing."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tools_testkit import copy_example, needs_run, record, save_example_png

from hylo.tools.render import LICENCE, render

DROP = """
<mujoco model="drop">
  <option timestep="0.002"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.78 0.70" rgb2="0.70 0.66 0.58" width="256" height="256"/>
    <material name="table" texture="grid" texrepeat="5 5" reflectance="0.05"/>
    <material name="metal" rgba="0.75 0.76 0.78 1" metallic="1" roughness="0.25"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.55 0.55 0.57 1"/>
    <geom name="table" type="box" size="0.5 0.4 0.02" pos="0 0 0.4" material="table"/>
    <geom type="cylinder" size="0.02 0.19" pos="0.45 0.35 0.19" material="metal"/>
    <geom type="cylinder" size="0.02 0.19" pos="-0.45 0.35 0.19" material="metal"/>
    <geom type="cylinder" size="0.02 0.19" pos="0.45 -0.35 0.19" material="metal"/>
    <geom type="cylinder" size="0.02 0.19" pos="-0.45 -0.35 0.19" material="metal"/>
    <body name="red_box" pos="-0.12 0.05 0.62" euler="20 30 10"><freejoint/>
      <geom type="box" size="0.04 0.03 0.04" rgba="0.85 0.15 0.12 1"/></body>
    <body name="blue_can" pos="0.10 0.10 0.75" euler="70 0 0"><freejoint/>
      <geom type="cylinder" size="0.035 0.06" rgba="0.15 0.35 0.85 1"/></body>
    <body name="green_ball" pos="0.02 -0.12 0.9"><freejoint/>
      <geom type="sphere" size="0.035" rgba="0.2 0.7 0.25 1"/></body>
    <body name="capsule" pos="0.2 -0.1 0.7" euler="0 80 30"><freejoint/>
      <geom type="capsule" size="0.02 0.05" rgba="0.95 0.8 0.15 1"/></body>
    <body name="egg" pos="-0.2 -0.15 0.8"><freejoint/>
      <geom type="ellipsoid" size="0.03 0.03 0.045" rgba="0.92 0.9 0.85 1"/></body>
    <camera name="side" pos="0 -1.2 0.9" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
</mujoco>
"""

_RUNS: dict = {}     # per-backend timings, recorded together


@pytest.fixture(scope="module")
def drop_scene(tmp_path_factory):
    p = tmp_path_factory.mktemp("render") / "drop.xml"
    p.write_text(DROP)
    return p


def _frame_differs(video) -> bool:
    import imageio.v3 as iio
    frames = iio.imread(video)
    return frames.shape[0] > 1 and np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean() > 2


@needs_run
def test_render_blender(drop_scene):
    t0 = time.perf_counter()
    r = render(drop_scene, simulate=1.5, backend="blender", width=480, height=272, fps=20, samples=16,
               orbit_degrees=60, elevation=-22)
    wall = time.perf_counter() - t0
    assert r.backend == "blender" and r.frames == 30 and r.video and r.video.exists()
    assert _frame_differs(r.video)
    from PIL import Image
    still = save_example_png(Image.open(r.still), "render_blender")
    mp4 = copy_example(r.video, "render_blender.mp4")
    o = r.result.outputs
    _RUNS["blender"] = {"wall_s": round(wall, 2), "seconds_per_frame": o["seconds_per_frame"],
                                     "frames": r.frames, "size": o["size"], "samples": 16,
                                     "revision": r.result.provenance.get("revision"), "examples": [still, mp4]}
    _record()


@needs_run
def test_render_mujoco_trajectory(drop_scene, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(drop_scene))
    d = mujoco.MjData(m)
    qpos, t = [], []
    while d.time < 1.5:
        qpos.append(d.qpos.copy())
        t.append(d.time)
        for _ in range(5):
            mujoco.mj_step(m, d)
    traj = tmp_path / "traj.npz"
    np.savez(traj, qpos=np.array(qpos), time=np.array(t))
    t0 = time.perf_counter()
    r = render(drop_scene, traj, backend="mujoco", width=480, height=272, fps=20, camera="track",
               elevation=-20, azimuth=110)
    wall = time.perf_counter() - t0
    assert r.backend == "mujoco" and r.frames == 30 and _frame_differs(r.video)
    from PIL import Image
    still = save_example_png(Image.open(r.still), "render_mujoco")
    mp4 = copy_example(r.video, "render_mujoco.mp4")
    o = r.result.outputs
    _RUNS["mujoco"] = {"wall_s": round(wall, 2), "seconds_per_frame": o["seconds_per_frame"],
                                    "frames": r.frames, "size": o["size"], "examples": [still, mp4]}
    _record()


@needs_run
def test_auto_falls_back_without_blender(drop_scene):
    r = render(drop_scene, still_only=True, width=320, height=192, env={"HYLO_BLENDER": "/nonexistent/blender",
                                                                         "PATH": "/usr/bin:/bin"})
    assert r.backend == "mujoco" and r.video is None and r.still.exists()
    assert r.fallback_reason and "Blender not found" in r.fallback_reason


def _record() -> None:
    runs = _RUNS
    b, m = runs.get("blender"), runs.get("mujoco")
    main = b or m
    notes = []
    if b:
        notes.append(f"Blender Cycles CPU ({b['revision']}): {b['frames']} frames {b['size'][0]}x{b['size'][1]}, "
                     f"{b['samples']} spp + OIDN, {b['seconds_per_frame']} s/frame")
    if m:
        notes.append(f"MuJoCo offscreen fallback (OSMesa, 2x supersampling): {m['frames']} frames, "
                     f"{m['seconds_per_frame']} s/frame, wall {m['wall_s']} s")
    record("render", status="works", model="Blender 5.2 Cycles (CPU); MuJoCo offscreen fallback", licence=LICENCE,
           wall_s=main["wall_s"], timing={k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in runs.items()},
           examples=[e for v in runs.values() for e in v["examples"]], notes="; ".join(notes))
