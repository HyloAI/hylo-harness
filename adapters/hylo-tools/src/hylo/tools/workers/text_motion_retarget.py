"""Retargeting human skeleton motion to a MuJoCo humanoid by keypoint inverse kinematics.

Pure numpy + mujoco and no Hylo imports, so both the `text_motion` worker (in its own environment) and Hylo code
(`hylo.tools.text_motion`) use it.

A `Retargeter` maps a skeleton motion (joint positions (T, J, 3), z up, metres, joint names) to robot joint targets
(qpos (T, nq) of a MuJoCo model). `KeypointIKRetargeter` is a generic one: for each frame, human keypoints (pelvis,
hips, knees, ankles, toes, shoulders, elbows, wrists, chest, head) are scaled to the robot's size about the origin
(the floor stays at z = 0); the floating base takes the human pelvis frame (hips and spine), and the robot's
matching sites are pulled onto the keypoints by damped least-squares IK over the joints, warm-started from the
previous frame and respecting joint limits. Give it any MJCF and a
map {human joint name: robot site or body name}; `HUMANOID_XML` is a small built-in humanoid (written for this
tool) with the sites of `SOMA_TO_HUMANOID`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class RobotMotion:
    """Robot joint targets: qpos (T, nq) for the MJCF in `xml`, at fps; joint_names (one per hinge/slide joint, in
    qpos order after the floating base); err (T,) mean keypoint error in metres after IK."""
    qpos: np.ndarray
    fps: float
    xml: str
    joint_names: list[str]
    err: np.ndarray = field(default_factory=lambda: np.zeros(0))


class Retargeter(Protocol):
    def __call__(self, joints: np.ndarray, names: Sequence[str], fps: float) -> RobotMotion:
        """joints (T, J, 3) world positions, z up, metres; names: J joint names."""
        ...


# A simple humanoid (about 1.5 m): floating pelvis, 3-dof waist, 3-dof hips and shoulders, knees, 2-dof ankles,
# elbows, 2-dof neck. Sites mark the keypoints the retargeter matches.
HUMANOID_XML = """
<mujoco model="hylo_humanoid">
  <compiler angle="degree"/>
  <option timestep="0.005"/>
  <visual><headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/><quality shadowsize="4096"/>
          <global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint damping="1" armature="0.01" limited="true"/>
    <geom type="capsule" rgba="0.75 0.78 0.82 1" contype="0" conaffinity="0"/>
    <site size="0.015" rgba="1 0.3 0.2 0.0"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.84" rgb2="0.76 0.76 0.74" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="20 20"/>
    <material name="shell" rgba="0.82 0.84 0.88 1" specular="0.6" shininess="0.6"/>
    <material name="joint" rgba="0.18 0.20 0.24 1"/>
    <material name="accent" rgba="0.95 0.55 0.10 1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.86 0.94" rgb2="0.40 0.46 0.56" width="256" height="256"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 4" dir="-0.35 0.45 -1" castshadow="false" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="floor" contype="1" conaffinity="1"/>
    <body name="pelvis" pos="0 0 0.92">
      <freejoint name="root"/>
      <site name="pelvis"/>
      <light name="key" mode="trackcom" pos="1.2 -1.6 3.2" dir="-0.35 0.5 -1" cutoff="40" exponent="2"
             castshadow="true" diffuse="0.7 0.7 0.7"/>
      <geom name="pelvis" type="box" size="0.07 0.12 0.06" material="shell"/>
      <body name="torso" pos="0 0 0.06">
        <joint name="waist_z" type="hinge" axis="0 0 1" range="-60 60"/>
        <joint name="waist_y" type="hinge" axis="0 1 0" range="-30 70"/>
        <joint name="waist_x" type="hinge" axis="1 0 0" range="-35 35"/>
        <geom name="abdomen" fromto="0 0 0.02 0 0 0.16" size="0.075" material="joint"/>
        <geom name="chest" type="box" pos="0 0 0.30" size="0.09 0.15 0.13" material="shell"/>
        <geom name="chest_badge" type="box" pos="0.092 0 0.32" size="0.004 0.05 0.03" material="accent"/>
        <site name="chest" pos="0 0 0.38"/>
        <body name="head" pos="0 0 0.45">
          <joint name="neck_z" type="hinge" axis="0 0 1" range="-70 70"/>
          <joint name="neck_y" type="hinge" axis="0 1 0" range="-40 50"/>
          <geom name="neck" fromto="0 0 0 0 0 0.06" size="0.035" material="joint"/>
          <geom name="head" type="sphere" pos="0 0 0.14" size="0.095" material="shell"/>
          <geom name="visor" type="box" pos="0.07 0 0.15" size="0.03 0.07 0.03" rgba="0.1 0.12 0.16 1"/>
          <site name="head" pos="0 0 0.14"/>
        </body>
        <body name="l_upper_arm" pos="0 0.20 0.38">
          <joint name="l_shoulder_y" type="hinge" axis="0 1 0" range="-180 60"/>
          <joint name="l_shoulder_x" type="hinge" axis="1 0 0" range="-20 180"/>
          <joint name="l_shoulder_z" type="hinge" axis="0 0 1" range="-90 90"/>
          <site name="l_shoulder"/>
          <geom name="l_shoulder" type="sphere" size="0.055" material="joint"/>
          <geom name="l_upper_arm" fromto="0 0 0 0 0 -0.26" size="0.045" material="shell"/>
          <body name="l_forearm" pos="0 0 -0.27">
            <joint name="l_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
            <site name="l_elbow"/>
            <geom name="l_elbow" type="sphere" size="0.045" material="joint"/>
            <geom name="l_forearm" fromto="0 0 0 0 0 -0.24" size="0.038" material="shell"/>
            <geom name="l_hand" type="sphere" pos="0 0 -0.27" size="0.045" material="accent"/>
            <site name="l_wrist" pos="0 0 -0.25"/>
          </body>
        </body>
        <body name="r_upper_arm" pos="0 -0.20 0.38">
          <joint name="r_shoulder_y" type="hinge" axis="0 1 0" range="-180 60"/>
          <joint name="r_shoulder_x" type="hinge" axis="1 0 0" range="-180 20"/>
          <joint name="r_shoulder_z" type="hinge" axis="0 0 1" range="-90 90"/>
          <site name="r_shoulder"/>
          <geom name="r_shoulder" type="sphere" size="0.055" material="joint"/>
          <geom name="r_upper_arm" fromto="0 0 0 0 0 -0.26" size="0.045" material="shell"/>
          <body name="r_forearm" pos="0 0 -0.27">
            <joint name="r_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
            <site name="r_elbow"/>
            <geom name="r_elbow" type="sphere" size="0.045" material="joint"/>
            <geom name="r_forearm" fromto="0 0 0 0 0 -0.24" size="0.038" material="shell"/>
            <geom name="r_hand" type="sphere" pos="0 0 -0.27" size="0.045" material="accent"/>
            <site name="r_wrist" pos="0 0 -0.25"/>
          </body>
        </body>
      </body>
      <body name="l_thigh" pos="0 0.09 -0.06">
        <joint name="l_hip_y" type="hinge" axis="0 1 0" range="-120 30"/>
        <joint name="l_hip_x" type="hinge" axis="1 0 0" range="-30 60"/>
        <joint name="l_hip_z" type="hinge" axis="0 0 1" range="-60 60"/>
        <site name="l_hip"/>
        <geom name="l_hip" type="sphere" size="0.06" material="joint"/>
        <geom name="l_thigh" fromto="0 0 0 0 0 -0.38" size="0.06" material="shell"/>
        <body name="l_shin" pos="0 0 -0.40">
          <joint name="l_knee" type="hinge" axis="0 1 0" range="0 150"/>
          <site name="l_knee"/>
          <geom name="l_knee" type="sphere" size="0.055" material="joint"/>
          <geom name="l_shin" fromto="0 0 0 0 0 -0.36" size="0.05" material="shell"/>
          <body name="l_foot" pos="0 0 -0.40">
            <joint name="l_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
            <joint name="l_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
            <site name="l_ankle"/>
            <geom name="l_foot" type="box" pos="0.05 0 -0.03" size="0.11 0.045 0.025" material="joint"/>
            <site name="l_toe" pos="0.15 0 -0.04"/>
          </body>
        </body>
      </body>
      <body name="r_thigh" pos="0 -0.09 -0.06">
        <joint name="r_hip_y" type="hinge" axis="0 1 0" range="-120 30"/>
        <joint name="r_hip_x" type="hinge" axis="1 0 0" range="-60 30"/>
        <joint name="r_hip_z" type="hinge" axis="0 0 1" range="-60 60"/>
        <site name="r_hip"/>
        <geom name="r_hip" type="sphere" size="0.06" material="joint"/>
        <geom name="r_thigh" fromto="0 0 0 0 0 -0.38" size="0.06" material="shell"/>
        <body name="r_shin" pos="0 0 -0.40">
          <joint name="r_knee" type="hinge" axis="0 1 0" range="0 150"/>
          <site name="r_knee"/>
          <geom name="r_knee" type="sphere" size="0.055" material="joint"/>
          <geom name="r_shin" fromto="0 0 0 0 0 -0.36" size="0.05" material="shell"/>
          <body name="r_foot" pos="0 0 -0.40">
            <joint name="r_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
            <joint name="r_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
            <site name="r_ankle"/>
            <geom name="r_foot" type="box" pos="0.05 0 -0.03" size="0.11 0.045 0.025" material="joint"/>
            <site name="r_toe" pos="0.15 0 -0.04"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# SOMA skeleton joint (Kimodo somaskel77) -> site of HUMANOID_XML, with a weight
SOMA_TO_HUMANOID: dict[str, tuple[str, float]] = {
    "Hips": ("pelvis", 3.0), "Chest": ("chest", 1.5), "Head": ("head", 1.0),
    "LeftLeg": ("l_hip", 1.0), "RightLeg": ("r_hip", 1.0),
    "LeftShin": ("l_knee", 1.5), "RightShin": ("r_knee", 1.5),
    "LeftFoot": ("l_ankle", 2.0), "RightFoot": ("r_ankle", 2.0),
    "LeftToeBase": ("l_toe", 1.0), "RightToeBase": ("r_toe", 1.0),
    "LeftArm": ("l_shoulder", 1.0), "RightArm": ("r_shoulder", 1.0),
    "LeftForeArm": ("l_elbow", 1.5), "RightForeArm": ("r_elbow", 1.5),
    "LeftHand": ("l_wrist", 2.0), "RightHand": ("r_wrist", 2.0),
}


def _segment_lengths(points: dict[str, np.ndarray], pairs: Sequence[tuple[str, str]]) -> float:
    return float(sum(np.linalg.norm(points[a] - points[b]) for a, b in pairs if a in points and b in points))


class KeypointIKRetargeter:
    """Damped least-squares keypoint IK on a MuJoCo model (see the module docstring). `scale`: human -> robot length
    ratio (None: from the leg and torso lengths of the first frame against the robot's rest pose)."""

    def __init__(self, xml: str = HUMANOID_XML, mapping: dict[str, tuple[str, float]] | None = None,
                 scale: float | None = None, iters: int = 30, damping: float = 1e-2, root: str = "Hips",
                 pelvis_frame: tuple[str, str, str] | None = ("LeftLeg", "RightLeg", "Chest")) -> None:
        import mujoco
        self.mujoco = mujoco
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml) if xml.lstrip().startswith("<") else \
            mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.mapping = dict(mapping or SOMA_TO_HUMANOID)
        self.scale, self.iters, self.damping, self.root = scale, iters, damping, root
        self.pelvis_frame = pelvis_frame
        m = self.model
        self.targets = []
        for human, (site, w) in self.mapping.items():
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
            kind = "site"
            if sid < 0:
                sid, kind = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, site), "body"
            if sid < 0:
                raise ValueError(f"retarget: no site or body {site!r} in the robot model")
            self.targets.append((human, kind, sid, float(w)))
        self.joint_names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]
        self.free = [j for j in range(m.njnt) if m.jnt_type[j] == 0]
        free_dofs = {int(m.jnt_dofadr[j]) + k for j in self.free for k in range(6)}
        self.ik_dofs = np.array([i for i in range(m.nv) if not (self.pelvis_frame and i in free_dofs)], int)
        self.root_target = next((k for k, (h, *_) in enumerate(self.targets) if h == root), None)

    def _root_pose(self, q: np.ndarray, pts: dict[str, np.ndarray], target: np.ndarray) -> None:
        """Floating base from the human pelvis frame: lateral axis left hip -> right hip reversed (+y left), up
        from the pelvis towards `pelvis_frame[2]` (orthogonalised), forward = y x z; then translate so the root
        keypoint lands on its target."""
        mj, m, d = self.mujoco, self.model, self.data
        lh, rh, up_ref = self.pelvis_frame
        y = pts[lh] - pts[rh]
        y /= np.linalg.norm(y) + 1e-9
        z = pts[up_ref] - pts[self.root]
        z = z - y * (z @ y)
        z /= np.linalg.norm(z) + 1e-9
        x = np.cross(y, z)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, np.stack([x, y, z], 1).ravel())
        a = m.jnt_qposadr[self.free[0]]
        q[a + 3:a + 7] = quat
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        _, kind, i, _ = self.targets[self.root_target]
        p = d.site_xpos[i] if kind == "site" else d.xpos[i]
        q[a:a + 3] += target - p

    def _robot_points(self) -> dict[str, np.ndarray]:
        d = self.data
        return {h: (d.site_xpos[i] if k == "site" else d.xpos[i]).copy() for h, k, i, _ in self.targets}

    def _auto_scale(self, first: dict[str, np.ndarray]) -> float:
        mj = self.mujoco
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        rob = self._robot_points()
        pairs = [(a, b) for a, b in (("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("RightLeg", "RightShin"),
                                     ("RightShin", "RightFoot"), ("Hips", "Head"))]
        h, r = _segment_lengths(first, pairs), _segment_lengths(rob, pairs)
        return r / h if h > 1e-6 and r > 1e-6 else 1.0

    def __call__(self, joints: np.ndarray, names: Sequence[str], fps: float) -> RobotMotion:
        mj, m, d = self.mujoco, self.model, self.data
        idx = {n: i for i, n in enumerate(names)}
        missing = [h for h, *_ in self.targets if h not in idx]
        if missing:
            raise ValueError(f"retarget: the motion has no joints {missing}")
        J = np.asarray(joints, float)
        T = len(J)
        first = {h: J[0, idx[h]] for h, *_ in self.targets}
        scale = self.scale or self._auto_scale(first)
        mj.mj_resetData(m, d)
        mj.mj_forward(m, d)
        qpos = np.zeros((T, m.nq))
        err = np.zeros(T)
        q = d.qpos.copy()
        lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
        w = np.concatenate([[wt] * 3 for *_, wt in self.targets])
        jac = np.zeros((3, m.nv))
        for t in range(T):
            # scaling about the origin keeps the floor (z = 0) where it is
            tgt = np.stack([J[t, idx[h]] * scale for h, *_ in self.targets])
            if self.pelvis_frame and self.free and self.root_target is not None:
                self._root_pose(q, {h: J[t, idx[h]] for h in (*self.pelvis_frame, self.root)},
                                tgt[self.root_target])
            elif t == 0:   # start upright above the first root position
                for j in self.free:
                    a = m.jnt_qposadr[j]
                    q[a:a + 3] = tgt[0]
            iters = self.iters * (4 if t == 0 else 1)
            for _ in range(iters):
                d.qpos[:] = q
                mj.mj_kinematics(m, d)
                mj.mj_comPos(m, d)
                rows, res = [], []
                for k, (h, kind, i, wt) in enumerate(self.targets):
                    if kind == "site":
                        mj.mj_jacSite(m, d, jac, None, i)
                        p = d.site_xpos[i]
                    else:
                        mj.mj_jacBody(m, d, jac, None, i)
                        p = d.xpos[i]
                    rows.append(jac.copy())
                    res.append(tgt[k] - p)
                A = (np.concatenate(rows) * w[:, None])[:, self.ik_dofs]
                b = np.concatenate(res) * w
                dq = np.zeros(m.nv)
                dq[self.ik_dofs] = np.linalg.solve(A.T @ A + self.damping * np.eye(len(self.ik_dofs)), A.T @ b)
                mj.mj_integratePos(m, q, dq, 1.0)
                for j in range(m.njnt):
                    if m.jnt_limited[j] and m.jnt_type[j] in (2, 3):
                        a = m.jnt_qposadr[j]
                        q[a] = np.clip(q[a], lo[j], hi[j])
                if np.linalg.norm(dq) < 1e-5:
                    break
            d.qpos[:] = q
            mj.mj_kinematics(m, d)
            rp = self._robot_points()
            err[t] = float(np.mean([np.linalg.norm(rp[h] - tgt[k]) for k, (h, *_) in enumerate(self.targets)]))
            qpos[t] = q
        return RobotMotion(qpos=qpos, fps=float(fps), xml=self.xml, joint_names=self.joint_names, err=err)


def render_robot(xml: str, qpos: np.ndarray, out_mp4: str, fps: float, size: tuple[int, int] = (480, 360),
                 camera: str | None = None, stride: int = 1, still: str | None = None, distance: float = 3.0,
                 azimuth: float = 120.0, elevation: float = -12.0) -> None:
    """Render a qpos trajectory with MuJoCo offscreen (needs MUJOCO_GL=osmesa or egl) to an MP4 (and a PNG strip
    of 5 frames when `still` is given). Without a named `camera`, a camera follows the robot's root body."""
    import imageio.v2 as imageio
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml) if xml.lstrip().startswith("<") else mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    W, H = size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    r = mujoco.Renderer(m, H, W)
    named = camera is not None and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    root = 1 if m.nbody > 1 else 0
    frames = []
    for t in range(0, len(qpos), stride):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [d.xpos[root][0], d.xpos[root][1], 0.75 * max(0.5, d.xpos[root][2])]
        r.update_scene(d, camera=camera if named else cam)
        frames.append(r.render().copy())
    r.close()
    with imageio.get_writer(out_mp4, fps=fps / stride, codec="libx264", quality=6,
                            macro_block_size=8) as wr:   # imageio-ffmpeg writes yuv420p
        for f in frames:
            wr.append_data(f)
    if still:
        from PIL import Image
        pick = np.linspace(0, len(frames) - 1, 5).astype(int)
        Image.fromarray(np.concatenate([frames[i] for i in pick], axis=1)).save(still, optimize=True)
