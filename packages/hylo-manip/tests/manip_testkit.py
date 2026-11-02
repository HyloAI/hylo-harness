"""Test fixtures for hylo-manip: a 7-DoF test arm with a parallel gripper (MJCF written here), its RobotSpec, the
Franka Panda spec when the Menagerie model is available, and a kinematic stub executor that renders depth, RGB and
segmentation by ray casting a MuJoCo scene."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec
from hylo.core.types import Observation

TCP_IN_HAND = [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.065], [0, 0, 0, 1]]   # +x along the hand's z
ARM_HOME = (0.0, 0.2, 0.0, 1.9, 0.0, 1.0, 0.0)

ARM_BODY = """
<body name="base">
  <geom type="cylinder" size="0.07 0.05" pos="0 0 0.05"/>
  <body name="link1" pos="0 0 0.1">
    <joint name="j1" axis="0 0 1" range="-2.9 2.9"/>
    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
    <body name="link2" pos="0 0 0.2">
      <joint name="j2" axis="0 1 0" range="-2.0 2.0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.045"/>
      <body name="link3" pos="0 0 0.35">
        <joint name="j3" axis="0 0 1" range="-2.9 2.9"/>
        <joint name="j4" axis="0 1 0" range="-2.6 2.6"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
        <body name="link4" pos="0 0 0.3">
          <joint name="j5" axis="0 0 1" range="-2.9 2.9"/>
          <joint name="j6" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.035"/>
          <body name="hand" pos="0 0 0.08">
            <joint name="j7" axis="0 0 1" range="-2.9 2.9"/>
            <geom type="box" size="0.025 0.06 0.015" pos="0 0 0.015" rgba="0.3 0.3 0.3 1"/>
            <body name="lf" pos="0 0 0.03">
              <joint name="f1" type="slide" axis="0 1 0" range="0 0.04"/>
              <geom type="box" size="0.01 0.005 0.025" pos="0 0.005 0.025" rgba="0.3 0.3 0.3 1"/>
            </body>
            <body name="rf" pos="0 0 0.03">
              <joint name="f2" type="slide" axis="0 -1 0" range="0 0.04"/>
              <geom type="box" size="0.01 0.005 0.025" pos="0 -0.005 0.025" rgba="0.3 0.3 0.3 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
</body>
"""

HEADER = """<compiler angle="radian" autolimits="true"/>
<default><joint damping="1"/><geom rgba="0.7 0.7 0.7 1"/></default>"""


def write_arm(directory: Path) -> Path:
    path = Path(directory) / "testarm.xml"
    path.write_text(f'<mujoco model="testarm">{HEADER}<worldbody>{ARM_BODY}</worldbody></mujoco>')
    return path


def arm_spec(model: Path, cameras=()) -> RobotSpec:
    hand = HandSpec(kind="parallel", tcp_in_ee=TCP_IN_HAND, open=(1.0,), closed=(0.0,), max_width=0.08,
                    joints=("f1", "f2"))
    arm = ArmSpec(name="arm", model=str(model), joints=tuple(f"j{i}" for i in range(1, 8)), ee_frame="hand",
                  lower=(-2.9, -2.0, -2.9, -2.6, -2.9, -2.2, -2.9), upper=(2.9, 2.0, 2.9, 2.6, 2.9, 2.2, 2.9),
                  home=ARM_HOME, hand=hand, action_slice=(0, 7), hand_slice=(7, 8))
    return RobotSpec(arms=(arm,), cameras=tuple(CameraSpec(name=c, height=96, width=128) for c in cameras),
                     table_height=0.0)


# ---------------------------------------------------------------------- Franka Panda (MuJoCo Menagerie)
PANDA_TCP = [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.1034], [0, 0, 0, 1]]


def find_panda() -> Path | None:
    cands = [os.environ.get("HYLO_PANDA_MJCF")]
    cache = Path.home() / ".cache" / "hylo" / "assets"
    if cache.exists():
        cands += [str(p) for p in sorted(cache.rglob("franka_emika_panda/panda.xml"))]
    cands.append("/tmp/claude-1002/-home-bingao-hyloai/49f6a616-2836-40ee-84cf-972b09b8fd75/scratchpad/menagerie/"
                 "franka_emika_panda/panda.xml")
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def panda_spec(model: Path) -> RobotSpec:
    hand = HandSpec(kind="parallel", tcp_in_ee=PANDA_TCP, open=(1.0,), closed=(0.0,), max_width=0.08,
                    joints=("finger_joint1", "finger_joint2"))
    arm = ArmSpec(name="arm", model=str(model), joints=tuple(f"joint{i}" for i in range(1, 8)), ee_frame="hand",
                  lower=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
                  upper=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
                  home=(0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853), hand=hand, action_slice=(0, 7),
                  hand_slice=(7, 8))
    return RobotSpec(arms=(arm,), table_height=0.0)


# ---------------------------------------------------------------------- stub executor
def look_at(eye, target) -> np.ndarray:
    """OpenCV camera pose (z forward, y down) at `eye` looking at `target`."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


CAMERAS = {"front": look_at(np.array([1.25, 0.0, 0.7]), [0.45, 0.0, 0.0]),
           "side": look_at(np.array([0.45, -0.85, 0.6]), [0.45, 0.0, 0.0]),
           "back": look_at(np.array([0.4, 0.75, 0.65]), [0.45, 0.0, 0.0])}


class StubExecutor:
    """A kinematic robot in a MuJoCo scene: joints track targets exactly; the hand moves 0.1 per step towards its
    command and stops on a block between the pads, which then moves with the TCP until the hand opens (it drops onto
    the table). Observations are ray-cast RGB-D with instance segmentation."""

    def __init__(self, directory: Path, objects: dict[str, tuple], spec: RobotSpec, max_steps: int = 2000,
                 hw=(96, 128), cameras=("front", "side", "back"), noise: float = 0.0, seed: int = 0) -> None:
        bodies = []
        for name, (pos, half, rgba, yaw) in objects.items():
            q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            bodies.append(f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{" ".join(map(str, q))}">'
                          f'<freejoint/><geom type="box" size="{" ".join(map(str, half))}" '
                          f'rgba="{" ".join(map(str, rgba))}"/></body>')
        xml = (f'<mujoco model="scene">{HEADER}<worldbody>'
               '<geom name="table" type="box" pos="0.5 0 -0.02" size="0.45 0.6 0.02" rgba="0.6 0.5 0.4 1"/>'
               f'{ARM_BODY}{"".join(bodies)}</worldbody></mujoco>')
        path = Path(directory) / "scene.xml"
        path.write_text(xml)
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.spec, self.arm = spec, spec.arms[0]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.arm.joints]
        self.qadr = self.m.jnt_qposadr[jid]
        self.fadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f)] for f in ("f1", "f2")]
        self.hand_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.objects = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in objects}
        self.half = {n: np.array(v[1], float) for n, v in objects.items()}
        self.q = np.array(self.arm.home, float)
        self.hand, self.cmd = 1.0, 1.0
        self.attached: tuple[str, np.ndarray] | None = None
        self.step, self.max_steps = 0, max_steps
        self.hw, self.cams = hw, cameras
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.log: list[np.ndarray] = []
        self._sync()

    # -- state
    def _sync(self) -> None:
        self.d.qpos[self.qadr] = self.q
        self.d.qpos[self.fadr] = 0.04 * self.hand
        mujoco.mj_kinematics(self.m, self.d)
        if self.attached:
            name, T_rel = self.attached
            self.set_pose(name, self.tcp() @ T_rel)
            mujoco.mj_kinematics(self.m, self.d)

    def tcp(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.hand_body].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.hand_body]
        return T @ np.asarray(TCP_IN_HAND, float)

    def pose(self, name: str) -> np.ndarray:
        b = self.objects[name]
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def set_pose(self, name: str, T: np.ndarray) -> None:
        adr = self.m.jnt_qposadr[self.m.body_jntadr[self.objects[name]]]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
        self.d.qpos[adr:adr + 3] = T[:3, 3]
        self.d.qpos[adr + 3:adr + 7] = q

    def _between_pads(self) -> tuple[str, float] | None:
        T = self.tcp()
        for name in self.objects:
            P = np.linalg.inv(T) @ self.pose(name)
            half = self.half[name]
            width = 2 * float(np.abs(P[1, :3]) @ half)          # extent along the closing axis
            reach_x = float(np.abs(P[0, :3]) @ half)
            if abs(P[0, 3]) < reach_x + 0.01 and abs(P[2, 3]) < 0.02 and abs(P[1, 3]) < 0.01:
                return name, width
        return None

    # -- Executor protocol
    def status(self) -> dict:
        return {"step": self.step, "max_steps": self.max_steps, "instruction": "test", "ended": False}

    def run(self, actions: np.ndarray) -> dict:
        n = 0
        for row in np.asarray(actions):
            self.log.append(row.copy())
            self.q = np.clip(row[:7], self.arm.lower, self.arm.upper)
            self.cmd = float(row[7])
            target = self.cmd
            held = self._between_pads()
            if held and self.cmd < self.hand:
                target = max(target, held[1] / 0.08)
            self.hand += float(np.clip(target - self.hand, -0.1, 0.1))
            if held and self.attached is None and self.cmd < 0.5 and abs(self.hand - held[1] / 0.08) < 1e-6:
                self.attached = (held[0], np.linalg.inv(self.tcp()) @ self.pose(held[0]))
            if self.attached and self.cmd > self.hand + 0.05:
                name = self.attached[0]
                self.attached = None
                T = self.pose(name)
                low = float(np.abs(T[2, :3]) @ self.half[name])
                T[2, 3] = low                                     # drops onto the table
                self.set_pose(name, T)
            self._sync()
            self.step += 1
            n += 1
        return {"steps": n, "ended": False, "success": None, "reason": None}

    def observation(self) -> Observation:
        images, cams = {}, {}
        H, W = self.hw
        f = (H / 2) / np.tan(np.radians(30))
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u, float)], -1).reshape(-1, 3)
        for name in self.cams:
            T = CAMERAS[name]
            dirs = rays @ T[:3, :3].T
            norm = np.linalg.norm(dirs, axis=1)
            gid = np.zeros(len(dirs), np.int32)
            dist = np.zeros(len(dirs))
            mujoco.mj_multiRay(self.m, self.d, T[:3, 3].copy(), (dirs / norm[:, None]).ravel(), None, 1, -1, gid,
                               dist, None, len(dirs), 10.0)
            hit = gid >= 0
            z = dist / norm + (self.rng.normal(0, self.noise, len(dirs)) if self.noise else 0.0)
            depth = np.where(hit, z, 0.0).astype(np.float32).reshape(H, W)
            rgb = np.zeros((len(dirs), 3), np.uint8)
            rgb[hit] = (self.m.geom_rgba[gid[hit], :3] * 255).astype(np.uint8)
            body = np.where(hit, self.m.geom_bodyid[np.maximum(gid, 0)], 0)
            seg = np.where(np.isin(body, list(self.objects.values())), body, 0).astype(np.int32)
            images[name], images[f"{name}.depth"], images[f"{name}.seg"] = rgb.reshape(H, W, 3), depth, seg.reshape(H, W)
            cams[name] = {"K": K.tolist(), "T_world_cam": T.tolist()}
        segments = {str(b): n.replace("_", " ") for n, b in self.objects.items()}
        return Observation(state={"q.arm": self.q.copy(), "hand.arm": np.array([self.hand])}, images=images,
                           extra={"cameras": cams, "segments": segments})


class KinematicExecutor:
    """State only: joints track targets exactly, the hand follows its command; no images."""

    def __init__(self, spec: RobotSpec, q=None) -> None:
        self.spec = spec
        self.q = {a.name: np.array(q if q is not None else a.home, float) for a in spec.arms}
        self.hand = {a.name: np.array(a.hand.open, float) for a in spec.arms}
        self.step, self.log = 0, []

    def status(self) -> dict:
        return {"step": self.step, "max_steps": 10000, "instruction": None, "ended": False}

    def run(self, actions) -> dict:
        for row in np.asarray(actions):
            self.log.append(row.copy())
            for a in self.spec.arms:
                self.q[a.name] = row[a.action_slice[0]:a.action_slice[1]].copy()
                self.hand[a.name] = row[a.hand_slice[0]:a.hand_slice[1]].copy()
            self.step += 1
        return {"steps": len(actions), "ended": False, "success": None, "reason": None}

    def observation(self) -> Observation:
        state = {f"q.{n}": q.copy() for n, q in self.q.items()}
        state.update({f"hand.{n}": h.copy() for n, h in self.hand.items()})
        return Observation(state=state)


def box_points(center, half, yaw=0.0, spacing=0.005, faces=("top", "sides")) -> np.ndarray:
    """Surface points of a box resting with its centre at `center` (no bottom face)."""
    hx, hy, hz = half
    pts = []
    xs = np.arange(-hx, hx + 1e-9, spacing)
    ys = np.arange(-hy, hy + 1e-9, spacing)
    zs = np.arange(-hz, hz + 1e-9, spacing)
    if "top" in faces:
        X, Y = np.meshgrid(xs, ys)
        pts.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, hz)])
    if "sides" in faces:
        for sx in (-hx, hx):
            Y, Z = np.meshgrid(ys, zs)
            pts.append(np.c_[np.full(Y.size, sx), Y.ravel(), Z.ravel()])
        for sy in (-hy, hy):
            X, Z = np.meshgrid(xs, zs)
            pts.append(np.c_[X.ravel(), np.full(X.size, sy), Z.ravel()])
    P = np.concatenate(pts)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return P @ R.T + np.asarray(center, float)
