# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.3", "numpy>=1.26"]
# ///
"""The replay-to-sim MuJoCo worker (RFC-0006 part 8): scene building, kinematic replay and physics replay.

Runs as an isolated script (`uv run --script replay_sim.py request.json`), like the hylo-tools workers, so MuJoCo
stays out of the agent's process (hylo.codeagents.replay_geom.SimClient sends the requests); tests import it by path.
No Hylo imports.

request.json: {"ops": ["build" | "kinematic" | "physics", ...], "params": {...}, "inputs": {"arrays": in.npz},
"out_dir": path}; the worker writes out_dir/result.json {"ok", "outputs": {op: {...}}, "timing"} and out.npz.

Scene: a table box at the perceived plane, one body per object (free joint when the object moves in the recording,
fixed otherwise) with its visual mesh (group 2) and convex collision parts (group 3, from the convex tool), a camera
`replay_cam` at the recording's camera pose and field of view, and the agent: the MuJoCo Menagerie Panda (the model
file RobotSpec names, attached at the robot base, gravity-compensated, with flat finger pads and a stiffer grip like
mujoco.panda_table) or, for human videos, a free-floating parallel gripper (TCP convention: +x approach, +y closing,
13 cm stroke) driven by a mocap weld, or a free-floating multi-finger hand (the hand body of a hand model file, e.g. the floating-hand
MJCF of mujoco.dex_table, attached under a gravity-compensated free base that follows a mocap target through a weld,
finger position servos from that model; its own links do not collide with each other).

For the hand agent the arrays carry `tcp` (T, 4, 4) TCP poses (params hand.tcp_in_ee maps them to the palm) and
`fingers` (T, n) finger joint targets; its "opening" is the mean finger closure (1 at HandSpec open, 0 at closed).

kinematic: the recorded motion written straight into qpos (robot joints, fingers, objects at their tracked poses);
  reports interpenetration of the agent with the objects and the pad-midpoint to object distance while held.
physics: the robot follows joint-position targets (linearly interpolated between frames) or the gripper its TCP path,
  the hand closes on the given command, and the objects move only by contact; per frame qpos, object poses, the
  measured opening and finger contacts.
"""

from __future__ import annotations

import json
import math
import sys
import time as _time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PREFIX = "panda/"
JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HAND_CTRL_MAX = 255.0
FINGER_TRAVEL = 0.04
GRIPPER_STROKE = 0.065             # floating gripper: each finger, metres (13 cm max opening)


def pose(R: np.ndarray | None = None, p: Any = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


@dataclass
class SimObject:
    name: str
    parts: list[str]                     # convex collision parts: OBJ files in the body frame
    visual: str | None                   # visual mesh OBJ (body frame)
    pose0: np.ndarray
    rgba: tuple[float, ...] = (0.7, 0.7, 0.7, 1.0)
    dynamic: bool = True
    mass: float = 0.1
    label: str = ""


def _f(v: Sequence[float]) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _xyaxes(R_world_cam_cv: np.ndarray) -> str:
    """MJCF camera axes from an OpenCV camera rotation (x right, y down, z forward); MuJoCo cameras look along -z
    with +y up."""
    x, y = R_world_cam_cv[:, 0], -R_world_cam_cv[:, 1]
    return _f([*x, *y])


@dataclass
class Scene:
    path: Path
    agent: str                                     # panda | gripper | none
    objects: list[SimObject]
    table_z: float
    model: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    hand: dict[str, Any] | None = None             # agent "hand": xml, ee, joints, tcp_in_ee, open, closed, kp_scale

    def load(self) -> Any:
        import mujoco

        if self.model is None:
            self.model = mujoco.MjModel.from_xml_path(str(self.path))
        return self.model


def _object_xml(o: SimObject) -> tuple[str, str]:
    assets, geoms = [], []
    if o.visual:
        assets.append(f'<mesh name="{o.name}_visual" file="{o.visual}"/>')
        geoms.append(f'<geom name="{o.name}_vis" type="mesh" mesh="{o.name}_visual" group="2" contype="0" '
                     f'conaffinity="0" mass="0" rgba="{_f(o.rgba)}"/>')
    per = o.mass / max(1, len(o.parts))
    for i, path in enumerate(o.parts):
        assets.append(f'<mesh name="{o.name}_part_{i:03d}" file="{path}"/>')
        vis = "" if o.visual else f' rgba="{_f(o.rgba)}"'
        geoms.append(f'<geom name="{o.name}_col_{i:03d}" type="mesh" mesh="{o.name}_part_{i:03d}" '
                     f'group="{3 if o.visual else 2}" mass="{per:.6g}" condim="4" '
                     f'friction="1.0 0.01 0.001" solref="0.005 1"{vis}/>')
    p, q = o.pose0[:3, 3], _quat(o.pose0[:3, :3])
    joint = f'<freejoint name="obj_{o.name}"/>' if o.dynamic else ""
    body = f'<body name="obj_{o.name}" pos="{_f(p)}" quat="{_f(q)}">{joint}{"".join(geoms)}</body>'
    return "".join(assets), body


def _gripper_xml(T0: np.ndarray) -> tuple[str, str, str]:
    p, q = T0[:3, 3], _quat(T0[:3, :3])
    s = GRIPPER_STROKE
    body = f"""
    <body name="gripper_target" mocap="true" pos="{_f(p)}" quat="{_f(q)}"/>
    <body name="gripper" pos="{_f(p)}" quat="{_f(q)}" gravcomp="1">
      <freejoint name="gripper_free"/>
      <geom name="gripper_palm" type="box" size="0.018 {s + 0.02:.4f} 0.016" pos="-0.05 0 0" rgba="0.2 0.2 0.22 1"
            mass="0.3"/>
      <geom name="gripper_wrist" type="cylinder" size="0.022 0.03" pos="-0.095 0 0" quat="0.7071 0 0.7071 0"
            rgba="0.85 0.85 0.85 1" mass="0.1"/>
      <body name="gripper_left" pos="0 0 0" gravcomp="1">
        <joint name="gripper_left" type="slide" axis="0 1 0" range="0 {s}" damping="5"/>
        <geom name="gripper_left_pad" type="box" size="0.02 0.005 0.012" pos="-0.01 0.005 0" rgba="0.9 0.9 0.9 1"
              condim="4" friction="1.2 0.01 0.001" solref="0.005 1" mass="0.05"/>
      </body>
      <body name="gripper_right" pos="0 0 0" gravcomp="1">
        <joint name="gripper_right" type="slide" axis="0 -1 0" range="0 {s}" damping="5"/>
        <geom name="gripper_right_pad" type="box" size="0.02 0.005 0.012" pos="-0.01 -0.005 0" rgba="0.9 0.9 0.9 1"
              condim="4" friction="1.2 0.01 0.001" solref="0.005 1" mass="0.05"/>
      </body>
    </body>"""
    eq = '<weld body1="gripper_target" body2="gripper" solref="0.02 1"/>'
    act = (f'<position name="gripper_left" joint="gripper_left" kp="400" ctrlrange="0 {s}" forcerange="-25 25"/>'
           f'<position name="gripper_right" joint="gripper_right" kp="400" ctrlrange="0 {s}" forcerange="-25 25"/>')
    return body, eq, act


def _quiet_attach(frame: Any, body: Any) -> Any:
    """Attach without MuJoCo's attach warnings on the C stderr."""
    import os
    import warnings

    try:
        fd = os.dup(2)
    except OSError:
        return frame.attach_body(body, "", "")
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return frame.attach_body(body, "", "")
    finally:
        os.dup2(fd, 2)
        os.close(fd)


def _name_nested_defaults(xml: str, root: str) -> str:
    """An attached model's own top default comes out unnamed inside the main one, which MJCF rejects: name it, and
    make it the child class of the attached root body so the elements under it keep those defaults."""
    out, seen, cls = [], 0, None
    for ln in xml.splitlines():
        if ln.strip() == "<default>":
            seen += 1
            if seen > 1:
                cls = f"attached_main_{seen}"
                ln = ln.replace("<default>", f'<default class="{cls}">')
        out.append(ln)
    if cls is not None:
        for i, ln in enumerate(out):
            if f'<body name="{root}"' in ln and "childclass=" not in ln:
                out[i] = ln.replace(f'<body name="{root}"', f'<body name="{root}" childclass="{cls}"', 1)
                break
    return "\n".join(out) + "\n"


def attach_hand(spec: Any, hand: dict[str, Any], T0: np.ndarray) -> None:
    """The floating hand: mocap `hand_target` welded to the free body `hand` (gravity-compensated); the hand body
    `hand["ee"]` of the model file `hand["xml"]` (with its fingers and their servos, nothing above it) attached under
    it at the palm frame; finger servos scaled by hand["kp_scale"]."""
    import mujoco

    T_ee = T0 @ np.linalg.inv(np.asarray(hand["tcp_in_ee"], float))
    p, q = T_ee[:3, 3], _quat(T_ee[:3, :3])
    spec.worldbody.add_body(name="hand_target", mocap=True, pos=list(p), quat=list(q))
    base = spec.worldbody.add_body(name="hand", pos=list(p), quat=list(q), gravcomp=1.0)
    base.add_freejoint(name="hand_free")
    # a non-colliding ball gives the base enough rotational inertia for the weld to stay stable with light hands
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0, 0], mass=0.5, contype=0, conaffinity=0, group=4)
    hs = mujoco.MjSpec.from_file(str(hand["xml"]))
    fingers = set(hand["joints"])
    for a in list(hs.actuators):
        if a.target not in fingers:
            hs.delete(a)
    root = hs.body(hand["ee"])
    for b in hs.bodies:
        b.gravcomp = 1.0
    for g in hs.geoms:
        if g.contype or g.conaffinity:
            g.friction = [1.0, 0.01, 0.001]
            g.condim = 4
            g.solref = [0.004, 1.0]
    k = float(hand.get("kp_scale", 1.0))
    for a in hs.actuators:
        a.gainprm[0] *= k
        a.biasprm[1] *= k
    root.pos, root.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    before = {b.name for b in spec.bodies}
    _quiet_attach(base.add_frame(), root)
    hb = ["hand"] + [b.name for b in spec.bodies if b.name not in before]
    for i, a in enumerate(hb):
        for b in hb[i + 1:]:
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = a, b
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="hand_target", name2="hand", objtype=mujoco.mjtObj.mjOBJ_BODY)
    eq.solref = [0.01, 1.0]
    eq.data = [0.0] * 6 + [1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * (len(eq.data) - 11)   # anchor at the origin, no offset


def attach_panda(spec: Any, model_path: str, base: np.ndarray) -> None:
    """The Menagerie Panda attached at `base` with names prefixed "panda/": absolute mesh paths, gravity
    compensation, flat finger pads over the fingertip boxes and a stiffer finger servo."""
    import mujoco

    folder = Path(model_path).parent
    panda = mujoco.MjSpec.from_file(str(model_path))
    meshdir = folder / (panda.meshdir or "")
    for mesh in panda.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((meshdir / mesh.file).resolve())
    for body in panda.bodies:
        body.gravcomp = 1.0
    for finger in ("left_finger", "right_finger"):
        panda.body(finger).add_geom(name=f"{finger}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0088, 0.0012, 0.0105],
                                    pos=[0.0, 0.0007, 0.0445], group=3, contype=1, conaffinity=1, condim=4,
                                    friction=[1.0, 0.01, 0.001], solref=[0.005, 1.0])
    grip = panda.actuator("actuator8")
    grip.gainprm[0] *= 5.0
    grip.biasprm[1] *= 5.0
    grip.biasprm[2] *= 5.0
    for key in list(panda.keys):
        panda.delete(key)
    frame = spec.worldbody.add_frame(pos=list(base[:3, 3]), quat=list(_quat(base[:3, :3])))
    frame.attach_body(panda.body("link0"), PREFIX, "")


def build_scene(path: str | Path, *, table_z: float, table_extent: np.ndarray, objects: list[SimObject],
                agent: str, arm_model: str | None = None, base: np.ndarray | None = None,
                camera: dict[str, Any] | None = None, gripper_T0: np.ndarray | None = None,
                image_size: tuple[int, int] = (640, 480), hand: dict[str, Any] | None = None) -> Scene:
    """Write the scene MJCF to `path` (meshes referenced by absolute paths) and check that it compiles."""
    import mujoco

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = np.asarray(table_extent, float)
    lo, hi = ext[:, 0] - 0.15, ext[:, 1] + 0.15
    if agent == "panda" and base is not None:            # the table reaches under the robot base
        lo = np.minimum(lo, base[:2, 3] - 0.15)
        hi = np.maximum(hi, base[:2, 3] + 0.15)
    centre, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.2)
    assets, bodies = [], []
    for o in objects:
        a, b = _object_xml(o)
        assets.append(a)
        bodies.append(b)
    eq = act = grip = ""
    if agent == "gripper":
        grip, eq, act = _gripper_xml(gripper_T0 if gripper_T0 is not None else pose(None, [0, 0, table_z + 0.3]))
    cam = ""
    if camera is not None:
        T, K = np.asarray(camera["T_world_cam"], float), np.asarray(camera["K"], float)
        fovy = math.degrees(2 * math.atan(0.5 * camera["height"] / K[1, 1]))
        cam = f'<camera name="replay_cam" pos="{_f(T[:3, 3])}" xyaxes="{_xyaxes(T[:3, :3])}" fovy="{fovy:.4f}"/>'
    w, h = image_size
    xml = f"""<mujoco model="replay">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"><flag multiccd="enable"/></option>
  <size memory="40M"/>
  <visual><global offwidth="{max(w, 64)}" offheight="{max(h, 64)}"/><map znear="0.01" zfar="20"/>
    <headlight ambient="0.12 0.12 0.12" diffuse="0.25 0.25 0.25" specular="0.02 0.02 0.02"/></visual>
  <asset>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.62 0.52 0.40" rgb2="0.56 0.47 0.36" mark="random"
             random="0.02" width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="4 4" reflectance="0.02"/>
    {"".join(assets)}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.5 -0.6 2.5" dir="-0.2 0.3 -1" diffuse="0.3 0.3 0.3"/>
    {cam}
    <geom name="table" type="box" size="{half[0]:.4f} {half[1]:.4f} 0.02" pos="{centre[0]:.4f} {centre[1]:.4f} {table_z - 0.02:.4f}"
          material="table" friction="0.9 0.01 0.001"/>
    {"".join(bodies)}
    {grip}
  </worldbody>
  {f"<equality>{eq}</equality>" if eq else ""}
  {f"<actuator>{act}</actuator>" if act else ""}
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    if agent == "panda":
        if arm_model is None or base is None:
            raise ValueError("a Panda scene needs the arm model and its base pose")
        attach_panda(spec, arm_model, base)
    if agent == "hand":
        if hand is None:
            raise ValueError("a hand scene needs params hand {xml, ee, joints, tcp_in_ee, open, closed}")
        attach_hand(spec, hand, gripper_T0 if gripper_T0 is not None else pose(None, [0, 0, table_z + 0.3]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.compile()
        path.write_text(_name_nested_defaults(spec.to_xml(), (hand or {}).get("ee", "")))
    scene = Scene(path, agent, objects, float(table_z), info={"table_centre": centre.tolist(), "table_half": half.tolist()},
                  hand=hand)
    m = scene.load()
    scene.info.update({"nq": int(m.nq), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
                       "objects": [o.name for o in objects], "dynamic": [o.name for o in objects if o.dynamic]})
    return scene


# ------------------------------------------------------------------------------------------------ indices
class Layout:
    """qpos / ctrl addresses of the agent and the objects in a compiled replay scene."""

    def __init__(self, scene: Scene) -> None:
        import mujoco

        m = scene.load()
        self.m, self.scene = m, scene

        def jadr(name: str) -> int:
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])

        self.obj_q = {o.name: jadr(f"obj_{o.name}") for o in scene.objects if o.dynamic}
        self.obj_body = {o.name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{o.name}") for o in scene.objects}
        self.obj_geoms = {n: set(np.flatnonzero(m.geom_bodyid == b).tolist()) for n, b in self.obj_body.items()}
        self.finger_geoms: set[int] = set()
        if scene.agent == "panda":
            self.arm_q = np.array([jadr(PREFIX + j) for j in JOINTS])
            self.finger_q = np.array([jadr(PREFIX + f) for f in ("finger_joint1", "finger_joint2")])
            fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + f) for f in ("left_finger", "right_finger")]
            self.fingers = fb
            self.agent_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(PREFIX)}
        elif scene.agent == "gripper":
            self.free_q = jadr("gripper_free")
            self.finger_q = np.array([jadr("gripper_left"), jadr("gripper_right")])
            self.fingers = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("gripper_left", "gripper_right")]
            self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_target")])
            gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            self.agent_bodies = {gb, *self.fingers}
        elif scene.agent == "hand":
            h = scene.hand or {}
            self.free_q = jadr("hand_free")
            self.finger_q = np.array([jadr(j) for j in h["joints"]])
            act = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
            self.finger_act = np.array([act.get(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j), -1) for j in h["joints"]])
            self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_target")])
            hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
            sub = {hb}
            for b in range(hb + 1, m.nbody):
                if m.body_parentid[b] in sub:
                    sub.add(b)
            self.agent_bodies = sub
            self.fingers = sorted(sub - {hb})
            self.tcp_in_ee = np.asarray(h["tcp_in_ee"], float)
            self.ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, h.get("ee", "hand"))
            self.hand_open, self.hand_closed = np.asarray(h["open"], float), np.asarray(h["closed"], float)
        else:
            self.fingers, self.agent_bodies = [], set()
        self.finger_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, self.fingers)).tolist())
        self.agent_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, list(self.agent_bodies))).tolist())
        self.pads = [g for g in sorted(self.finger_geoms) if (m.geom(g).name or "").endswith("_pad")]

    def tcp_point(self, d: Any) -> np.ndarray | None:
        """Midpoint of the finger pads (the TCP of a parallel hand); a multi-finger hand's TCP."""
        if self.scene.agent == "hand":
            T = pose(d.xmat[self.ee_body].reshape(3, 3), d.xpos[self.ee_body]) @ self.tcp_in_ee
            return T[:3, 3]
        return np.mean([d.geom_xpos[g] for g in self.pads], axis=0) if self.pads else None

    def set_agent(self, d: Any, q: np.ndarray | None, tcp: np.ndarray | None, opening: float) -> None:
        if self.scene.agent == "panda" and q is not None:
            d.qpos[self.arm_q] = q
            d.qpos[self.finger_q] = FINGER_TRAVEL * float(np.clip(opening, 0, 1))
        elif self.scene.agent == "gripper" and tcp is not None:
            a = self.free_q
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = tcp[:3, 3], _quat(tcp[:3, :3])
            d.qpos[self.finger_q] = GRIPPER_STROKE * float(np.clip(opening, 0, 1))
        elif self.scene.agent == "hand" and tcp is not None:
            T_ee = tcp @ np.linalg.inv(self.tcp_in_ee)
            a = self.free_q
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = T_ee[:3, 3], _quat(T_ee[:3, :3])
            if q is not None:
                d.qpos[self.finger_q] = q

    def hand_target(self, d: Any, tcp: np.ndarray) -> None:
        T_ee = tcp @ np.linalg.inv(self.tcp_in_ee)
        d.mocap_pos[self.mocap], d.mocap_quat[self.mocap] = T_ee[:3, 3], _quat(T_ee[:3, :3])

    def finger_ctrl(self, d: Any, q: np.ndarray) -> None:
        ok = self.finger_act >= 0
        d.ctrl[self.finger_act[ok]] = np.asarray(q, float)[ok]

    def set_object(self, d: Any, name: str, T: np.ndarray) -> None:
        if name in self.obj_q:
            a = self.obj_q[name]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = T[:3, 3], _quat(T[:3, :3])

    def object_pose(self, d: Any, name: str) -> np.ndarray:
        b = self.obj_body[name]
        return pose(d.xmat[b].reshape(3, 3), d.xpos[b])

    def opening(self, d: Any) -> float:
        if self.scene.agent == "hand":
            span = self.hand_open - self.hand_closed
            ok = np.abs(span) > 1e-6
            f = (d.qpos[self.finger_q][ok] - self.hand_closed[ok]) / span[ok]
            return float(np.clip(f, 0, 1).mean()) if ok.any() else 1.0
        travel = FINGER_TRAVEL if self.scene.agent == "panda" else GRIPPER_STROKE
        return float(np.clip(np.mean(d.qpos[self.finger_q]) / travel, 0, 1)) if len(self.finger_q) else 1.0

    def contacts(self, d: Any) -> dict[str, dict[str, Any]]:
        """Per object: fingers touching it, deepest penetration with the agent."""
        out = {n: {"fingers": 0, "agent_pen": 0.0} for n in self.obj_geoms}
        fingers_seen = {n: set() for n in self.obj_geoms}
        m = self.m
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for n, gs in self.obj_geoms.items():
                other = g2 if g1 in gs else g1 if g2 in gs else None
                if other is None:
                    continue
                if other in self.finger_geoms:
                    fingers_seen[n].add(int(m.geom_bodyid[other]))
                if other in self.agent_geoms:
                    out[n]["agent_pen"] = min(out[n]["agent_pen"], float(c.dist))
        for n in out:
            out[n]["fingers"] = len(fingers_seen[n])
        return out


# ------------------------------------------------------------------------------------------------ replays
def kinematic_replay(scene: Scene, q: np.ndarray | None, tcp: np.ndarray | None, opening: np.ndarray,
                     obj_poses: dict[str, np.ndarray], intervals: list[tuple[int, int]],
                     held: dict[int, str] | None = None) -> dict[str, Any]:
    """qpos per frame from the recorded motion; checks interpenetration and the hand-object distance while held."""
    import mujoco

    lay = Layout(scene)
    m = lay.m
    d = mujoco.MjData(m)
    T = len(opening)
    qpos = np.zeros((T, m.nq))
    pen = np.zeros(T)
    held_dist = []
    for t in range(T):
        mujoco.mj_resetData(m, d)
        lay.set_agent(d, None if q is None else q[t], None if tcp is None else tcp[t], opening[t])
        for n, P in obj_poses.items():
            lay.set_object(d, n, P[t])
        mujoco.mj_forward(m, d)
        qpos[t] = d.qpos
        con = lay.contacts(d)
        pen[t] = min([c["agent_pen"] for c in con.values()] + [0.0])
        if held and t in held and held[t] in lay.obj_body and (lay.pads or scene.agent == "hand"):
            mid = lay.tcp_point(d)
            held_dist.append(float(np.linalg.norm(lay.object_pose(d, held[t])[:3, 3] - mid)))
    return {"qpos": qpos, "penetration_max_m": round(float(-pen.min()), 4),
            "frames_penetrating_5mm": int(np.sum(pen < -0.005)),
            "held_hand_object_dist_m": None if not held_dist else {
                "mean": round(float(np.mean(held_dist)), 4), "max": round(float(np.max(held_dist)), 4)}}


def physics_replay(scene: Scene, time: np.ndarray, q: np.ndarray | None, tcp: np.ndarray | None,
                   hand_cmd: np.ndarray, obj_poses0: dict[str, np.ndarray], *, settle_s: float = 0.3,
                   substep_ctrl: bool = True) -> dict[str, Any]:
    """Drive the agent along the recorded motion and let the objects move by contact. Returns per-frame qpos,
    object poses, measured opening and finger contacts at the recording's frame times."""
    import mujoco

    lay = Layout(scene)
    m = lay.m
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    open0 = float(hand_cmd[0])
    lay.set_agent(d, None if q is None else q[0], None if tcp is None else tcp[0], open0)
    for n, P in obj_poses0.items():
        lay.set_object(d, n, P)
    if scene.agent == "panda":
        d.ctrl[:7] = q[0]
        d.ctrl[7] = HAND_CTRL_MAX * open0
    elif scene.agent == "gripper":
        d.mocap_pos[lay.mocap], d.mocap_quat[lay.mocap] = tcp[0][:3, 3], _quat(tcp[0][:3, :3])
        d.ctrl[:2] = GRIPPER_STROKE * open0
    elif scene.agent == "hand":
        lay.hand_target(d, tcp[0])
        lay.finger_ctrl(d, q[0])
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=int(settle_s / m.opt.timestep))
    T = len(time)
    qpos = np.zeros((T, m.nq))
    poses = {n: np.zeros((T, 4, 4)) for n in lay.obj_body}
    opening = np.zeros(T)
    fingers = {n: np.zeros(T, int) for n in lay.obj_body}
    t0 = d.time
    diverged = False

    def record(k: int) -> None:
        qpos[k] = d.qpos
        for n in poses:
            poses[n][k] = lay.object_pose(d, n)
        opening[k] = lay.opening(d)
        for n, c in lay.contacts(d).items():
            fingers[n][k] = c["fingers"]

    record(0)
    for k in range(1, T):
        dt = float(time[k] - time[k - 1])
        n = max(1, int(round(dt / m.opt.timestep)))
        for i in range(n):
            a = (i + 1) / n if substep_ctrl else 1.0
            if scene.agent == "panda":
                d.ctrl[:7] = (1 - a) * q[k - 1] + a * q[k]
                d.ctrl[7] = HAND_CTRL_MAX * float(hand_cmd[k])
            elif scene.agent == "hand":
                Tk = tcp[k].copy()
                Tk[:3, 3] = (1 - a) * tcp[k - 1][:3, 3] + a * tcp[k][:3, 3]
                if a <= 0.5:
                    Tk[:3, :3] = tcp[k - 1][:3, :3]
                lay.hand_target(d, Tk)
                lay.finger_ctrl(d, (1 - a) * q[k - 1] + a * q[k])
            elif scene.agent == "gripper":
                p = (1 - a) * tcp[k - 1][:3, 3] + a * tcp[k][:3, 3]
                d.mocap_pos[lay.mocap] = p
                d.mocap_quat[lay.mocap] = _quat(tcp[k][:3, :3] if a > 0.5 else tcp[k - 1][:3, :3])
                d.ctrl[:2] = GRIPPER_STROKE * float(hand_cmd[k])
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            diverged = True
            qpos[k:] = qpos[k - 1]
            for nm in poses:
                poses[nm][k:] = poses[nm][k - 1]
            break
        record(k)
    return {"qpos": qpos, "poses": poses, "opening": opening, "fingers": fingers, "sim_time": float(d.time - t0),
            "diverged": diverged}


# ------------------------------------------------------------------------------------------------ checks
# ------------------------------------------------------------------------------------------------ requests
def objects_from(params: dict[str, Any]) -> list[SimObject]:
    return [SimObject(o["name"], list(o["parts"]), o.get("visual"), np.asarray(o["pose0"], float),
                      tuple(o.get("rgba", (0.7, 0.7, 0.7, 1.0))), bool(o.get("dynamic", True)), float(o.get("mass", 0.1)),
                      o.get("label", "")) for o in params["objects"]]


def scene_from(params: dict[str, Any]) -> Scene:
    return Scene(Path(params["scene_path"]), params["agent"], objects_from(params), float(params["table_z"]),
                 hand=params.get("hand"))


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Run the request's ops in order; returns {"outputs": {op: summary}, "arrays": {name: array}}."""
    p = request["params"]
    arrays: dict[str, np.ndarray] = {}
    if request.get("inputs", {}).get("arrays"):
        with np.load(request["inputs"]["arrays"]) as z:
            arrays = {k: z[k] for k in z.files}
    outputs: dict[str, Any] = {}
    out_arrays: dict[str, np.ndarray] = {}
    scene = None
    for op in request["ops"]:
        if op == "build":
            cam = p.get("camera")
            scene = build_scene(p["scene_path"], table_z=p["table_z"], table_extent=np.asarray(p["table_extent"]),
                                objects=objects_from(p), agent=p["agent"], arm_model=p.get("arm_model"),
                                base=None if p.get("base") is None else np.asarray(p["base"], float), camera=cam,
                                gripper_T0=None if p.get("gripper_T0") is None else np.asarray(p["gripper_T0"], float),
                                image_size=tuple(p.get("image_size", (640, 480))), hand=p.get("hand"))
            outputs["build"] = {"path": str(scene.path), **scene.info}
            continue
        scene = scene or scene_from(p)
        names = p.get("objects_names", [o.name for o in scene.objects])
        poses = {n: arrays[f"obj_{n}"] for n in names if f"obj_{n}" in arrays}
        q, tcp = arrays.get("q"), arrays.get("tcp")
        if scene.agent == "hand":
            q = arrays.get("fingers")                     # finger joint targets per frame
        if op == "kinematic":
            held = {int(k): v for k, v in (p.get("held") or {}).items()}
            kin = kinematic_replay(scene, q, tcp, arrays["opening"], poses, [], held)
            out_arrays["kin_qpos"] = kin.pop("qpos")
            outputs["kinematic"] = kin
        elif op == "physics":
            ph = physics_replay(scene, arrays["time"], q, tcp, arrays["hand_cmd"], {n: P[0] for n, P in poses.items()})
            out_arrays["phys_qpos"], out_arrays["phys_opening"] = ph["qpos"], ph["opening"]
            for n in ph["poses"]:
                out_arrays[f"phys_pose_{n}"] = ph["poses"][n]
                out_arrays[f"phys_fingers_{n}"] = ph["fingers"][n]
            outputs["physics"] = {"objects": list(ph["poses"]), "sim_time": ph["sim_time"], "diverged": ph["diverged"]}
        else:
            raise ValueError(f"unknown op {op!r}")
    return {"outputs": outputs, "arrays": out_arrays}


def main(argv: list[str]) -> int:
    request = json.loads(Path(argv[0]).read_text())
    out = Path(request["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = _time.perf_counter()
    try:
        res = handle(request)
        if res["arrays"]:
            np.savez_compressed(out / "out.npz", **res["arrays"])
        reply = {"ok": True, "outputs": res["outputs"], "timing": {"wall_s": round(_time.perf_counter() - t0, 3)}}
    except Exception as exc:  # noqa: BLE001 - the error goes back to the caller
        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
    (out / "result.json").write_text(json.dumps(reply, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return 0 if reply["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
