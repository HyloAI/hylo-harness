"""A dexterous hand on its own: the subtree of an arm model's hand body, cut out of any MJCF (or URDF) with MuJoCo's
model editor and hung under a mocap root so its pose in the world can be set freely. The grasp planner and the
retargeting solver work on it.

    hand = HandModel.from_arm(robot_spec.arm("right"))      # the hand of an arm, as its ArmSpec describes it
    hand = HandModel(model_path, "palm", hand_spec)          # any model file, its hand body and a HandSpec
    hand.set(T_world_tcp, q)                                 # place it (TCP pose) with finger joints q
    pts, link = hand.surface()                               # collision surface samples in the world, link per point

Fingers are found from the kinematic tree: each child chain of the hand body that holds finger joints is one finger
(its joints in HandSpec.joints order, its links, its tip: the collision point of its last link farthest from the
finger's first joint). The thumb is the finger whose names say so ("thumb", "th"), else the one whose base lies
farthest from the others; the other fingers are named by hints in their body and joint names (index / ff, middle /
mf, ring / rf, little / lf), else in order. The human wrist's place on the hand (for retargeting) is a site named
"wrist" on the hand body when the model has one, else the end of the palm's collision geometry opposite the fingers.
"""

from __future__ import annotations

import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import mujoco
import numpy as np

from hylo.core.robot import ArmSpec, HandSpec

from .geometry import inv, pose

FINGER_HINTS = (("thumb", ("thumb", "th")), ("index", ("index", "ff", "first", "if")),
                ("middle", ("middle", "mf", "mid")), ("ring", ("ring", "rf")),
                ("little", ("little", "pinky", "lf", "pf")))


@contextmanager
def _quiet_stderr() -> Iterator[None]:
    """MuJoCo's model editor prints attach warnings on the C stderr; keep them out of logs."""
    try:
        fd = os.dup(2)
    except OSError:
        yield
        return
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
    finally:
        os.dup2(fd, 2)
        os.close(fd)


def _absolute_assets(spec: mujoco.MjSpec, model_path: Path) -> None:
    base = model_path.parent
    meshdir = base / (spec.meshdir or "")
    texdir = base / (spec.texturedir or "")
    for m in spec.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str((meshdir / m.file).resolve())
    for t in spec.textures:
        if getattr(t, "file", "") and not Path(t.file).is_absolute():
            t.file = str((texdir / t.file).resolve())


def extract_hand(model_path: str | Path, ee_body: str, finger_joints: Sequence[str], *,
                 root: str = "hand_root") -> mujoco.MjSpec:
    """A new model spec holding only `ee_body`'s subtree under a mocap body `root` (world pose = the ee frame).
    Joints of the ee body itself (the arm's last joints) are dropped, actuators kept only on finger joints, and
    meshes no geom uses are removed, so only the hand's mesh files are read."""
    path = Path(model_path).expanduser().resolve()
    src = mujoco.MjSpec.from_file(str(path))
    if path.suffix.lower() == ".urdf":
        src.compiler.fusestatic = False
    _absolute_assets(src, path)
    body = src.body(ee_body)
    if body is None:
        raise KeyError(f"{path.name}: no body {ee_body!r}")
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    spec.option.timestep = src.option.timestep
    rb = spec.worldbody.add_body(name=root, mocap=True)
    with _quiet_stderr():
        rb.add_frame().attach_body(body, "", "")
    own = spec.body(ee_body)
    own.pos, own.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]      # the root's frame is the ee frame
    for j in list(own.joints):
        spec.delete(j)
    keep = set(finger_joints)
    for a in list(spec.actuators):
        if a.target not in keep:
            spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for s in list(spec.sensors):
        spec.delete(s)
    for t in list(spec.tendons):
        spec.delete(t)
    for e in list(spec.equalities):
        spec.delete(e)
    for ex in list(spec.excludes):
        names = {b.name for b in spec.bodies}
        if ex.bodyname1 not in names or ex.bodyname2 not in names:
            spec.delete(ex)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for m in list(spec.meshes):
        if m.name not in used:
            spec.delete(m)
    return spec


def _subtree(m: mujoco.MjModel, root: int) -> list[int]:
    out = [root]
    for b in range(root + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.append(b)
    return out


def _is_collision(m: mujoco.MjModel, g: int) -> bool:
    return bool(m.geom_contype[g] or m.geom_conaffinity[g])


@dataclass
class Finger:
    name: str                      # thumb, index, middle, ring, little, or finger<k>
    joints: list[int]              # indices into HandSpec.joints
    bodies: list[int]              # body ids of its links
    tip_body: int
    tip_local: np.ndarray          # tip point in the tip body's frame
    base_body: int                 # the body of its first joint


@dataclass
class HandModel:
    """See the module docstring. `spacing`: distance between collision surface samples (m)."""
    model_path: str
    ee_body: str
    hand: HandSpec
    name: str = "hand"
    spacing: float = 0.004
    wrist_in_ee: Sequence[float] | None = None     # the human wrist's place on this hand (ee frame); None: from the model
    finger_names: dict[str, str] | None = None     # finger label -> a body name on that finger (overrides the hints)
    spec: mujoco.MjSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = str(Path(self.model_path).expanduser())
        self.spec = extract_hand(self.model_path, self.ee_body, self.hand.joints)
        with _quiet_stderr():
            self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.mocap = int(m.body_mocapid[m.body("hand_root").id])
        self.ee = m.body(self.ee_body).id
        self.tcp_in_ee = np.asarray(self.hand.tcp_in_ee, float).reshape(4, 4)
        jids = []
        for j in self.hand.joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid < 0:
                raise KeyError(f"{self.name}: finger joint {j!r} not under body {self.ee_body!r}")
            jids.append(jid)
        self.jids = np.array(jids, int)
        self.qadr = m.jnt_qposadr[self.jids]
        self.dofadr = m.jnt_dofadr[self.jids]
        self.lower, self.upper = m.jnt_range[self.jids, 0].copy(), m.jnt_range[self.jids, 1].copy()
        self.open, self.closed = np.asarray(self.hand.open, float), np.asarray(self.hand.closed, float)
        self.bodies = _subtree(m, self.ee)
        self._fingers()
        self._sample_surface()
        self.wrist_local = self._wrist() if self.wrist_in_ee is None else np.asarray(self.wrist_in_ee, float)

    @classmethod
    def from_arm(cls, arm: ArmSpec, **kw) -> "HandModel":
        from .kinematics import resolve_model_path

        return cls(resolve_model_path(arm.model), arm.ee_frame, arm.hand, name=arm.name, **kw)

    # ------------------------------------------------------------------ structure
    def _fingers(self) -> None:
        m = self.m
        children = [b for b in self.bodies if m.body_parentid[b] == self.ee]
        fingers: list[Finger] = []
        for c in children:
            sub = _subtree(m, c)
            js = [i for i, j in enumerate(self.jids) if m.jnt_bodyid[j] in sub]
            if not js:
                continue
            leaves = [b for b in sub if not any(m.body_parentid[x] == b for x in sub)]
            depth = {b: 0 for b in sub}
            for b in sub:
                if b != c:
                    depth[b] = depth[int(m.body_parentid[b])] + 1
            # the tip link: the deepest leaf that has collision geometry (else the deepest leaf)
            geomy = [b for b in leaves if any(_is_collision(m, g) for g in np.flatnonzero(m.geom_bodyid == b))]
            tip = max(geomy or leaves, key=lambda b: depth[b])
            if not geomy:            # a geometry-less tip site body: use its parent link's geometry
                while tip != c and not any(_is_collision(m, g) for g in np.flatnonzero(m.geom_bodyid == tip)):
                    tip = int(m.body_parentid[tip])
            base = int(m.jnt_bodyid[self.jids[js[0]]])
            fingers.append(Finger(f"finger{len(fingers)}", js, sub, tip, np.zeros(3), base))
        if not fingers:
            raise ValueError(f"{self.name}: no finger joints found under {self.ee_body!r}")
        self.fingers = fingers
        self._label_fingers()
        # tips: at the open pose, the tip link's collision point farthest from the finger's first joint
        self.set(np.eye(4), self.open)
        for f in self.fingers:
            gs = [g for g in np.flatnonzero(m.geom_bodyid == f.tip_body) if _is_collision(m, g)]
            local = self._geom_points(gs, f.tip_body, 0.002) if gs else np.zeros((1, 3))
            R, p = self.d.xmat[f.tip_body].reshape(3, 3), self.d.xpos[f.tip_body]
            world = local @ R.T + p
            anchor = self.d.xanchor[self.jids[f.joints[0]]]
            f.tip_local = local[int(np.argmax(np.linalg.norm(world - anchor, axis=1)))]
        self.thumb = next(i for i, f in enumerate(self.fingers) if f.name == "thumb")

    def _label_fingers(self) -> None:
        m = self.m
        names = {}
        if self.finger_names:
            for label, body in self.finger_names.items():
                b = m.body(body).id
                for i, f in enumerate(self.fingers):
                    if b in f.bodies:
                        names[i] = label
        else:
            for i, f in enumerate(self.fingers):
                words = " ".join([m.body(b).name for b in f.bodies] +
                                 [m.joint(int(self.jids[j])).name for j in f.joints]).lower()
                tokens = set(re.split(r"[^a-z]+", words)) | {t[:2] for t in re.split(r"[^a-z0-9]+", words) if t}
                for label, hints in FINGER_HINTS:
                    if label not in names.values() and any(h in tokens for h in hints):
                        names[i] = label
                        break
        if "thumb" not in names.values():
            self.set(np.eye(4), self.open)
            bases = np.array([self.d.xpos[f.base_body] for f in self.fingers])
            spread = [np.linalg.norm(bases - b, axis=1).sum() for b in bases]
            t = int(np.argmax(spread))
            names = {k: v for k, v in names.items() if k != t}
            names[t] = "thumb"
        order = ["index", "middle", "ring", "little"]
        free = [i for i in range(len(self.fingers)) if i not in names]
        for i in free:
            label = next(x for x in order + [f"finger{k}" for k in range(9)] if x not in names.values())
            names[i] = label
        for i, f in enumerate(self.fingers):
            f.name = names[i]

    def _wrist(self) -> np.ndarray:
        """The model's `wrist` site on the hand body, else the palm's far end from the fingers (ee frame)."""
        m = self.m
        for sid in range(m.nsite):
            if m.site_bodyid[sid] == self.ee and (m.site(sid).name or "").split("/")[-1] == "wrist":
                return m.site_pos[sid].copy()
        self.set_ee(np.eye(4), self.open)
        palm = self.local_pts[self.pt_body == self.ee]
        bases = self.finger_bases()
        if not len(palm):
            return np.zeros(3)
        fwd = bases.mean(0) - palm.mean(0)
        if np.linalg.norm(fwd) < 1e-6:
            return np.zeros(3)
        fwd /= np.linalg.norm(fwd)
        s = (palm - palm.mean(0)) @ fwd
        return palm.mean(0) + fwd * float(s.min())

    def finger(self, name: str) -> Finger:
        for f in self.fingers:
            if f.name == name:
                return f
        raise KeyError(f"{self.name}: no finger {name!r}; fingers: {[f.name for f in self.fingers]}")

    def _geom_points(self, geoms: Sequence[int], body: int, h: float) -> np.ndarray:
        """Surface samples of `geoms` in the frame of `body`."""
        from .collision import geom_surface

        rng = np.random.default_rng(0)
        out = []
        for g in geoms:
            local = geom_surface(self.m, int(g), h, rng)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
            out.append(local @ R.reshape(3, 3).T + self.m.geom_pos[g])
        return np.concatenate(out) if out else np.zeros((0, 3))

    def _sample_surface(self) -> None:
        m = self.m
        pts, owner = [], []
        for b in self.bodies:
            gs = [g for g in np.flatnonzero(m.geom_bodyid == b) if _is_collision(m, g)]
            if not gs:
                continue
            p = self._geom_points(gs, b, self.spacing)
            pts.append(p)
            owner.append(np.full(len(p), b))
        self.local_pts = np.concatenate(pts)
        self.pt_body = np.concatenate(owner)
        body_finger = {b: -1 for b in self.bodies}
        for i, f in enumerate(self.fingers):
            for b in f.bodies:
                body_finger[b] = i
        self.pt_finger = np.array([body_finger[b] for b in self.pt_body])
        self.link_ids = sorted(set(self.pt_body.tolist()))

    # ------------------------------------------------------------------ pose
    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.lower, self.upper)

    def set(self, T_world_tcp: np.ndarray, q: Sequence[float] | None = None) -> None:
        """Place the hand: TCP pose in the world and finger joints q (HandSpec.joints order; None: open)."""
        T_ee = np.asarray(T_world_tcp, float) @ inv(self.tcp_in_ee)
        self.set_ee(T_ee, q)

    def set_ee(self, T_world_ee: np.ndarray, q: Sequence[float] | None = None) -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(T_world_ee, float)[:3, :3].ravel())
        self.d.mocap_pos[self.mocap] = T_world_ee[:3, 3]
        self.d.mocap_quat[self.mocap] = quat
        self.d.qpos[self.qadr] = self.clip(self.open if q is None else q)
        mujoco.mj_kinematics(self.m, self.d)

    def ee_pose(self) -> np.ndarray:
        return pose(self.d.xmat[self.ee].reshape(3, 3), self.d.xpos[self.ee])

    def surface(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World collision samples after `set` (all, or those selected by a boolean mask), and their body ids."""
        lp, b = (self.local_pts, self.pt_body) if mask is None else (self.local_pts[mask], self.pt_body[mask])
        R = self.d.xmat[b].reshape(-1, 3, 3)
        return np.einsum("nij,nj->ni", R, lp) + self.d.xpos[b], b

    def tips(self) -> np.ndarray:
        """(n_fingers, 3) fingertip points in the world after `set`."""
        out = []
        for f in self.fingers:
            R = self.d.xmat[f.tip_body].reshape(3, 3)
            out.append(R @ f.tip_local + self.d.xpos[f.tip_body])
        return np.asarray(out)

    def tip_jacobians(self) -> np.ndarray:
        """(n_fingers, 3, n_joints) Jacobians of the fingertip points w.r.t. the finger joints, after `set` (needs
        mj_kinematics + mj_comPos, done here)."""
        mujoco.mj_comPos(self.m, self.d)
        out = np.zeros((len(self.fingers), 3, len(self.jids)))
        jacp = np.zeros((3, self.m.nv))
        for i, (f, p) in enumerate(zip(self.fingers, self.tips())):
            mujoco.mj_jac(self.m, self.d, jacp, None, p, f.tip_body)
            out[i] = jacp[:, self.dofadr]
        return out

    def wrist(self) -> np.ndarray:
        R = self.d.xmat[self.ee].reshape(3, 3)
        return R @ self.wrist_local + self.d.xpos[self.ee]

    def finger_bases(self) -> np.ndarray:
        return np.asarray([self.d.xanchor[self.jids[f.joints[0]]] for f in self.fingers])

    def points_in_tcp(self, q: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Collision samples in the TCP frame at finger joints q (None: open), and the finger index per point
        (-1: palm)."""
        self.set(np.eye(4), q)
        pts, _ = self.surface()
        return pts, self.pt_finger.copy()


__all__ = ["Finger", "HandModel", "extract_hand"]
