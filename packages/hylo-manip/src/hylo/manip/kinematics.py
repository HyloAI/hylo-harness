"""Kinematics from a RobotSpec: each arm's model loaded with MuJoCo (MJCF directly, URDF through MuJoCo's importer),
forward kinematics of the TCP in the world, geometric Jacobians and damped-least-squares IK with joint limits."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from typing import Iterator

import mujoco
import numpy as np

from hylo.core.robot import ArmSpec, RobotSpec

from .geometry import inv, rotvec

_MODELS: dict[str, mujoco.MjModel] = {}


def resolve_model_path(ref: str) -> str:
    """An absolute path, or a "pkg://<distribution>/<path>" resource of an installed distribution."""
    if not ref.startswith("pkg://"):
        return os.path.expanduser(ref)
    dist_name, _, rel = ref[len("pkg://"):].partition("/")
    dist = importlib.metadata.distribution(dist_name)
    path = str(dist.locate_file(rel))
    if os.path.exists(path):
        return path
    for f in dist.files or ():
        if str(f).replace("\\", "/").endswith(rel):
            return str(dist.locate_file(f))
    raise FileNotFoundError(f"{ref}: not found in distribution {dist_name}")


def load_model(ref: str) -> mujoco.MjModel:
    path = resolve_model_path(ref)
    if path not in _MODELS:
        if path.lower().endswith(".urdf"):
            spec = mujoco.MjSpec.from_file(path)
            spec.compiler.fusestatic = False      # keep fixed links (e.g. a flange or hand frame) as bodies
            model = spec.compile()
        else:
            model = mujoco.MjModel.from_xml_path(path)
        _MODELS[path] = model
    return _MODELS[path]


def _mat(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(4, 4)


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float        # metres
    rot_err: float        # radians (of the constrained axes when a rotation is free)
    ok: bool


class ArmModel:
    """One arm in its MuJoCo model. Arms loaded from the same file share the model and its data."""

    def __init__(self, arm: ArmSpec, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.spec, self.m, self.d = arm, model, data
        self.name = arm.name
        jids = []
        for j in arm.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid < 0:
                raise KeyError(f"arm {arm.name}: joint {j!r} not in {arm.model}")
            if int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"arm {arm.name}: joint {j!r} must be a hinge or slide joint")
            jids.append(jid)
        self.jids = np.array(jids)
        self.qadr = model.jnt_qposadr[self.jids]
        self.hinge = model.jnt_type[self.jids] == int(mujoco.mjtJoint.mjJNT_HINGE)
        self.ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_frame)
        if self.ee < 0:
            raise KeyError(f"arm {arm.name}: ee_frame {arm.ee_frame!r} is not a body of {arm.model}")
        self.tcp_in_ee = _mat(arm.hand.tcp_in_ee)
        self.base = _mat(arm.base_in_world)
        self.lower, self.upper = np.array(arm.lower, float), np.array(arm.upper, float)
        self.home = np.array(arm.home, float)
        self.n = len(jids)
        fj = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm.hand.joints]
        self.finger_adr = np.array([model.jnt_qposadr[j] for j in fj if j >= 0], dtype=int)
        self.finger_range = np.array([model.jnt_range[j] for j in fj if j >= 0]).reshape(-1, 2)
        h = arm.hand
        self.hand_open, self.hand_closed = np.array(h.open, float), np.array(h.closed, float)

    # ------------------------------------------------------------------ state
    def set_q(self, q) -> None:
        self.d.qpos[self.qadr] = q

    def set_hand(self, cmd) -> None:
        """Finger joints at command `cmd`. A parallel hand's one value (0 closed .. 1 open) maps to the joint range's
        lower .. upper end; a dexterous hand's command with one value per finger joint is those joint positions."""
        if len(self.finger_adr) == 0 or cmd is None:
            return
        c = np.asarray(cmd, float).ravel()
        if self.spec.hand.kind == "dexterous" and len(c) == len(self.finger_adr):
            self.d.qpos[self.finger_adr] = np.clip(c, self.finger_range[:, 0], self.finger_range[:, 1])
            return
        span = self.hand_open[0] - self.hand_closed[0]
        c = float(np.clip((np.asarray(cmd, float).ravel()[0] - self.hand_closed[0]) / (span or 1.0), 0.0, 1.0))
        lo, hi = self.finger_range[:, 0], self.finger_range[:, 1]
        self.d.qpos[self.finger_adr] = lo + c * (hi - lo)

    def forward(self) -> None:
        mujoco.mj_kinematics(self.m, self.d)

    def ee_model(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.ee].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.ee]
        return T

    def tcp(self) -> np.ndarray:
        """TCP pose in the world after `forward()`."""
        return self.base @ self.ee_model() @ self.tcp_in_ee

    def fk(self, q) -> np.ndarray:
        """TCP pose (4x4, world) at joint positions q."""
        self.set_q(q)
        self.forward()
        return self.tcp()

    def jacobian(self, T_tcp: np.ndarray) -> np.ndarray:
        """(6, n) world Jacobian of the TCP (linear; angular) after `forward()`."""
        p = (inv(self.base) @ T_tcp)[:3, 3]
        axes = self.d.xaxis[self.jids]
        anchors = self.d.xanchor[self.jids]
        J = np.zeros((6, self.n))
        J[:3] = np.where(self.hinge[:, None], np.cross(axes, p - anchors), axes).T
        J[3:] = np.where(self.hinge[:, None], axes, 0.0).T
        Rb = self.base[:3, :3]
        return np.vstack([Rb @ J[:3], Rb @ J[3:]])

    # ------------------------------------------------------------------ IK
    def solve(self, target: np.ndarray, q0, free: str | None = None, pos_tol: float = 5e-4,
              rot_tol: float = 5e-3, iters: int = 150, damping: float = 0.02, w_rot: float = 0.3) -> IKResult:
        """Damped least squares from q0, clipped to the joint limits."""
        q = np.clip(np.asarray(q0, float), self.lower, self.upper)
        axis = {"x": 0, "z": 2}.get(free) if free else None
        best = None
        for _ in range(iters):
            T = self.fk(q)
            e_p = target[:3, 3] - T[:3, 3]
            if axis is None:
                e_r = rotvec(target[:3, :3] @ T[:3, :3].T)
                rot = float(np.linalg.norm(e_r))
            else:
                a_t, a_c = target[:3, axis], T[:3, axis]
                e_r = np.cross(a_c, a_t)
                rot = float(np.arccos(np.clip(a_c @ a_t, -1.0, 1.0)))
                if rot > np.pi / 2:          # cross product vanishes at 180 degrees: push along any normal
                    e_r = e_r + 1e-3 * np.cross(a_c, np.eye(3)[np.argmin(np.abs(a_c))])
            pos = float(np.linalg.norm(e_p))
            if best is None or pos + w_rot * rot < best.pos_err + w_rot * best.rot_err:
                best = IKResult(q.copy(), pos, rot, pos <= pos_tol and rot <= rot_tol)
            if best.ok:
                return best
            J = self.jacobian(T)
            Jr = J[3:]
            if axis is not None:
                P = np.eye(3) - np.outer(target[:3, axis], target[:3, axis])
                Jr, e_r = P @ Jr, P @ e_r
            A = np.vstack([J[:3], w_rot * Jr])
            e = np.concatenate([e_p, w_rot * e_r])
            dq = A.T @ np.linalg.solve(A @ A.T + damping ** 2 * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lower, self.upper)
        return best

    def seeds(self, q_now, rng: np.random.Generator, n_random: int = 12) -> Iterator[np.ndarray]:
        if q_now is not None:
            yield np.asarray(q_now, float)
        yield self.home
        for _ in range(n_random):
            yield rng.uniform(self.lower, self.upper)

    def ik(self, target: np.ndarray, q_now=None, free: str | None = None, rng: np.random.Generator | None = None,
           n_random: int = 12, **kw) -> IKResult:
        """First converged solution over the seeds (current q, home, random), else the smallest residual."""
        rng = rng or np.random.default_rng(0)
        best = None
        for s in self.seeds(q_now, rng, n_random):
            r = self.solve(target, s, free=free, **kw)
            if r.ok:
                return r
            if best is None or r.pos_err + 0.3 * r.rot_err < best.pos_err + 0.3 * best.rot_err:
                best = r
        return best

    def ik_solutions(self, target: np.ndarray, q_now=None, free: str | None = None,
                     rng: np.random.Generator | None = None, n_random: int = 12, **kw) -> Iterator[IKResult]:
        """Distinct converged solutions, in seed order (current q, home, random)."""
        rng = rng or np.random.default_rng(0)
        found: list[np.ndarray] = []
        for s in self.seeds(q_now, rng, n_random):
            r = self.solve(target, s, free=free, **kw)
            if r.ok and all(np.max(np.abs(r.q - f)) > 0.05 for f in found):
                found.append(r.q)
                yield r


class Kinematics:
    """Every arm of a RobotSpec."""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        datas: dict[str, mujoco.MjData] = {}
        self.arms: dict[str, ArmModel] = {}
        for a in spec.arms:
            model = load_model(a.model)
            key = resolve_model_path(a.model)
            if key not in datas:
                datas[key] = mujoco.MjData(model)
            self.arms[a.name] = ArmModel(a, model, datas[key])

    def __getitem__(self, arm: str) -> ArmModel:
        if arm not in self.arms:
            raise KeyError(f"no arm {arm!r}; arms: {list(self.arms)}")
        return self.arms[arm]


__all__ = ["ArmModel", "IKResult", "Kinematics", "load_model", "resolve_model_path"]
