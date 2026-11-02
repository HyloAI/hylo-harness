# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.3", "numpy>=1.26", "pillow>=10", "imageio>=2.30", "imageio-ffmpeg>=0.5"]
# ///
"""The motion studio's MuJoCo worker: retargeting to MJCF characters, mapping validation, plausibility checks and
a physics rollout.

Runs as an isolated script (`uv run --script motion_sim.py request.json`), like the hylo-tools workers, so MuJoCo
stays out of the agent's process; tests import it by path and call `handle` directly. No Hylo imports: the shared
retargeting math (motion_retarget.py, pure numpy) is loaded from this file's directory.

request.json: {"op": "retarget" | "check" | "validate", "params": {...}, "inputs": {name: path}, "out_dir": path};
the worker writes out_dir/result.json: {"ok", "outputs", "files", "timing"} (or {"ok": false, "error"}).

A robot is described by {"xml": MJCF path, "targets": {keypoint: [body or site, [x, y, z] offset, weight]},
"floor": add a floor plane, "feet": [left foot body, right foot body], "root": keypoint at the floating base,
"frame": [left hip, right hip, chest] keypoints, "scaling": "bones" | "uniform", "clearance": {keypoint: metres},
"height_m"} (motion_characters builds it from a mapping file). Offsets add sites to the bodies (group 5, invisible).

retarget: keypoints (T, K, 3) -> qpos (T, nq). The source is cleaned (smoothed for video, feet locked while down)
  and given the robot's proportions (motion_retarget.proportional_targets; "uniform": one scale); the floating base
  takes the pelvis frame; damped least-squares IK pulls the robot's targets onto those points within joint limits,
  warm-started from the previous frame, with a weak pull to the rest pose, a lateral clearance for hands and
  elbows, heavier and flat (rest-pose sole orientation) planted feet; then the height is corrected so the lowest
  foot point is on the floor while the source's feet are down. Reports keypoint error (metres and % of height),
  limb angle error, and joint-limit hits (frames where the solver still pushed past a limit). Writes scene.xml
  (absolute mesh paths, floor) for the render tool.
validate: a mapping against its model (bodies exist, floating base, limb chains, left on +y, rest-pose order).
check: kinematic metrics on qpos (ground penetration, foot skating, joint limits, self-collision, centre of mass
  over the support polygon) and, for models with actuators, a physics rollout: the model's position servos (PD)
  track the joint trajectory from the first frame with no balance controller (and, with ankle_gain, once more
  with a simple ankle strategy); reports how long the robot stays up.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"hylo_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _sibling("motion_retarget")

CONTACT_Z = 0.02          # a foot point this close to the floor counts as support


# ------------------------------------------------------------------------------------------------- model
def load_robot(robot: dict, out_dir: Path | None = None):
    """Compile the robot (with floor and target sites); returns (model, scene xml text, target list)."""
    import mujoco
    path = Path(robot["xml"])
    spec = mujoco.MjSpec.from_file(str(path))
    spec.meshdir = str((path.parent / (spec.meshdir or ".")).resolve())
    spec.texturedir = str((path.parent / (spec.texturedir or ".")).resolve())
    if robot.get("floor"):
        g = spec.worldbody.add_geom()
        g.name, g.type, g.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [30, 30, 0.1]
        g.rgba = [0.8, 0.8, 0.78, 1]
    for kp, (name, offset, _w) in robot["targets"].items():
        if offset is not None:
            s = spec.body(name).add_site()
            s.name, s.pos, s.group = f"hylo_{kp}", list(offset), 5
    model = spec.compile()
    xml = spec.to_xml()
    targets = []
    for kp, (name, offset, w) in robot["targets"].items():
        site = f"hylo_{kp}" if offset is not None else name
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid >= 0:
            targets.append((kp, "site", sid, float(w)))
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"robot has no site or body {name!r}")
            targets.append((kp, "body", bid, float(w)))
    if out_dir is not None:
        (out_dir / "scene.xml").write_text(xml)
    return model, xml, targets


def _point(d, kind, i):
    return d.site_xpos[i] if kind == "site" else d.xpos[i]


def body_geoms(m, bodies=None):
    """Geoms that touch the world: collision geoms of the robot (all robot geoms if it has none), optionally only
    those of the given bodies and their children."""
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    coll = [g for g in robot if m.geom_contype[g] or m.geom_conaffinity[g]]
    geoms = coll or robot
    if bodies is None:
        return geoms
    keep = set()
    for b in bodies:
        keep |= {c for c in range(m.nbody) if c == b or _ancestor(m, b, c)}
    return [g for g in geoms if m.geom_bodyid[g] in keep]


def _ancestor(m, a, c):
    while c > 0:
        c = m.body_parentid[c]
        if c == a:
            return True
    return False


def geom_points(m, d, geoms):
    """Lowest-candidate points of geoms in world coordinates (sphere bottoms, capsule ends, box / mesh corners)."""
    import mujoco
    pts = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for g in geoms:
        t, R, p, s = m.geom_type[g], d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(p - [0, 0, s[0]])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            for e in (-1, 1):
                pts.append(p + e * s[1] * R[:, 2] - [0, 0, s[0]])
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        else:
            half = s if t == mujoco.mjtGeom.mjGEOM_BOX else m.geom_aabb[g, 3:]
            centre = np.zeros(3) if t == mujoco.mjtGeom.mjGEOM_BOX else m.geom_aabb[g, :3]
            pts.extend(p + (centre + signs * half) @ R.T)
    return np.array(pts).reshape(-1, 3)


def hinge_joints(m):
    return [j for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]


# ------------------------------------------------------------------------------------------------- retarget
def _mat2quat(Rm):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rm).ravel())
    return q


def rest_pose(m, d, targets):
    """Reset to the model's first keyframe (or qpos0); returns {keypoint: rest position}."""
    import mujoco
    mujoco.mj_resetData(m, d)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return {kp: _point(d, kind, i).copy() for kp, kind, i, _ in targets}


def retarget(robot: dict, K: np.ndarray, names: list[str], fps: float, out_dir: Path, *, iters: int = 40,
             damping: float = 1e-2, smooth_s: float = 0.0) -> dict:
    import mujoco
    m, _xml, targets = load_robot(robot, out_dir)
    d = mujoco.MjData(m)
    idx = {n: i for i, n in enumerate(names)}
    targets = [t for t in targets if t[0] in idx]
    rest = rest_pose(m, d, targets)
    q_rest = d.qpos.copy()
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in robot["feet"]]
    foot_up = [d.xmat[f].reshape(3, 3).T @ [0.0, 0.0, 1.0] for f in feet]     # the sole's up axis, body frame
    reg = [(int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j])) for j in hinge_joints(m)]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == 0]
    if not free:
        raise ValueError("retarget: the robot needs a floating base (free joint)")
    fa, fd = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
    dofs = np.array([i for i in range(m.nv) if not fd <= i < fd + 6])
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    hj = [j for j in hinge_joints(m) if m.jnt_limited[j]]
    root_kp = robot.get("root", "Hips")
    lh, rh, up = robot.get("frame", ["LeftLeg", "RightLeg", "Chest"])
    rk = next(k for k, t in enumerate(targets) if t[0] == root_kp)
    w = np.repeat([t[3] for t in targets], 3)
    K = rt.lock_feet(rt.smooth(K, int(round(smooth_s * fps))), idx)      # video noise, sliding feet
    scaling = robot.get("scaling", "uniform")
    if scaling == "bones" and all(n in idx for n in ("LeftArm", "RightArm", "LeftLeg", "RightLeg", "Chest", "Hips")):
        full = {n: rest.get(n) for n in idx}
        if any(v is None for v in full.values()):
            raise ValueError(f"bones scaling needs a target for every keypoint: missing "
                             f"{[n for n, v in full.items() if v is None]}")
        Tg, scale = rt.proportional_targets(K, idx, full)
    else:
        scale = float(robot.get("scale") or rt.uniform_scale(K, idx, rest))
        Tg, scaling = K * scale, "uniform"
    T = len(K)
    q = d.qpos.copy()
    qpos, err = np.zeros((T, m.nq)), np.zeros(T)
    achieved = np.zeros_like(Tg)
    jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    planted = []                                          # per foot: source foot near the ground -> keep it flat
    for side in ("Left", "Right"):
        h = np.stack([Tg[:, idx[f"{side}{n}"], 2] for n in ("Foot", "ToeBase") if f"{side}{n}" in idx], 1).min(1)
        planted.append(h < np.percentile(h, 5) + 0.03)
    wfoot = {k for k, t in enumerate(targets) if t[0].endswith(("Foot", "ToeBase"))}
    clearance = robot.get("clearance", {})
    chest_h = rest[up][2] - rest[root_kp][2] if up in rest else 0.3
    at_limit = np.zeros((T, len(hj)), bool)
    clipped = np.zeros(len(hj), bool)
    for t in range(T):
        tgt = np.stack([Tg[t, idx[kp]] for kp, *_ in targets])
        y = Tg[t, idx[lh]] - Tg[t, idx[rh]]
        y /= np.linalg.norm(y) + 1e-9
        z = Tg[t, idx[up]] - Tg[t, idx[root_kp]]
        z -= y * (z @ y)
        z /= np.linalg.norm(z) + 1e-9
        q[fa + 3:fa + 7] = _mat2quat(np.stack([np.cross(y, z), y, z], 1))
        for k, (kp, *_r) in enumerate(targets):            # arms clear of the hips: a lateral minimum below the chest
            c = clearance.get(kp)
            v = tgt[k] - tgt[rk]
            if c and v @ z < chest_h:
                side = 1.0 if kp.startswith("Left") else -1.0
                lat = side * (v @ y)
                if lat < c:
                    tgt[k] += side * (c - lat) * y
        for it in range(iters * (3 if t == 0 else 1)):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            if it == 0:                                    # the base: root keypoint onto its target
                q[fa:fa + 3] += tgt[rk] - _point(d, targets[rk][1], targets[rk][2])
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, res = [], []
            for k, (_kp, kind, i, _w) in enumerate(targets):
                (mujoco.mj_jacSite if kind == "site" else mujoco.mj_jacBody)(m, d, jac, None, i)
                rows.append(jac.copy())
                res.append(tgt[k] - _point(d, kind, i))
            wt = list(w)
            for k in wfoot:                                 # planted feet weigh more
                side = 0 if targets[k][0].startswith("Left") else 1
                if planted[side][t]:
                    wt[3 * k:3 * k + 3] = [3 * w[3 * k]] * 3
            for f, pl, fu in zip(feet, planted, foot_up):   # sole flat: the foot's rest up axis towards world z
                mujoco.mj_jacBody(m, d, jac, jacr, f)
                rows.append(jacr.copy())
                res.append(np.cross(d.xmat[f].reshape(3, 3) @ fu, [0.0, 0.0, 1.0]))
                wt += [2.0 if pl[t] else 0.3] * 3
            for dof, adr in reg:                            # a weak pull to the rest pose: no drift in free twists
                row = np.zeros((1, m.nv))
                row[0, dof] = 1.0
                rows.append(row)
                res.append([q_rest[adr] - q[adr]])
                wt.append(0.01)
            wt = np.array(wt)
            A = (np.concatenate(rows) * wt[:, None])[:, dofs]
            b = np.concatenate([np.ravel(r) for r in res]) * wt
            dq = np.zeros(m.nv)
            dq[dofs] = np.linalg.solve(A.T @ A + damping * np.eye(len(dofs)), A.T @ b)
            mujoco.mj_integratePos(m, q, dq, 1.0)
            clipped = np.zeros(len(hj), bool)
            for n, j in enumerate(hj):
                a = m.jnt_qposadr[j]
                clipped[n] = q[a] < lo[j] - 1e-4 or q[a] > hi[j] + 1e-4
                q[a] = np.clip(q[a], lo[j], hi[j])
            if np.linalg.norm(dq) < 1e-5:
                break
        at_limit[t] = clipped                             # the solver still pushed past these limits at the end
        qpos[t] = q
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pts = np.stack([_point(d, kind, i) for _kp, kind, i, _w in targets])
        achieved[t, [idx[kp] for kp, *_ in targets]] = pts
        err[t] = float(np.mean(np.linalg.norm(pts - tgt, axis=1)))
    # ground: the lowest foot point on the floor whenever the source feet are near the ground
    fg = body_geoms(m, feet)
    low = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        low[t] = geom_points(m, d, fg)[:, 2].min()
    corr = rt.ground_correction(low, rt.feet_height(Tg, idx), scale)
    qpos[:, fa + 2] += corr
    np.save(out_dir / "qpos.npy", qpos)
    np.save(out_dir / "targets.npy", Tg)
    have = [idx[kp] for kp, *_ in targets]
    sub = {n: i for i, n in enumerate([kp for kp, *_ in targets])}
    ang = rt.segment_angles(achieved[:, have], Tg[:, have], sub)
    hits = at_limit.mean(0)
    height = float(robot.get("height_m") or 1.7)
    return {"outputs": {"frames": T, "fps": fps, "nq": int(m.nq), "scale": round(scale, 4), "scaling": scaling,
                        "mean_keypoint_err_m": round(float(err.mean()), 4),
                        "max_keypoint_err_m": round(float(err.max()), 4),
                        "keypoint_err_pct_height": round(100 * float(err.mean()) / height, 2),
                        "limb_angle_err_deg": round(float(ang.mean()), 2),
                        "limit_hit_frac": round(float(at_limit.any(1).mean()), 3) if hj else 0.0,
                        "limit_hits": {m.joint(hj[n]).name: round(float(hits[n]), 3)
                                       for n in np.argsort(-hits)[:5] if hits[n] > 0},
                        "ground_correction_m": [round(float(corr.min()), 3), round(float(corr.max()), 3)],
                        "joint_names": [m.joint(j).name for j in hinge_joints(m)]},
            "files": {"qpos": str(out_dir / "qpos.npy"), "scene": str(out_dir / "scene.xml"),
                      "targets": str(out_dir / "targets.npy")}}


def validate(robot: dict) -> dict:
    """Check a character mapping against its model: every body exists (load_robot raises otherwise), a floating
    base, the feet are bodies at the bottom of the leg chains, limbs are chains (hip above knee above ankle in the
    kinematic tree and in the rest pose), left is +y. Returns the rest-pose points and the problems found."""
    import mujoco
    m, _xml, targets = load_robot(robot, None)
    d = mujoco.MjData(m)
    rest = rest_pose(m, d, targets)
    body_of = {kp: (m.site_bodyid[i] if kind == "site" else i) for kp, kind, i, _w in targets}
    problems = []
    if not any(m.jnt_type[j] == 0 for j in range(m.njnt)):
        problems.append("the model has no floating base (free joint)")
    for s in ("Left", "Right"):
        for chain in (("Leg", "Shin", "Foot"), ("Arm", "ForeArm", "Hand")):
            for a, b in zip(chain, chain[1:]):
                ba, bb = body_of.get(s + a), body_of.get(s + b)
                if ba is not None and bb is not None and not (ba == bb or _ancestor(m, ba, bb)):
                    problems.append(f"{s}{a} ({m.body(ba).name}) is not above {s}{b} ({m.body(bb).name}) in the tree")
    for f, s in zip(robot["feet"], ("Left", "Right")):
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f)
        if fb < 0:
            problems.append(f"foot body {f!r} not found")
        elif body_of.get(f"{s}Shin") is not None and not _ancestor(m, body_of[f"{s}Shin"], fb):
            problems.append(f"foot {f} is not below the {s.lower()} knee")
    P = {k: v for k, v in rest.items()}
    for r in ("Leg", "Shin", "Foot", "ToeBase", "Arm", "ForeArm", "Hand"):
        if P[f"Left{r}"][1] <= P[f"Right{r}"][1]:
            problems.append(f"Left{r} is not on the left (+y) of Right{r}")
        if abs(P[f"Left{r}"][1] + P[f"Right{r}"][1]) > 0.03 or abs(P[f"Left{r}"][2] - P[f"Right{r}"][2]) > 0.03:
            problems.append(f"Left{r} and Right{r} are not mirror images ({np.round(P[f'Left{r}'], 3)} vs "
                            f"{np.round(P[f'Right{r}'], 3)})")
    for s in ("Left", "Right"):
        if not P[f"{s}Leg"][2] > P[f"{s}Shin"][2] > P[f"{s}Foot"][2]:
            problems.append(f"{s} hip, knee and ankle are not ordered top to bottom in the rest pose")
        if P[f"{s}ToeBase"][0] <= P[f"{s}Foot"][0]:
            problems.append(f"{s} toe is not in front of the ankle (the model should face +x)")
    if not P["Head"][2] > P["Chest"][2] > P["Hips"][2]:
        problems.append("pelvis, chest and head are not ordered bottom to top")
    fg = body_geoms(m, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in robot["feet"]])
    sole = float(geom_points(m, d, fg)[:, 2].min()) if fg else float("nan")
    top = float(geom_points(m, d, body_geoms(m))[:, 2].max())
    return {"outputs": {"problems": problems, "rest": {k: np.round(v, 4).tolist() for k, v in P.items()},
                        "height_m": round(top - sole, 3), "nq": int(m.nq), "nu": int(m.nu),
                        "hinges": len(hinge_joints(m))}, "files": {}}


# ------------------------------------------------------------------------------------------------- check
def _hull(P):
    P = sorted(set(map(tuple, np.round(P, 5))))
    if len(P) < 3:
        return np.array(P)

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    return np.array(half(P)[:-1] + half(P[::-1])[:-1])


def outside_distance(c, P):
    """How far point c (2D) lies outside the convex hull of P (<= 0 inside)."""
    H = _hull(P)
    if len(H) == 0:
        return float("inf")
    if len(H) < 3:
        if len(H) == 1:
            return float(np.linalg.norm(c - H[0]))
        a, b = H
        t = np.clip((c - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1)
        return float(np.linalg.norm(c - (a + t * (b - a))))
    out = -np.inf
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        e = b - a
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)   # outward for a counter-clockwise hull
        out = max(out, float((c - a) @ n))
    return out


def kinematic_check(m, qpos: np.ndarray, fps: float, feet_names: list[str]) -> dict:
    import mujoco
    d = mujoco.MjData(m)
    T = len(qpos)
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in feet_names]
    allg = body_geoms(m)
    fg = [body_geoms(m, [f]) for f in feet]
    robot_geoms = set(allg)
    # self-collision: pairs of robot geoms penetrating > 1 cm that are not already touching in the first frame
    m2 = m
    if not any(m.geom_contype[g] for g in allg):         # a character without collision geoms: robot vs robot only
        import copy
        m2 = copy.deepcopy(m)
        m2.geom_contype[allg] = 2
        m2.geom_conaffinity[allg] = 2

    def pairs_now():
        return {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]
                if c.geom1 in robot_geoms and c.geom2 in robot_geoms and c.dist < -0.01}
    mujoco.mj_resetData(m2, d)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m2, d, 0)
    mujoco.mj_forward(m2, d)
    base_pairs = pairs_now()                              # touching in the rest pose: part of the design
    low, com_out, flight, skate, pen, selfc = np.zeros(T), [], 0, [0.0, 0.0], 0.0, 0
    contact_prev = [None, None]
    sat = np.zeros(m.njnt)
    viol = np.zeros(m.njnt)
    hj = [j for j in hinge_joints(m) if m.jnt_limited[j]]
    skate_frames = [0, 0]
    selfc_pairs: dict[str, int] = {}
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m2, d)
        pts = geom_points(m, d, allg)
        low[t] = pts[:, 2].min()
        pen = max(pen, -low[t])
        support = []
        for s in (0, 1):
            fp = geom_points(m, d, fg[s])
            if fp[:, 2].min() < CONTACT_Z:
                support.append(fp[fp[:, 2] < CONTACT_Z, :2])
            prev, contact_prev[s] = contact_prev[s], fp
            planted = (fp[:, 2] < 0.01) & (prev[:, 2] < 0.01) if prev is not None else np.zeros(0, bool)
            if planted.any():          # the least-moving planted point: rolling over heel or toe is not skating
                skate[s] += float(np.linalg.norm(fp[planted, :2] - prev[planted, :2], axis=1).min())
                skate_frames[s] += 1
        if support:
            com = d.subtree_com[1, :2]
            com_out.append(outside_distance(com, np.concatenate(support)))
        else:
            flight += 1
        new = pairs_now() - base_pairs
        selfc += bool(new)
        for g1, g2 in new:
            key = "/".join(sorted({m.body(m.geom_bodyid[g1]).name, m.body(m.geom_bodyid[g2]).name}))
            selfc_pairs[key] = selfc_pairs.get(key, 0) + 1
        for j in hj:
            a = m.jnt_qposadr[j]
            lo, hi = m.jnt_range[j]
            viol[j] += qpos[t, a] < lo - 1e-3 or qpos[t, a] > hi + 1e-3
            sat[j] += min(qpos[t, a] - lo, hi - qpos[t, a]) < 0.02
    com_out = np.array(com_out) if com_out else np.zeros(0)
    stance = max(1, sum(skate_frames))
    top_sat = sorted(((sat[j] / T, m.joint(j).name) for j in hj if sat[j]), reverse=True)[:4]
    return {
        "ground_penetration_max_m": round(float(pen), 4),
        "frames_below_floor_1cm": int((low < -0.01).sum()),
        "frames_floating_5cm": int((low > 0.05).sum()),
        "flight_frames": int(flight),
        "foot_skate_m": [round(s, 3) for s in skate],
        "foot_skate_mean_speed_mps": round(sum(skate) / stance * fps, 3),
        "stance_frames": skate_frames,
        "joint_limit_violation_frames": int(viol.max()) if hj else 0,
        "joints_at_limit": [[n, round(float(f), 2)] for f, n in top_sat],
        "self_collision_frames": int(selfc),
        "self_collision_pairs": dict(sorted(selfc_pairs.items(), key=lambda kv: -kv[1])[:3]),
        "com_outside_support_frac": round(float((com_out > 0.03).mean()), 3) if len(com_out) else None,
        "com_outside_support_max_m": round(float(com_out.max()), 3) if len(com_out) else None,
    }


def rollout(m, qpos: np.ndarray, fps: float, feet_names: list[str], max_s: float = 20.0, ankle_gain: float = 0.0,
            after_s: float = 1.0):
    """Track the reference with the model's position actuators from the first frame. ankle_gain = 0: no balance
    control at all; > 0: an ankle strategy adds the pelvis tilt error (reference - simulated, pitch and roll in
    the heading frame, with rate damping) to the ankle pitch / roll targets. Simulates `after_s` past a fall for
    the video. Returns (report, simulated qpos at fps)."""
    import mujoco
    if m.nu == 0:
        return {"physics": "n/a: the model has no actuators"}, np.zeros((0, m.nq))
    d = mujoco.MjData(m)
    T = len(qpos)
    dt = m.opt.timestep
    act_q = [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)]
    names = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
    pitch = [a for a, n in enumerate(names) if "ankle_pitch" in n or "ankle_y" in n]
    roll = [a for a, n in enumerate(names) if "ankle_roll" in n or "ankle_x" in n]
    feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in feet_names}
    foot_bodies = {c for c in range(m.nbody) if c in feet or any(_ancestor(m, f, c) for f in feet)}
    d.qpos[:] = qpos[0]
    if T > 1:
        mujoco.mj_differentiatePos(m, d.qvel, 1.0 / fps, qpos[0], qpos[1])
    mujoco.mj_forward(m, d)
    dur = min((T - 1) / fps, max_s)
    steps = int(dur / dt)
    tf = np.arange(T) / fps
    ts = np.arange(steps) * dt
    ref = np.stack([np.interp(ts, tf, qpos[:, a]) for a in act_q], 1) if steps else np.zeros((0, m.nu))
    fi = np.clip(np.round(ts * fps).astype(int), 0, T - 1)
    Rref = np.zeros((T, 3, 3))
    for t in range(T):
        mujoco.mju_quat2Mat(Rref[t].reshape(9), qpos[t, 3:7])
    fell, reason, err, track, every = None, None, [], [], max(1, int(round(1.0 / (fps * dt))))
    k = 0
    while k < steps:
        d.ctrl[:] = ref[k]
        if ankle_gain and fell is None:
            R = d.xmat[1].reshape(3, 3)
            e = 0.5 * sum(np.cross(R[:, i], Rref[fi[k]][:, i]) for i in range(3))   # world tilt error
            w = R @ d.qvel[3:6]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, s_ = np.cos(yaw), np.sin(yaw)
            el = np.array([c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1]])
            wl = np.array([c * w[0] + s_ * w[1], -s_ * w[0] + c * w[1]])
            d.ctrl[pitch] -= ankle_gain * (el[1] - 0.1 * wl[1])
            d.ctrl[roll] -= ankle_gain * (el[0] - 0.1 * wl[0])
        mujoco.mj_step(m, d)
        if fell is None:
            for c_ in d.contact[:d.ncon]:
                b1, b2 = m.geom_bodyid[c_.geom1], m.geom_bodyid[c_.geom2]
                other = b2 if b1 == 0 else b1 if b2 == 0 else None
                if other is not None and other not in foot_bodies and c_.dist < 0:
                    reason = f"{m.body(other).name} touched the floor"
                    break
            if reason is None and d.qpos[2] < 0.4 * qpos[0, 2]:
                reason = "pelvis dropped below 40% of its start height"
            if reason:
                fell = k * dt
                steps = min(steps, k + int(after_s / dt))
            else:
                err.append(float(np.abs(d.qpos[act_q] - ref[k]).mean()))
        if k % every == 0:
            track.append(d.qpos.copy())
        k += 1
    up = fell if fell is not None else steps * dt
    i_end = min(T - 1, int(round(up * fps)))
    control = f"ankle strategy, gain {ankle_gain:g}" if ankle_gain else "no balance controller"
    return {"physics": f"position servos (PD) track the joint trajectory; {control}", "upright_s": round(up, 2),
            "clip_s": round(dur, 2), "upright_frac": round(up / dur, 3) if dur else 1.0, "fell": fell is not None,
            "fall_reason": reason, "joint_track_err_rad": round(float(np.mean(err)), 4) if err else None,
            "root_drift_m": round(float(np.linalg.norm(d.qpos[:2] - qpos[i_end, :2])), 3)}, np.array(track)


def check(robot: dict, qpos: np.ndarray, fps: float, out_dir: Path, physics: bool = True,
          ankle_gain: float = 0.0) -> dict:
    m, _xml, _t = load_robot(robot, None)
    rep = kinematic_check(m, qpos, fps, robot["feet"])
    files = {}
    if physics:
        res, sim = rollout(m, qpos, fps, robot["feet"])
        if len(sim):
            np.save(out_dir / "sim_qpos.npy", sim)
            files["sim_qpos"] = str(out_dir / "sim_qpos.npy")
        rep.update(res)
        if ankle_gain and m.nu:
            res2, sim2 = rollout(m, qpos, fps, robot["feet"], ankle_gain=ankle_gain)
            rep["with_ankle_strategy"] = {k: res2[k] for k in ("upright_s", "fell", "fall_reason",
                                                               "joint_track_err_rad")}
            np.save(out_dir / "sim_ankle_qpos.npy", sim2)
            files["sim_ankle_qpos"] = str(out_dir / "sim_ankle_qpos.npy")
    return {"outputs": rep, "files": files}


# ------------------------------------------------------------------------------------------------- entry
def handle(op: str, params: dict, inputs: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if op == "retarget":
        res = retarget(params["robot"], np.load(inputs["keypoints"]), list(params["names"]), float(params["fps"]),
                       out, smooth_s=float(params.get("smooth_s", 0.0)))
    elif op == "check":
        res = check(params["robot"], np.load(inputs["qpos"]), float(params["fps"]), out,
                    physics=bool(params.get("physics", True)), ankle_gain=float(params.get("ankle_gain", 0.0)))
    elif op == "validate":
        res = validate(params["robot"])
    else:
        raise ValueError(f"unknown op {op!r} (retarget, check, validate)")
    res["timing"] = {"seconds": round(time.perf_counter() - t0, 3)}
    res["ok"] = True
    return res


def main(argv: list[str]) -> int:
    req = json.loads(Path(argv[0]).read_text())
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        res = handle(req["op"], req.get("params", {}), req.get("inputs", {}), out)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        import traceback
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
    (out / "result.json").write_text(json.dumps(res, default=float))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
