# hylo: worker-side (runs inside Blender's own Python; never imported by Hylo packages)
"""Blender side of the rigged characters: blender -b --factory-startup --python motion_blender.py -- rig.json

Drives one skinned character (FBX or glTF) with the studio's 17 keypoints and optionally renders it. The rig is
scaled to its mapped height and stood on the floor; the target points get the character's proportions
(motion_retarget.proportional_targets); the pelvis bone takes the hips' position and turn, spine bones share the
turn from the hips to the chest, and each limb bone is swung (shortest rotation) so the segment between its
mapped joints points like the source's; bones can be pinned to another bone's end (IK feet of the original rig).
A height offset then keeps the lowest foot point on the floor while the source feet are down. Metrics: keypoint
error against the proportional targets, limb angle error, knees bending forwards (anatomical limit hits), floor
penetration and foot sliding.

The look is the hylo-tools render tool's (render_blender.py, loaded from `render_blender` in the config): render
settings, sky, light rig, ground and camera; the camera follows the pelvis like the render tool's "track" camera
(MuJoCo azimuth / elevation convention), so rig and robot renders match.

rig.json: {"rig": motion_characters.rig_spec(...), "keypoints": .npy (T, K, 3), "names", "fps", "render": bool,
"width", "height", "samples", "threads", "camera": {"distance", "azimuth", "elevation", "fovy"}, "frames_dir",
"out_dir", "render_blender"}. Writes frames_dir/*.png and out_dir/rig_result.json ({"outputs", "files"}).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("hylo_motion_retarget", Path(__file__).resolve().with_name("motion_retarget.py"))
SIDES = ("left", "right")
SEGMENTS = {"upper_arm": ("Arm", "ForeArm"), "forearm": ("ForeArm", "Hand"), "thigh": ("Leg", "Shin"),
            "shin": ("Shin", "Foot"), "foot": ("Foot", "ToeBase")}


def srgb(c):
    return [x ** 2.2 for x in c[:3]] + [1.0]


def _np(M):
    return np.array([[M[i][j] for j in range(4)] for i in range(4)], float)


def _mat(A):
    return Matrix([list(map(float, row)) for row in A])


def _end(rest, length, bone_end):
    b, e = bone_end.rsplit(".", 1)
    M = rest[b]
    return M[:3, 3] if e == "head" else (M @ np.array([0.0, length[b], 0.0, 1.0]))[:3]


def _rot_between(u, v):
    q = Vector(u).rotation_difference(Vector(v))
    return np.array(q.to_matrix())


def _slerp(A, B, s):
    qa, qb = Matrix(A.tolist()).to_quaternion(), Matrix(B.tolist()).to_quaternion()
    return np.array(qa.slerp(qb, s).to_matrix())


def import_rig(rig: dict):
    """Import the character file; clear its animation and constraints; unlock the driven bones; apply the
    texture and colour overrides. Returns (armature, meshes)."""
    before = set(bpy.data.objects)
    p = rig["file"]
    if p.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=p)
    elif p.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=p)
    else:
        raise ValueError(f"unsupported character file {p}")
    new = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise ValueError(f"{p}: no armature")
    arm = arms[0]
    meshes = [o for o in new if o.type == "MESH"]
    for o in new:
        o.animation_data_clear()
    names = {b.name for b in arm.data.bones}
    drive = rig["drive"]
    driven = [drive["pelvis"], *drive["spine"], drive["neck"]] + [drive[f"{s}_{r}"] for s in SIDES for r in SEGMENTS]
    needed = set(driven) | {v.rsplit(".", 1)[0] for v in rig["joints"].values()} | set(rig.get("attach", {})) \
        | {v.rsplit(".", 1)[0] for v in rig.get("attach", {}).values()}
    missing = sorted(needed - names)
    if missing:
        raise ValueError(f"{rig['name']}: bones not in the rig: {missing} (bones: {sorted(names)[:60]})")
    for pb in arm.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for b in driven:
        arm.data.edit_bones[b].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in meshes:                                   # keep every mesh under the armature (scaling moves it too)
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = arm
            o.matrix_world = mw
    if rig.get("texture"):
        img = bpy.data.images.load(rig["texture"])
        for o in meshes:
            for mat in o.data.materials:
                if mat is None:
                    continue
                mat.use_nodes = True
                nt = mat.node_tree
                bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if bsdf is None:
                    continue
                tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None) or nt.nodes.new("ShaderNodeTexImage")
                tex.image = img
                tex.interpolation = "Closest"
                nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    colours = rig.get("colours") or {}
    for o in meshes:
        for mat in o.data.materials:
            if mat is not None and mat.use_nodes:
                b = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if b is not None:
                    for link in list(mat.node_tree.links):          # FBX imports often leave alpha at 0
                        if link.to_node == b and link.to_socket.name in ("Alpha", "Normal"):
                            mat.node_tree.links.remove(link)
                    b.inputs["Alpha"].default_value = 1.0
                    try:
                        mat.surface_render_method = "DITHERED"
                    except (AttributeError, TypeError):
                        pass
                    if mat.name in colours:
                        b.inputs["Base Color"].default_value = srgb(colours[mat.name])
                    b.inputs["Roughness"].default_value = 0.7
                    if "Specular IOR Level" in b.inputs:
                        b.inputs["Specular IOR Level"].default_value = 0.3
    return arm, meshes


def mesh_bounds(meshes) -> tuple[np.ndarray, np.ndarray]:
    """World bounding box of the evaluated (deformed) meshes."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        v = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3)
        M = _np(ev.matrix_world)
        w = v @ M[:3, :3].T + M[:3, 3]
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        ev.to_mesh_clear()
    return lo, hi


def drive_rig(a: dict) -> dict:
    """Import, scale and drive one rigged character from a's keypoints; returns place(t), the root path,
    the posed joints (world) and the metrics."""
    rig = a["rig"]
    arm, meshes = import_rig(rig)
    lo, hi = mesh_bounds(meshes)
    k = float(rig["height_m"]) / float(hi[2] - lo[2])
    arm.scale = arm.scale * k
    bpy.context.view_layer.update()
    lo, hi = mesh_bounds(meshes)
    arm.location = arm.location - Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    bpy.context.view_layer.update()
    mw = _np(arm.matrix_world)
    sc = float(np.cbrt(abs(np.linalg.det(mw[:3, :3]))))
    Rmw = mw[:3, :3] / sc
    bones = {b.name: b for b in arm.data.bones}
    rest = {n: _np(b.matrix_local) for n, b in bones.items()}
    length = {n: float(b.length) for n, b in bones.items()}
    parent = {n: (b.parent.name if b.parent else None) for n, b in bones.items()}
    names = list(a["names"])
    idx = {n: i for i, n in enumerate(names)}
    joints = rig["joints"]
    rest_arm = {kp: _end(rest, length, joints[kp]) for kp in names}
    rest_w = {kp: (mw @ np.r_[p, 1])[:3] for kp, p in rest_arm.items()}
    K = np.load(a["keypoints"]).astype(float)
    T = len(K)
    Tg, s = rt.proportional_targets(K, idx, rest_w)
    Ta = (Tg - mw[:3, 3]) @ np.linalg.inv(mw[:3, :3]).T          # world -> armature space
    Fp, Fc = rt.body_frames(K, idx)
    RA = np.stack([rest_arm[n] for n in names])
    Rp0, Rc0 = rt.body_frames(RA, idx)
    Dp = np.einsum("ij,tjk,lk->til", Rmw.T, Fp, Rp0)               # world frames -> armature-space turns
    Dc = np.einsum("ij,tjk,lk->til", Rmw.T, Fc, Rc0)
    drive = rig["drive"]
    attach = dict(rig.get("attach") or {})
    role = {drive["pelvis"]: ("pelvis", None), drive["neck"]: ("seg", ("Chest", "Head"))}
    n_sp = len(drive["spine"])
    for i, b in enumerate(drive["spine"]):
        role[b] = ("spine", (i + 1) / n_sp)
    for sd in SIDES:
        S = sd.capitalize()
        for r, (j0, j1) in SEGMENTS.items():
            role[drive[f"{sd}_{r}"]] = ("seg", (S + j0, S + j1))
    order, done = [], set()                          # parents (and attach targets) before children

    def visit(n):
        if n in done:
            return
        done.add(n)
        if parent[n]:
            visit(parent[n])
        if n in attach:
            visit(attach[n].rsplit(".", 1)[0])
        order.append(n)
    for n in bones:
        visit(n)
    inv_rest = {n: np.linalg.inv(M) for n, M in rest.items()}
    up_arm = np.linalg.inv(mw[:3, :3]) @ np.array([0.0, 0.0, 1.0])
    pelvis_head = rest[drive["pelvis"]][:3, 3] - rest_arm["Hips"]

    def solve(t, lift=0.0):
        pose, basis = {}, {}
        for n in order:
            p = parent[n]
            inh = pose[p] @ inv_rest[p] @ rest[n] if p else rest[n].copy()
            if n in role:
                kind, arg = role[n]
                if kind == "pelvis":
                    R = Dp[t] @ rest[n][:3, :3]
                    head = Ta[t, idx["Hips"]] + Dp[t] @ pelvis_head + lift * up_arm
                elif kind == "spine":
                    R = _slerp(Dp[t], Dc[t], arg) @ rest[n][:3, :3]
                    head = inh[:3, 3]
                else:
                    a0, a1 = arg
                    Dinh = inh[:3, :3] @ rest[n][:3, :3].T
                    u = Dinh @ rt.unit(rest_arm[a1] - rest_arm[a0])
                    v = rt.unit(Ta[t, idx[a1]] - Ta[t, idx[a0]])
                    R = _rot_between(u, v) @ inh[:3, :3]
                    head = inh[:3, 3]
                if n in attach:
                    head = _end(pose, length, attach[n])
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R, head
                basis[n] = np.linalg.inv(inh) @ M
                pose[n] = M
            else:
                pose[n] = inh
        J = np.stack([_end(pose, length, joints[kp]) for kp in names])
        return basis, (J @ mw[:3, :3].T) + mw[:3, 3]
    feet = [("LeftFoot", "LeftToeBase"), ("RightFoot", "RightToeBase")]
    h_rest = {kp: rest_w[kp][2] for f in feet for kp in f}

    def lowest(J):
        return min(J[idx[kp], 2] - h_rest[kp] for f in feet for kp in f)
    J1 = np.stack([solve(t)[1] for t in range(T)])
    low = np.array([lowest(J1[t]) for t in range(T)])
    corr = rt.ground_correction(low, rt.feet_height(Tg, idx), s)
    lift = corr                                       # up_arm already carries the 1/scale
    poses, J2 = [], []
    for t in range(T):
        b, J = solve(t, lift[t])
        poses.append(b)
        J2.append(J)
    J2 = np.stack(J2)
    err = np.linalg.norm(J1 - Tg, axis=2).mean(1)
    ang = rt.segment_angles(J1, Tg, idx)
    left = Fp[:, :, 1]                                # source pelvis left axis, world

    def bend(a0, a1, a2):
        u, v = J2[:, idx[a1]] - J2[:, idx[a0]], J2[:, idx[a2]] - J2[:, idx[a1]]
        c = np.cross(u, v)
        sgn = np.sum(c * left, 1)
        angle = np.degrees(np.arctan2(np.linalg.norm(c, axis=1), np.sum(u * v, 1)))
        return sgn, angle
    wrong = {}
    for S in ("Left", "Right"):
        sgn, angle = bend(f"{S}Leg", f"{S}Shin", f"{S}Foot")            # a knee bends with the shin going back
        wrong[f"{S.lower()}_knee"] = (sgn < 0) & (angle > 10)
    hits = {k: float(v.mean()) for k, v in wrong.items()}
    low2 = np.array([lowest(J2[t]) for t in range(T)])
    skate, n_st = 0.0, 0
    for f in feet:
        for kp in f[:1]:
            p = J2[:, idx[kp]]
            down = (p[:, 2] - h_rest[kp]) < 0.02
            both = down[1:] & down[:-1]
            if both.any():
                skate += float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)[both].sum())
                n_st += int(both.sum())
    fps = float(a.get("fps", 30))
    metrics = {"scale": round(s, 4), "mean_keypoint_err_m": round(float(err.mean()), 4),
               "max_keypoint_err_m": round(float(err.max()), 4),
               "keypoint_err_pct_height": round(100 * float(err.mean()) / float(rig["height_m"]), 2),
               "limb_angle_err_deg": round(float(ang.mean()), 2),
               "limit_hit_frac": round(float(np.any(np.stack(list(wrong.values())), 0).mean()), 3),
               "limit_hits": {k: round(v, 3) for k, v in hits.items() if v > 0},
               "limits": ("anatomical: a knee bent forwards by more than 10 degrees (the rig has no joint limits; any elbow "
                          "bend can be reached by turning the shoulder, so elbows are not counted)"),
               "ground_penetration_max_m": round(float(max(0.0, -low2.min())), 4),
               "frames_floating_5cm": int((low2 > 0.05).sum()),
               "foot_skate_mean_speed_mps": round(skate / max(n_st, 1) * fps, 3),
               "ground_correction_m": [round(float(corr.min()), 3), round(float(corr.max()), 3)],
               "height_m": float(rig["height_m"])}
    pbones = {n: arm.pose.bones[n] for n in role}

    def place(t):
        t = min(t, T - 1)
        for n, B in poses[t].items():
            pbones[n].matrix_basis = _mat(B)
    return {"place": place, "root": J2[:, idx["Hips"]], "frames": T, "metrics": metrics, "joints": J2,
            "height": float(rig["height_m"])}


# ------------------------------------------------------------------------------------------------- camera
def track_camera(root: np.ndarray, fps: float, height: float, distance: float, azimuth: float,
                 elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera positions and look-at points per frame: the smoothed pelvis path at 55% of the height, seen from
    (azimuth, elevation) in degrees at `distance` (MuJoCo's free-camera convention, like the render tool)."""
    k = int(fps * 0.5)
    tgt = root.copy()
    if len(tgt) > 2 * k + 1 and k:
        pad = np.pad(tgt, ((k, k), (0, 0)), mode="edge")
        tgt = np.stack([np.convolve(pad[:, c], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for c in range(3)], 1)
    tgt[:, 2] = 0.55 * height
    a, e = math.radians(azimuth), math.radians(elevation)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return tgt - distance * fwd, tgt


def main() -> None:
    cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    out = Path(cfg["out_dir"])
    t0 = time.time()
    rb = _load("hylo_render_blender", Path(cfg["render_blender"]))
    cam_cfg = dict(cfg.get("camera") or {})
    look = {"samples": int(cfg.get("samples", 16)), "threads": int(cfg.get("threads", 32)),
            "width": int(cfg.get("width", 640)), "height": int(cfg.get("height", 360)), "denoise": True,
            "fovy": float(cam_cfg.get("fovy", 40.0))}
    sc = rb.setup_render(look)
    rb.world(sc)
    act = drive_rig(dict(cfg, fps=cfg.get("fps", 30)))
    T = act["frames"]
    np.save(out / "joints.npy", act["joints"])
    outputs = {"metrics": act["metrics"], "frames": T, "setup_s": round(time.time() - t0, 2)}
    if cfg.get("render", True):
        root = act["root"]
        centre = [float(np.mean(root[:, 0])), float(np.mean(root[:, 1])), 0.5 * act["height"]]
        radius = max(1.0, float(np.ptp(root[:, :2], axis=0).max()) / 2 + act["height"] / 2)
        rb.ground(dict(look, centre=centre, radius=radius, floor=0.0, floor_rgba=None))
        rb.lights(dict(look, centre=centre, radius=radius))
        cam = rb.camera(look)
        pos, tgt = track_camera(root, float(cfg.get("fps", 30)), act["height"], float(cam_cfg.get("distance", 3.0)),
                                float(cam_cfg.get("azimuth", 150.0)), float(cam_cfg.get("elevation", -12.0)))
        frames = Path(cfg["frames_dir"])
        for t in range(T):
            act["place"](t)
            cam.location = Vector(pos[t])
            cam.rotation_euler = (Vector(tgt[t]) - Vector(pos[t])).to_track_quat("-Z", "Y").to_euler()
            bpy.context.view_layer.update()
            sc.render.filepath = str(frames / f"{t:05d}.png")
            bpy.ops.render.render(write_still=True)
        outputs["render_s"] = round(time.time() - t0 - outputs["setup_s"], 1)
    (out / "rig_result.json").write_text(json.dumps({"outputs": outputs, "files": {"joints": str(out / "joints.npy")}}))


if __name__ == "__main__":
    main()
