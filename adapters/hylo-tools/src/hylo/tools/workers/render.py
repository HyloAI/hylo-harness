# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["mujoco>=3.3", "numpy>=1.26", "pillow>=10", "imageio>=2.30", "imageio-ffmpeg>=0.5"]
# ///
"""Offline rendering of a MuJoCo scene or trajectory to MP4.

Backends:
  blender  Blender (external program, $HYLO_BLENDER) renders with Cycles on the CPU: the worker exports every visible
           geom as a triangle mesh with its material and per-frame world pose, and a camera path; the Blender-side
           script (render_blender.py) builds principled materials, an area-light rig with soft shadows, a ground plane
           and renders each frame.
  mujoco   MuJoCo's offscreen renderer (OSMesa or EGL) with a better look than the default: a three-light rig with
           shadows, a 4096 shadow map, 8x MSAA, 2x supersampling, a sky gradient and a floor when the scene has none.
  auto     blender when available, otherwise (or when it fails) mujoco.

params: backend (auto), width (640), height (360), fps (30), samples (Cycles, 32), camera ("orbit", "fixed",
        "track" or an MJCF camera name), azimuth (deg, 135), elevation (deg, -25), orbit_degrees (90), fovy (40),
        distance (auto), lookat ([x, y, z], auto), duration (s, for a static scene: 3), simulate (s: simulate the
        scene from its initial state and render that, 0), keyframe (initial keyframe index), groups ([0, 1, 2]),
        max_frames (300), supersample (mujoco backend, 2), still_only (false)
inputs: scene (MJCF, or a compiled .mjb for the blender backend), trajectory (optional .npz: qpos (T, nq), optional time (T,) seconds or fps)
outputs: video.mp4, still.png, frames, backend, per-frame seconds, fallback reason
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from _common import load_array, main, threads, versions

HERE = Path(__file__).resolve().parent
BLENDER_DEFAULT = Path.home() / ".local" / "opt" / "blender" / "blender"
BLENDER_LIBS_DEFAULT = [Path.home() / ".local" / "blenderlibs" / "root" / "usr" / "lib" / "x86_64-linux-gnu",
                        Path.home() / ".local" / "pwlibs" / "usr" / "lib" / "x86_64-linux-gnu"]
LICENCE = {"blender": "Blender: GPL-3.0-or-later (external program, not linked); MuJoCo: Apache-2.0",
           "mujoco": "MuJoCo: Apache-2.0"}
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


# ---------------------------------------------------------------------- frames (qpos per frame)
def frame_states(m, req) -> tuple[np.ndarray, float]:
    """qpos per output frame and the fps."""
    import mujoco
    fps = float(req.get("fps", 30))
    kmax = int(req.get("max_frames", 300))
    traj = req.input("trajectory", required=False)
    if traj:
        with np.load(traj) as z:
            qpos = np.asarray(z["qpos"], float)
            t = np.asarray(z["time"], float) if "time" in z.files else None
            src_fps = float(z["fps"]) if "fps" in z.files else None
        if qpos.ndim != 2 or qpos.shape[1] != m.nq:
            raise ValueError(f"trajectory qpos has shape {qpos.shape}, the model has nq={m.nq}")
        if t is None and src_fps:
            t = np.arange(len(qpos)) / src_fps
        if t is not None:
            grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
            qpos = qpos[np.clip(np.searchsorted(t, grid), 0, len(qpos) - 1)]
        return qpos[:kmax], fps
    d = mujoco.MjData(m)
    if req.get("keyframe") is not None and m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, int(req.get("keyframe")))
    sim = float(req.get("simulate", 0.0))
    n = 1 if req.get("still_only") else max(1, int(round((sim or float(req.get("duration", 3.0))) * fps)))
    n = min(n, kmax)
    if sim <= 0:
        return np.repeat(d.qpos[None].copy(), n, 0), fps
    out, steps = [], max(1, int(round(1.0 / (fps * m.opt.timestep))))
    for _ in range(n):
        out.append(d.qpos.copy())
        for _ in range(steps):
            mujoco.mj_step(m, d)
    return np.array(out), fps


# ---------------------------------------------------------------------- geometry
def _grid_sphere(nu: int = 40, nv: int = 20, split: float = 0.0):
    """A UV sphere of radius 1 (z poles); with split > 0 the upper half is shifted by +split and the lower by -split
    (a capsule). Returns verts, faces, uvs (per vertex)."""
    th = np.linspace(0, np.pi, nv + 1)                    # polar angle from +z
    shift = np.zeros(nv + 1)
    if split:   # duplicate the equator ring: upper half up, lower half down, a straight band between
        h = nv // 2
        th = np.concatenate([th[:h + 1], th[h:]])
        shift = np.concatenate([np.full(h + 1, split), np.full(nv + 1 - h, -split)])
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T) + shift[:, None]], -1).reshape(-1, 3)
    uv = np.stack([P / (2 * np.pi), 1 - T / np.pi], -1).reshape(-1, 2)
    idx = np.arange(len(th) * (nu + 1)).reshape(len(th), nu + 1)
    a, b, c, dd = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    f = np.concatenate([np.stack([a, dd, c], -1).reshape(-1, 3), np.stack([a, c, b], -1).reshape(-1, 3)])
    # drop degenerate triangles at the poles
    keep = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 1e-12
    return v, f[keep], uv


def _cylinder(r: float, h: float, nu: int = 48):
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    ring = np.stack([r * np.cos(ph), r * np.sin(ph)], -1)
    side = np.concatenate([np.c_[ring, np.full(nu + 1, -h)], np.c_[ring, np.full(nu + 1, h)]])
    suv = np.concatenate([np.c_[ph / (2 * np.pi), np.zeros(nu + 1)], np.c_[ph / (2 * np.pi), np.ones(nu + 1)]])
    i = np.arange(nu)
    sf = np.concatenate([np.stack([i, i + 1, nu + 1 + i + 1], -1), np.stack([i, nu + 1 + i + 1, nu + 1 + i], -1)])
    verts, faces, uvs = [side], [sf], [suv]
    base = len(side)
    for z, flip in ((h, False), (-h, True)):
        cap = np.concatenate([[[0, 0, z]], np.c_[ring[:-1], np.full(nu, z)]])
        cuv = 0.5 + 0.5 * cap[:, :2] / max(r, 1e-9)
        j = np.arange(nu)
        cf = np.stack([np.zeros(nu, int), 1 + j, 1 + (j + 1) % nu], -1)
        if flip:
            cf = cf[:, ::-1]
        verts.append(cap)
        faces.append(cf + base)
        uvs.append(cuv)
        base += len(cap)
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs)


def _box(sx: float, sy: float, sz: float, rep=(1.0, 1.0), uniform: bool = False):
    s = np.array([sx, sy, sz])
    verts, faces, uvs = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            corners = []
            for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u_ax], p[v_ax] = sign * s[axis], cu * s[u_ax], cv * s[v_ax]
                corners.append(p)
            c = np.array(corners)
            n = len(verts) * 4
            f = np.array([[0, 1, 2], [0, 2, 3]]) + n
            if (np.cross(c[1] - c[0], c[2] - c[0]) @ (np.eye(3)[axis] * sign)) < 0:
                f = f[:, ::-1]
            if uniform:
                uv = np.c_[c[:, u_ax] * rep[0], c[:, v_ax] * rep[1]]
            else:
                uv = np.c_[(c[:, u_ax] / s[u_ax] + 1) / 2 * rep[0], (c[:, v_ax] / s[v_ax] + 1) / 2 * rep[1]]
            verts.append(c)
            faces.append(f)
            uvs.append(uv)
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs)


def _plane(sx: float, sy: float, rep=(1.0, 1.0), uniform: bool = False):
    v = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = v[:, :2] * np.array(rep) if uniform else (v[:, :2] / [sx, sy] + 1) / 2 * np.array(rep)
    return v, f, uv


def scene_extent(m, d, frames: np.ndarray, groups) -> tuple[np.ndarray, float, float]:
    """Centre and radius of the visible non-plane geoms over all frames, and the lowest point (floor height)."""
    import mujoco
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    sel = [g for g in range(m.ngeom) if m.geom_group[g] in groups and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
           and m.geom_rgba[g, 3] > 0]
    for q in frames[:: max(1, len(frames) // 30)]:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for g in sel:
            r = m.geom_rbound[g] if m.geom_rbound[g] > 0 else 0.05
            p = d.geom_xpos[g]
            lo, hi = np.minimum(lo, p - r), np.maximum(hi, p + r)
    if not np.isfinite(lo).all():
        return np.zeros(3), 1.0, 0.0
    planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    floor = float(min([d.geom_xpos[g][2] for g in planes] + [lo[2]]))
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2), floor


def _material(m, g):
    """rgba, roughness, metallic, emission and the 2D texture (id, repeat, uniform) of geom g."""
    rgba = np.array(m.geom_rgba[g], float)
    mat = int(m.geom_matid[g])
    rough, metal, emis, tex = 0.5, 0.0, 0.0, None
    if mat >= 0:
        if np.allclose(rgba, DEFAULT_RGBA):
            rgba = np.array(m.mat_rgba[mat], float)
        rough = float(np.clip(m.mat_roughness[mat], 0.05, 1.0)) if hasattr(m, "mat_roughness") else 0.5
        metal = float(m.mat_metallic[mat]) if hasattr(m, "mat_metallic") else 0.0
        if rough == 1.0 and m.mat_shininess[mat] > 0:      # older defaults: derive from shininess
            rough = float(np.clip(1.0 - 0.8 * m.mat_shininess[mat], 0.1, 1.0))
        if m.mat_reflectance[mat] > 0:
            rough = min(rough, 0.35)
        emis = float(m.mat_emission[mat])
        texids = np.atleast_1d(m.mat_texid[mat])
        tid = int(texids[1] if len(texids) > 1 and texids[1] >= 0 else texids[0])
        if tid >= 0 and int(m.tex_type[tid]) == 0:    # 2D texture
            tex = (tid, tuple(float(x) for x in m.mat_texrepeat[mat]), bool(m.mat_texuniform[mat]))
    return rgba, rough, metal, emis, tex


def export_geometry(m, frames: np.ndarray, cams: np.ndarray, groups, work: Path, extent, req) -> dict:
    """Meshes (geom frame), materials, per-frame geom poses and the camera path for the Blender-side script."""
    import mujoco
    from PIL import Image
    G = mujoco.mjtGeom
    d = mujoco.MjData(m)
    centre, radius, floor = extent
    arrays, geoms, textures = {}, [], {}
    has_plane = False
    for g in range(m.ngeom):
        if m.geom_group[g] not in groups or m.geom_rgba[g, 3] <= 0:
            continue
        t, s = int(m.geom_type[g]), m.geom_size[g]
        rgba, rough, metal, emis, tex = _material(m, g)
        rep, uni = (tex[1], tex[2]) if tex else ((1.0, 1.0), False)
        smooth = True
        if t == G.mjGEOM_PLANE:
            has_plane = True
            big = max(radius * 8, 5.0)
            v, f, uv = _plane(s[0] or big, s[1] or big, rep, uni)
            smooth = False
        elif t == G.mjGEOM_BOX:
            v, f, uv = _box(*s, rep, uni)
            smooth = False
        elif t == G.mjGEOM_SPHERE:
            v, f, uv = _grid_sphere()
            v = v * s[0]
        elif t == G.mjGEOM_ELLIPSOID:
            v, f, uv = _grid_sphere()
            v = v * s[:3]
        elif t == G.mjGEOM_CAPSULE:
            v, f, uv = _grid_sphere(40, 20, split=s[1] / max(s[0], 1e-9))
            v = v * s[0]
        elif t == G.mjGEOM_CYLINDER:
            v, f, uv = _cylinder(s[0], s[1])
        elif t == G.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v = np.array(m.mesh_vert[va:va + vn], float)
            f = np.array(m.mesh_face[fa:fa + fn], int)
            uv = None
            if tex is not None and m.mesh_texcoordnum[mid] > 0:   # per-corner texcoords
                ta = m.mesh_texcoordadr[mid]
                tc = np.array(m.mesh_texcoord[ta:ta + m.mesh_texcoordnum[mid]], float)
                ftc = np.array(m.mesh_facetexcoord[fa:fa + fn], int)
                v, f, uv = v[f].reshape(-1, 3), np.arange(fn * 3).reshape(-1, 3), tc[ftc].reshape(-1, 2)
                uv[:, 1] = 1 - uv[:, 1]
            smooth = "angle"
        else:
            continue   # hfield, sdf: not exported
        key = f"g{g}"
        arrays[f"v_{key}"], arrays[f"f_{key}"] = np.asarray(v, np.float32), np.asarray(f, np.int32)
        if uv is not None and tex is not None:
            arrays[f"uv_{key}"] = np.asarray(uv, np.float32)
        tex_path = None
        if tex is not None and f"uv_{key}" in arrays:
            tid = tex[0]
            if tid not in textures:
                w, h, c = int(m.tex_width[tid]), int(m.tex_height[tid]), int(m.tex_nchannel[tid])
                a = np.array(m.tex_data[m.tex_adr[tid]: m.tex_adr[tid] + w * h * c], np.uint8).reshape(h, w, c)
                p = work / f"tex{tid}.png"
                Image.fromarray(a[..., :3] if c >= 3 else a[..., 0]).save(p)
                textures[tid] = str(p)
            tex_path = textures[tid]
        geoms.append({"key": key, "name": m.geom(g).name or key, "geom": g, "rgba": [float(x) for x in rgba],
                      "roughness": rough, "metallic": metal, "emission": emis, "texture": tex_path,
                      "smooth": smooth, "plane": t == G.mjGEOM_PLANE})
    poses = np.zeros((len(frames), len(geoms), 4, 4), np.float32)
    for i, q in enumerate(frames):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for j, gm in enumerate(geoms):
            g = gm["geom"]
            poses[i, j, :3, :3] = d.geom_xmat[g].reshape(3, 3)
            poses[i, j, :3, 3] = d.geom_xpos[g]
            poses[i, j, 3, 3] = 1
    arrays["poses"], arrays["cams"] = poses, cams.astype(np.float32)
    np.savez(work / "geometry.npz", **arrays)
    return {"geoms": geoms, "has_plane": has_plane, "centre": centre.tolist(), "radius": radius, "floor": floor,
            "geometry": str(work / "geometry.npz")}


# ---------------------------------------------------------------------- camera path
def free_camera(lookat, distance, azimuth, elevation) -> np.ndarray:
    """MuJoCo free-camera parameters -> camera-to-world (camera looks along -z, +y up; same for Blender)."""
    az, el = np.radians(azimuth), np.radians(elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, up, -fwd
    T[:3, 3] = np.asarray(lookat) - distance * fwd
    return T


def camera_path(m, frames, req, extent) -> tuple[np.ndarray, list[dict] | None, float, int | None]:
    """Camera-to-world per frame, the free-camera parameters per frame (None for an MJCF camera), fovy and the MJCF
    camera id (or None)."""
    import mujoco
    centre, radius, _ = extent
    mode = str(req.get("camera", "orbit"))
    W, H = int(req.get("width", 640)), int(req.get("height", 360))
    n = len(frames)
    if mode not in ("orbit", "fixed", "track"):
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, mode)
        if cid < 0:
            raise ValueError(f"camera {mode!r}: not 'orbit', 'fixed', 'track' or a camera of the model")
        d = mujoco.MjData(m)
        cams = np.zeros((n, 4, 4))
        for i, q in enumerate(frames):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            cams[i, :3, :3] = d.cam_xmat[cid].reshape(3, 3)
            cams[i, :3, 3] = d.cam_xpos[cid]
            cams[i, 3, 3] = 1
        return cams, None, float(m.cam_fovy[cid]), cid
    fovy = float(req.get("fovy", 40.0))
    half = np.radians(fovy) / 2
    half_x = np.arctan(np.tan(half) * W / H)
    dist = float(req.get("distance") or 0.95 * radius / np.sin(min(half, half_x)))
    look = np.asarray(req.get("lookat") or centre, float)
    az0, el = float(req.get("azimuth", 135.0)), float(req.get("elevation", -25.0))
    span = float(req.get("orbit_degrees", 90.0)) if mode == "orbit" else 0.0
    looks = np.repeat(look[None], n, 0)
    if mode == "track" and req.get("lookat") is None:
        # follow the centroid of the moving geoms, smoothed
        d = mujoco.MjData(m)
        pts = []
        for q in frames:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            moving = [g for g in range(m.ngeom) if m.body_dofnum[m.geom_bodyid[g]] > 0 or m.geom_bodyid[g] > 0]
            pts.append(d.geom_xpos[moving].mean(0) if moving else look)
        pts = np.array(pts)
        k = max(1, n // 10)
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        looks = np.stack([np.convolve(np.pad(pts[:, i], k, mode="edge"), ker, "valid") for i in range(3)], 1)
    s = np.linspace(0, 1, n) if n > 1 else np.zeros(1)
    ease = 0.5 - 0.5 * np.cos(np.pi * s)        # ease in and out
    params = [{"lookat": looks[i].tolist(), "distance": dist, "azimuth": az0 + span * ease[i], "elevation": el}
              for i in range(n)]
    cams = np.stack([free_camera(p["lookat"], p["distance"], p["azimuth"], p["elevation"]) for p in params])
    return cams, params, fovy, None


# ---------------------------------------------------------------------- video
def write_video(frames_dir: Path, path: Path, fps: float) -> int:
    import imageio.v2 as imageio
    files = sorted(frames_dir.glob("*.png"))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
                            macro_block_size=8, ffmpeg_log_level="error") as w:
        for f in files:
            w.append_data(imageio.imread(f)[..., :3])
    return len(files)


# ---------------------------------------------------------------------- blender backend
def blender_exe() -> tuple[Path | None, dict[str, str]]:
    exe = Path(os.environ.get("HYLO_BLENDER") or BLENDER_DEFAULT).expanduser()
    if not exe.exists():
        found = shutil.which("blender")
        exe = Path(found) if found else None
    libs = os.environ.get("HYLO_BLENDER_LIBS")
    lib_dirs = [p for p in (libs.split(os.pathsep) if libs else map(str, BLENDER_LIBS_DEFAULT)) if p and Path(p).exists()]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + [p for p in [env.get("LD_LIBRARY_PATH")] if p])
    return exe, env


def blender_version(exe: Path, env: dict[str, str]) -> str:
    try:
        out = subprocess.run([str(exe), "-b", "--factory-startup", "--version"], env=env, capture_output=True,
                             text=True, timeout=120).stdout
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def render_blender(m, frames, cams, fovy, req, work: Path, extent, groups) -> dict:
    exe, env = blender_exe()
    if exe is None:
        raise RuntimeError("Blender not found (set HYLO_BLENDER to the blender executable)")
    scene = export_geometry(m, frames, cams, groups, work, extent, req)
    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    scene.update({"width": int(req.get("width", 640)), "height": int(req.get("height", 360)), "fovy": fovy,
                  "samples": int(req.get("samples", 32)), "frames_dir": str(frames_dir), "threads": threads(),
                  "denoise": bool(req.get("denoise", True)), "floor_rgba": req.get("floor_rgba")})
    (work / "scene.json").write_text(json.dumps(scene))
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe), "-b", "--factory-startup", "-t", str(threads()), "--python",
                           str(HERE / "render_blender.py"), "--", str(work / "scene.json")],
                          env=env, capture_output=True, text=True)
    (work / "blender.log").write_text(proc.stdout + proc.stderr)
    n = len(list(frames_dir.glob("*.png")))
    if proc.returncode != 0 or n < len(frames):
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"Blender failed (exit {proc.returncode}, {n}/{len(frames)} frames):\n{tail}")
    return {"frames_dir": frames_dir, "render_s": time.perf_counter() - t0, "version": blender_version(exe, env),
            "exe": str(exe)}


# ---------------------------------------------------------------------- mujoco backend
def styled_model(path: str, W: int, H: int, fovy: float, extent):
    """The scene with a better offscreen look: light rig with shadows, big shadow map, MSAA, sky, floor if none."""
    import mujoco
    spec = mujoco.MjSpec.from_file(path)
    centre, radius, floor = extent
    q = spec.visual.quality
    q.shadowsize, q.offsamples = max(q.shadowsize, 4096), max(q.offsamples, 8)
    gl = spec.visual.global_
    gl.offwidth, gl.offheight = max(gl.offwidth, W), max(gl.offheight, H)
    gl.fovy = fovy
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = [0.25] * 3, [0.25] * 3, [0.05] * 3
    try:
        spec.visual.map.znear = min(spec.visual.map.znear, 0.01)
    except Exception:  # noqa: BLE001
        pass
    r = max(radius, 0.2)
    c = np.asarray(centre, float)
    rig = [((1.2, -1.0, 2.2), 0.65, True), ((-1.4, -0.6, 1.4), 0.25, False), ((-0.3, 1.5, 1.8), 0.3, False)]
    for off, inten, shadow in rig:
        pos = c + np.array(off) * 2.0 * r
        light = spec.worldbody.add_light()
        light.pos = pos
        light.dir = (c - pos) / np.linalg.norm(c - pos)
        light.diffuse, light.specular = [inten] * 3, [0.3 * inten] * 3
        light.castshadow = shadow
        try:
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            light.cutoff, light.exponent = 60, 2
        except Exception:  # noqa: BLE001
            pass
    has_sky = any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures)
    if not has_sky:
        spec.add_texture(name="hylo_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.82, 0.86, 0.92],
                         rgb2=[0.42, 0.47, 0.56], width=512, height=512)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.add_texture(name="hylo_floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.86, 0.86, 0.84],
                         rgb2=[0.78, 0.78, 0.76], width=512, height=512)
        mat = spec.add_material(name="hylo_floor")
        mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "hylo_floor_tex"
        mat.texrepeat, mat.texuniform, mat.reflectance = [4, 4], True, 0.08
        fl = spec.worldbody.add_geom()
        fl.type, fl.size, fl.pos = mujoco.mjtGeom.mjGEOM_PLANE, [r * 8, r * 8, 0.05], [c[0], c[1], floor - 1e-3]
        fl.material, fl.contype, fl.conaffinity, fl.group = "hylo_floor", 0, 0, 0
    return spec.compile()


def render_mujoco(path, frames, params, fovy, cid, req, work: Path, extent) -> dict:
    import mujoco
    from PIL import Image
    W, H = int(req.get("width", 640)), int(req.get("height", 360))
    ss = max(1, int(req.get("supersample", 2)))
    m = styled_model(path, W * ss, H * ss, fovy, extent)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H * ss, W * ss)
    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    opt = mujoco.MjvOption()
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1 if g in (req.get("groups") or [0, 1, 2]) else 0
    t0 = time.perf_counter()
    for i, q in enumerate(frames):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        if cid is not None:
            cam = cid
        else:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            p = params[i]
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = p["lookat"], p["distance"], p["azimuth"], p["elevation"]
        r.update_scene(d, camera=cam, scene_option=opt)
        sc = r.scene
        for flag in ("mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_SKYBOX"):
            sc.flags[getattr(mujoco.mjtRndFlag, flag)] = 1
        img = Image.fromarray(r.render())
        if ss > 1:
            img = img.resize((W, H), Image.LANCZOS)
        img.save(frames_dir / f"{i:05d}.png")
    r.close()
    return {"frames_dir": frames_dir, "render_s": time.perf_counter() - t0,
            "gl": os.environ.get("MUJOCO_GL", "default")}


# ---------------------------------------------------------------------- handle
def handle(state, req):
    import mujoco
    path = req.input("scene")
    # a compiled model (.mjb) keeps runtime edits (colours, poses of an episode); the mujoco backend needs MJCF
    m = mujoco.MjModel.from_binary_path(path) if str(path).endswith(".mjb") else mujoco.MjModel.from_xml_path(path)
    groups = list(req.get("groups") or [0, 1, 2])
    frames, fps = frame_states(m, req)
    extent = scene_extent(m, mujoco.MjData(m), frames, groups)
    cams, params, fovy, cid = camera_path(m, frames, req, extent)
    work = req.out_dir / "work"
    work.mkdir(exist_ok=True)
    backend = str(req.get("backend", "auto"))
    fallback, info, used = None, None, None
    if backend in ("auto", "blender"):
        try:
            info, used = render_blender(m, frames, cams, fovy, req, work, extent, groups), "blender"
        except Exception as exc:  # noqa: BLE001
            if backend == "blender":
                raise
            fallback = f"{type(exc).__name__}: {exc}"
            shutil.rmtree(work / "frames", ignore_errors=True)
    if info is None:
        if cid is None and params is None:
            raise RuntimeError("no camera parameters for the mujoco backend")
        info, used = render_mujoco(path, frames, params, fovy, cid, req, work, extent), "mujoco"
    frames_dir = info["frames_dir"]
    files = sorted(frames_dir.glob("*.png"))
    still = req.path("still.png")
    shutil.copy2(files[len(files) // 2], still)
    n = 1
    if not req.get("still_only") and len(files) > 1:
        n = write_video(frames_dir, req.path("video.mp4"), fps)
    if not req.get("keep_frames"):
        shutil.rmtree(frames_dir, ignore_errors=True)
    req.provenance.update({
        "model": "Blender Cycles (CPU)" if used == "blender" else "MuJoCo offscreen renderer",
        "revision": info.get("version") or versions("mujoco").get("mujoco"),
        "licence": LICENCE[used], "backend": used, "packages": versions("mujoco", "imageio", "imageio-ffmpeg"),
        **({"blender": info["exe"]} if used == "blender" else {"gl": info.get("gl")})})
    req.timing["render_s"] = round(info["render_s"], 3)
    return {"backend": used, "fallback_reason": fallback, "frames": len(files), "video_frames": n, "fps": fps,
            "size": [int(req.get("width", 640)), int(req.get("height", 360))],
            "seconds_per_frame": round(info["render_s"] / max(1, len(files)), 3),
            "camera": str(req.get("camera", "orbit")), "fovy": fovy,
            "scene_centre": [round(float(x), 4) for x in extent[0]], "scene_radius": round(extent[1], 4)}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle))
