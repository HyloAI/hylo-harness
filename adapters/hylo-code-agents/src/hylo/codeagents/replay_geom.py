"""Geometry, checks, video and the simulation client of replay-to-sim (RFC-0006 part 8), all without MuJoCo.

MuJoCo runs in the isolated worker `replay_sim.py` (`uv run --script`, like the hylo-tools and motion workers), so
it never enters the agent's process. `SimClient` writes the objects' meshes, sends one request per call (build the
scene, kinematic replay, physics replay) and reads the arrays back. Here: meshes (OBJ I/O, convex hulls, boxes,
extrusion to the table), pose and size errors, event checks of the simulation, final-state success and the
side-by-side video.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .replay_track import closed_intervals, detect_events

SIM_WORKER = Path(__file__).with_name("replay_sim.py")


@dataclass
class SceneObject:
    name: str
    label: str
    parts: list[tuple[np.ndarray, np.ndarray]]          # convex collision parts (V, F) in the body frame
    visual: tuple[np.ndarray, np.ndarray] | None        # visual mesh in the body frame
    pose0: np.ndarray                                   # (4, 4) body pose in the world at frame 0
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    dynamic: bool = True
    mass: float = 0.1
    info: dict[str, Any] = field(default_factory=dict)


def write_obj(path: Path, V: np.ndarray, F: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(V, float)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(F, int)]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vs, fs = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
            fs += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(vs, float), np.asarray(fs, int)


def hull_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    h = ConvexHull(points)
    F = h.simplices.copy()
    c = points[h.vertices].mean(0)
    for i, f in enumerate(F):                          # outward winding
        n = np.cross(points[f[1]] - points[f[0]], points[f[2]] - points[f[0]])
        if n @ (points[f[0]] - c) < 0:
            F[i] = f[[0, 2, 1]]
    used = np.unique(F)
    remap = -np.ones(len(points), int)
    remap[used] = np.arange(len(used))
    return points[used], remap[F]


def box_mesh(size: Sequence[float], yaw: float, centre: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """An oriented box (world frame) as a triangle mesh."""
    hx, hy, hz = (0.5 * float(x) for x in size)
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c, s_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return hull_mesh(corners @ R.T + np.asarray(centre, float))


def column_parts(points: np.ndarray, table_z: float, cell: float = 0.015, min_pts: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """A 2.5D height field of observed points as box columns standing on the table (one convex part per cell,
    height = the cell's 95th-percentile point). Keeps the concavity of a container seen from above (a bowl's floor
    is lower than its rim), which a convex hull would fill."""
    pts = np.asarray(points, float)
    keys = np.floor(pts[:, :2] / cell).astype(np.int64)
    uniq, inv_idx, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    parts = []
    for k, (ij, n) in enumerate(zip(uniq, counts)):
        if n < min_pts:
            continue
        top = float(np.percentile(pts[inv_idx.ravel() == k, 2], 95))
        h = max(top - table_z, 0.002)
        x0, y0 = ij * cell
        parts.append(box_mesh([cell, cell, h], 0.0, [x0 + cell / 2, y0 + cell / 2, table_z + h / 2]))
    return parts


def ground_mesh(V: np.ndarray, table_z: float, max_gap: float = 0.03) -> np.ndarray:
    """Points of a mesh extended down to the table (its footprint projected onto the plane) when the mesh floats
    less than `max_gap` above it: single-view meshes miss the hidden lower part of objects resting on the table."""
    V = np.asarray(V, float)
    if V[:, 2].min() - table_z > max_gap:
        return V
    foot = V.copy()
    foot[:, 2] = table_z
    return np.concatenate([V, foot])


def pose_errors(ref: np.ndarray, est: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """Translation (m) and rotation (deg) error per frame and summarised."""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    v = np.ones(len(ref), bool) if valid is None else np.asarray(valid, bool)
    te = np.linalg.norm(ref[:, :3, 3] - est[:, :3, 3], axis=1)
    cos = np.clip((np.einsum("tij,tij->t", ref[:, :3, :3], est[:, :3, :3]) - 1) / 2, -1, 1)
    re = np.degrees(np.arccos(cos))
    tv, rv = te[v], re[v]
    if not len(tv):
        return {"frames": 0}
    return {"frames": int(v.sum()), "trans_mean_m": round(float(tv.mean()), 4), "trans_median_m": round(float(np.median(tv)), 4),
            "trans_max_m": round(float(tv.max()), 4), "trans_final_m": round(float(te[-1]), 4),
            "rot_mean_deg": round(float(rv.mean()), 2), "rot_final_deg": round(float(re[-1]), 2),
            "per_frame_trans_m": [round(float(x), 4) for x in te]}


def size_error(est_size: Sequence[float], gt_size: Sequence[float]) -> dict[str, Any]:
    """Axis-agnostic: sorted extents compared."""
    e, g = np.sort(np.asarray(est_size, float)), np.sort(np.asarray(gt_size, float))
    return {"est_m": [round(float(x), 4) for x in e], "gt_m": [round(float(x), 4) for x in g],
            "abs_err_m": [round(float(x), 4) for x in np.abs(e - g)], "max_abs_err_m": round(float(np.max(np.abs(e - g))), 4),
            "rel_err": [round(float(x), 3) for x in np.abs(e - g) / np.maximum(g, 1e-6)]}


def sim_events(time: np.ndarray, phys: dict[str, Any], table_z: float, hand_cmd: np.ndarray) -> list[dict[str, Any]]:
    """Events of the physics replay with the same detector as the recording (closed intervals from the command,
    since the simulated hand opening is a measurement of the same command), plus finger contact on grasps."""
    objs = {n: P[:, :3, 3] for n, P in phys["poses"].items()}
    ivs = closed_intervals(hand_cmd)
    ev = detect_events(time, phys["opening"], objs, table_z=table_z, intervals=ivs, closed_level=0.02)
    for e in ev:
        if e["type"] == "grasp" and e["object"] in phys["fingers"]:
            f = e["frame"]
            seg = phys["fingers"][e["object"]][f:f + 15]
            e["finger_contact"] = bool(len(seg) and seg.max() >= 2)
    return ev


def final_state(ref_final: dict[str, np.ndarray], sim_final: dict[str, np.ndarray], moved: Sequence[str],
                tol: float = 0.03) -> dict[str, Any]:
    per = {}
    for n in moved:
        if n in ref_final and n in sim_final:
            err = float(np.linalg.norm(ref_final[n][:3, 3] - sim_final[n][:3, 3]))
            per[n] = {"err_m": round(err, 4), "ok": err <= tol}
    return {"tolerance_m": tol, "objects": per, "success": bool(per) and all(v["ok"] for v in per.values())}


# ------------------------------------------------------------------------------------------------ video
def side_by_side(original: np.ndarray, sim_video: str | Path | None, out: str | Path, fps: float, *,
                 height: int = 360, max_bytes: int = 4_000_000, labels: tuple[str, str] = ("recording", "simulation"),
                 sim_frames: np.ndarray | None = None) -> dict[str, Any]:
    """original | simulation as one MP4 (H.264), re-encoded at a lower bitrate until it fits max_bytes."""
    from PIL import Image, ImageDraw

    if sim_frames is None:
        sim_frames = read_video(sim_video)
    n = len(original)
    idx = np.clip(np.round(np.linspace(0, len(sim_frames) - 1, n)).astype(int), 0, len(sim_frames) - 1)

    def fit(img: np.ndarray) -> Image.Image:
        im = Image.fromarray(np.asarray(img, np.uint8))
        return im.resize((max(2, int(round(im.width * height / im.height)) // 2 * 2), height), Image.BILINEAR)

    frames = []
    for k in range(n):
        a, b = fit(original[k]), fit(sim_frames[idx[k]])
        canvas = Image.new("RGB", (a.width + b.width, height))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        dr = ImageDraw.Draw(canvas)
        for x0, text in ((4, labels[0]), (a.width + 4, labels[1])):
            dr.rectangle([x0, 4, x0 + 7 * len(text) + 6, 18], fill=(0, 0, 0))
            dr.text((x0 + 3, 5), text, fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    return write_video(frames, out, fps, max_bytes)


def read_video(path: str | Path) -> np.ndarray:
    """(T, H, W, 3) frames of a video file."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    w, h = next(reader)["size"]
    return np.stack([np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader])


def write_video(frames: Sequence[np.ndarray], out: str | Path, fps: float, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """Frames (same size, even dimensions after padding) as one H.264 MP4, re-encoded at a lower bitrate until it
    fits max_bytes."""
    import imageio_ffmpeg

    frames = [np.asarray(f, np.uint8) for f in frames]
    H, W = frames[0].shape[:2]
    if H % 2 or W % 2:
        frames = [np.pad(f, ((0, H % 2), (0, W % 2), (0, 0))) for f in frames]
        H, W = H + H % 2, W + W % 2
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    kbps = int(min(3000, 0.85 * max_bytes * 8 / max(n / fps, 1e-3) / 1000))
    for _ in range(6):
        w = imageio_ffmpeg.write_frames(str(out), (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                        bitrate=f"{kbps}k", macro_block_size=2, output_params=["-preset", "slow"])
        w.send(None)
        for f in frames:
            w.send(np.ascontiguousarray(f))
        w.close()
        if out.stat().st_size <= max_bytes:
            break
        kbps = int(kbps * 0.7)
    return {"path": str(out), "bytes": out.stat().st_size, "frames": n, "fps": fps, "size": [W, H], "kbps": kbps}


# ------------------------------------------------------------------------------------------------ worker client
class SimClient:
    """Runs replay_sim.py in its own uv environment. One call = one request with ops run in order."""

    def __init__(self, out_dir: str | Path, timeout: float = 3600.0) -> None:
        self.out = Path(out_dir)
        self.timeout = timeout
        self.scene_info: dict[str, Any] = {}
        self._n = 0

    def call(self, ops: list[str], params: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        self._n += 1
        work = self.out / "sim" / f"call-{self._n:02d}"
        work.mkdir(parents=True, exist_ok=True)
        inputs = {}
        if arrays:
            np.savez(work / "in.npz", **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
            inputs["arrays"] = str(work / "in.npz")
        req = work / "request.json"
        req.write_text(json.dumps({"ops": ops, "params": params, "inputs": inputs, "out_dir": str(work)},
                                  default=_json))
        uv = os.environ.get("HYLO_TOOLS_UV") or shutil.which("uv") or "uv"
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
        proc = subprocess.run([uv, "run", "--quiet", "--script", str(SIM_WORKER), str(req)], text=True, env=env,
                              capture_output=True, timeout=self.timeout)
        res_path = work / "result.json"
        if not res_path.is_file():
            raise RuntimeError(f"replay_sim {ops}: no result (exit {proc.returncode})\n{proc.stderr[-3000:]}")
        res = json.loads(res_path.read_text())
        if not res.get("ok"):
            raise RuntimeError(f"replay_sim {ops}: {res.get('error')}")
        out = dict(res["outputs"])
        if (work / "out.npz").is_file():
            with np.load(work / "out.npz") as z:
                out["arrays"] = {k: z[k] for k in z.files}
        return out

    def build(self, *, table_z: float, table_extent: np.ndarray, objects: list[SceneObject], agent: str,
              arm_model: str | None = None, base: np.ndarray | None = None, camera: dict[str, Any] | None = None,
              gripper_T0: np.ndarray | None = None, image_size: tuple[int, int] = (640, 480),
              hand: dict[str, Any] | None = None) -> dict[str, Any]:
        mesh_dir = self.out / "meshes"
        objs = []
        for o in objects:
            parts = [str(write_obj(mesh_dir / f"{o.name}_part_{i:03d}.obj", V, F).resolve()) for i, (V, F) in enumerate(o.parts)]
            vis = None if o.visual is None or not len(o.visual[1]) else \
                str(write_obj(mesh_dir / f"{o.name}_visual.obj", *o.visual).resolve())
            objs.append({"name": o.name, "label": o.label, "parts": parts, "visual": vis, "pose0": o.pose0,
                         "rgba": list(o.rgba), "dynamic": bool(o.dynamic), "mass": float(o.mass)})
        cam = None
        if camera is not None:
            cam = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in camera.items()}
        params = {"scene_path": str((self.out / "scene.xml").resolve()), "table_z": float(table_z),
                  "table_extent": np.asarray(table_extent).tolist(), "objects": objs, "agent": agent,
                  "arm_model": arm_model, "base": None if base is None else np.asarray(base).tolist(), "camera": cam,
                  "gripper_T0": None if gripper_T0 is None else np.asarray(gripper_T0).tolist(),
                  "image_size": list(image_size), "hand": _plain(hand)}
        self.scene_params = params
        self.scene_info = self.call(["build"], params)["build"]
        return self.scene_info

    def replay(self, *, time: np.ndarray, q: np.ndarray | None, tcp: np.ndarray | None, opening: np.ndarray,
               hand_cmd: np.ndarray, obj_poses: dict[str, np.ndarray], held: dict[int, str] | None = None,
               ops: Sequence[str] = ("kinematic", "physics"), fingers: np.ndarray | None = None) -> dict[str, Any]:
        """Kinematic replay (recorded motion into qpos) and physics replay in one worker call. fingers: (T, n)
        finger joint targets of a multi-finger hand (agent "hand")."""
        arrays: dict[str, Any] = {"time": time, "opening": opening, "hand_cmd": hand_cmd, "q": q, "tcp": tcp,
                                  "fingers": fingers}
        for n, P in obj_poses.items():
            arrays[f"obj_{n}"] = P
        params = {**self.scene_params, "objects_names": list(obj_poses), "held": {str(k): v for k, v in (held or {}).items()}}
        res = self.call(list(ops), params, arrays)
        a = res.pop("arrays", {})
        out: dict[str, Any] = {}
        if "kinematic" in res:
            out["kinematic"] = {**res["kinematic"], "qpos": a.get("kin_qpos")}
        if "physics" in res:
            names = res["physics"]["objects"]
            out["physics"] = {"qpos": a.get("phys_qpos"), "opening": a.get("phys_opening"),
                              "poses": {n: a[f"phys_pose_{n}"] for n in names},
                              "fingers": {n: a[f"phys_fingers_{n}"] for n in names},
                              "sim_time": res["physics"]["sim_time"], "diverged": res["physics"]["diverged"]}
        return out


def _plain(o: Any) -> Any:
    """JSON-ready copy (arrays to lists)."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _json(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


__all__ = ["SIM_WORKER", "SceneObject", "SimClient", "read_video", "write_video", "box_mesh", "column_parts", "final_state", "ground_mesh", "hull_mesh",
           "pose_errors", "read_obj", "side_by_side", "sim_events", "size_error", "write_obj"]
