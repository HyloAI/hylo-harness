"""Replay to sim (RFC-0006 part 8): a recorded episode -> tracked objects and hand -> MuJoCo scene -> replay -> checks.

    hylo replay SOURCE --out DIR [--profile libero|droid|video] [--camera KEY] [--agent] [--model SPEC]
                       [--labels "red block,plate"] [--step K] [--max-frames N] [--stub] [--no-video]
                       [--hand g1|allegro|<name>]

SOURCE: a saved recording folder, panda:TASK:SEED (records one of our episodes with the privileged oracle, ground
truth kept for evaluation), hf:REPO:EPISODE (a LeRobot dataset episode; --camera names the image key), or a video
file (a human hand demonstration). See replay_data.py.

Steps (each a method of `ReplayPipeline`, callable on its own and in this order by `run()`):
  labels    a vision chat model names the objects in the first frame (fallback: the instruction's noun phrases)
  segment   SAM 3 masks for those labels (best mask per label, duplicates and table-sized masks dropped)
  lift      depth (recorded, or MoGe-2 per frame rescaled to frame 0), the camera in a +z-up frame (recorded pose,
            or GeoCalib gravity refined by the RANSAC table plane), the table plane
  hands     (human videos) hand keypoints -> metric scale from the hand's size -> parallel-gripper TCP and opening;
            with --hand also the keypoints in the world retargeted onto a multi-finger robot hand (hylo.manip.retarget:
            finger joints per frame and the palm pose; the hand's side follows the detected handedness)
  track     object poses through the video (mask propagation + ICP on the per-frame depth)
  register  (robot videos without a camera pose) scale, yaw and offset to the robot frame from the carried object's
            path and the TCP path; IK of the TCP path on the Panda (base pose fitted when unknown)
  objects   per object: mesh (mesh tool) -> convex parts (convex tool) + a support part down to the table
  grasps    (--hand) each recorded grasp refined with the dexterous grasp planner on the object's convex parts at
            the grasp moment, the plan nearest the retargeted hand chosen; the palm path is bent onto it (pre-pose,
            straight approach, then the hand-relative offset kept while carrying), the fingers open before it,
            close to the plan's squeeze command and hold until the release
  scene     MJCF with the table, the objects (moving ones free, others fixed), the camera and the Panda, a
            free-floating gripper or (--hand) a free-floating multi-finger hand
  replay    kinematic replay (recorded motion written into qpos) and physics replay (joint targets / TCP path,
            hand closed between grasp and release; objects move only by contact)
  checks    pose error per object over time (sim vs tracked, and vs ground truth when known), contact events
            (grasp, lift, place) matched to the recording, final-state success, mesh size error (ground truth)
  video     side-by-side MP4, recording | simulation, rendered with the render tool

With --agent a chat model drives the same steps from a sandboxed Python session (the code-agents sandbox and turn
loop): it sees each step's summary and overlay images, may change labels, masks, tracking mode or thresholds and
re-run steps, and calls finish() when the checks are done. Without it the steps run in order with defaults.
Writes DIR/report.json, DIR/scene.xml (+ meshes/), DIR/tracks.npz, DIR/replay.mp4 and DIR/overlays/*.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from .replay_agent import AGENT_SYSTEM, ReplayAPI, run_agent
from .replay_data import Recording, load_source, pose
from .replay_geom import (SceneObject, SimClient, column_parts, final_state, ground_mesh, hull_mesh, pose_errors, side_by_side,
                          sim_events, size_error, write_obj)
from .replay_track import (Lift, alternative_labels, anchor_grasps, apply, box_extent, carried_pairs, closed_intervals, detect_events, hand_scale,
                           hand_to_gripper, init_tracks, inv, labels_from_text, level_frame, match_events,
                           normalise_depth_scale, propose_labels, register_similarity, rotz, similarity_matrix,
                           smooth_poses, table_in_world, track_objects)

DEFAULT_MODEL: dict[str, Any] = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True,
                                 "reasoning_effort": "none", "max_tokens": 3000, "timeout_s": 900}


@dataclass
class ReplayConfig:
    n_labels: int = 4
    track_mode: str = "yaw"            # xyz | yaw | full
    gate: float = 0.04                 # m, how far observed points may lie from the predicted model
    static_eps: float = 0.015          # m: an object that never moves more is fixed in the scene
    mass: float = 0.08                 # kg per moving object
    final_tol: float = 0.03            # m
    event_tol_s: float = 1.0
    video_height: int = 360
    render_scale: float = 2.0          # simulation render size relative to the recording's frames
    max_mask_frac: float = 0.25        # masks covering more of the image are dropped (table, background)
    min_scale_path: float = 0.12       # m: a shorter carried path keeps the depth model's metric scale
    mesh_min_pixels: int = 1500        # smaller masks are upsampled before meshing (the mesh tool erodes masks)
    hand: str | None = None            # human videos: a multi-finger robot hand (hylo.manip.hands name or g1/allegro)
    hand_kp_scale: float = 1.0         # finger servo stiffness multiplier in the simulation
    refine_grasps: bool = True         # bend the retargeted motion onto planned grasps (else pure retargeting)
    grasp_ramp_s: float = 0.6          # time to blend onto the planned grasp before the grasp moment


def _overlay(rgb: np.ndarray, masks: list[tuple[str, np.ndarray]], points: dict[str, np.ndarray] | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw

    pal = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    img = np.asarray(rgb, float).copy()
    for i, (_, m) in enumerate(masks):
        img[m] = 0.45 * img[m] + 0.55 * np.array(pal[i % len(pal)])
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for i, (lab, m) in enumerate(masks):
        if m.any():
            v, u = np.nonzero(m)
            dr.text((int(u.mean()), int(v.min()) - 10), f"{i}:{lab}", fill=pal[i % len(pal)])
    for i, (name, uv) in enumerate((points or {}).items()):
        for u, v in uv:
            dr.ellipse([u - 1, v - 1, u + 1, v + 1], fill=pal[i % len(pal)])
    return np.asarray(im)


class ReplayPipeline:
    """One replay: the recording, the tools, the chat model (labels; optional) and the state of every step."""

    def __init__(self, rec: Recording, out: str | Path, tools: Any, *, model: Any = None,
                 config: ReplayConfig | None = None, log: Callable[[str], None] | None = None) -> None:
        self.rec, self.out, self.tools, self.model = rec, Path(out), tools, model
        self.cfg = config or ReplayConfig()
        self.say = log or (lambda m: None)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "overlays").mkdir(exist_ok=True)
        self.report: dict[str, Any] = {"recording": {"name": rec.name, "frames": rec.T, "fps": round(rec.fps, 3),
                                                     "instruction": rec.instruction, "agent": rec.agent,
                                                     "meta": {k: v for k, v in rec.meta.items()}},
                                       "steps": {}, "timing": {}}
        self.labels: list[str] = []
        self.masks: list[tuple[str, np.ndarray, float]] = []
        self.lift: Lift | None = None
        self.tracks: list[Any] = []
        self.tcp: np.ndarray | None = rec.tcp if rec.meta.get("frame") == "world" else None
        self.gripper = rec.gripper
        self.q = rec.q
        self.base = rec.base
        self.intervals: list[tuple[int, int]] = []
        self.objects: list[SceneObject] = []
        self.body_poses: dict[str, np.ndarray] = {}
        self.scene = None
        self.kin: dict[str, Any] | None = None
        self.phys: dict[str, Any] | None = None
        self.images: list[tuple[np.ndarray, str]] = []      # for the agent's next turn
        self.dex: dict[str, Any] | None = None               # --hand: model, retargeting, palm path, finger targets

    # -- helpers ------------------------------------------------------------------------------------------------
    def _step(self, name: str, t0: float, summary: dict[str, Any]) -> dict[str, Any]:
        self.report["steps"][name] = summary
        self.report["timing"][name] = round(time.monotonic() - t0, 2)
        self.say(f"{name}: {json.dumps(summary, default=str)[:400]}")
        return summary

    def _save_png(self, img: np.ndarray, name: str, note: str = "") -> str:
        from PIL import Image

        p = self.out / "overlays" / f"{name}.png"
        Image.fromarray(np.asarray(img, np.uint8)).save(p)
        self.images.append((np.asarray(img, np.uint8), note or name))
        return str(p)

    # -- 1. labels ----------------------------------------------------------------------------------------------
    def propose_labels(self, labels: list[str] | None = None) -> dict[str, Any]:
        t0 = time.monotonic()
        if labels:
            res = {"labels": [s.strip().lower() for s in labels if s.strip()], "by": "given"}
        elif (self.out / "labels.json").is_file():          # an earlier run's model answer (re-runs skip the call)
            res = {**json.loads((self.out / "labels.json").read_text()), "cached": True}
        else:
            who = "a human hand" if self.rec.agent == "hand" else "a robot arm"
            res = propose_labels(self.model, self.rec.rgb[0], self.rec.instruction, who=who, n=self.cfg.n_labels)
            if res.get("by") == "vlm":
                (self.out / "labels.json").write_text(json.dumps(res, default=_json))
        self.labels = res["labels"]
        return self._step("labels", t0, res)

    # -- 2. segment ---------------------------------------------------------------------------------------------
    def segment(self, labels: list[str] | None = None, keep: list[int] | None = None) -> dict[str, Any]:
        """Masks for the labels in frame 0: the best-scoring mask per label; overlapping or table-sized ones dropped.
        `keep` selects masks by index from the previous result instead of running the segmenter again."""
        t0 = time.monotonic()
        if keep is not None and self.masks:
            self.masks = [self.masks[i] for i in keep if 0 <= i < len(self.masks)]
        else:
            if labels:
                self.labels = list(labels)
            raw = list(self.tools.segment(self.rec.rgb[0], self.labels))
            found = {lab for lab, m, _ in raw if self.cfg.max_mask_frac >= m.mean() and m.sum() >= 20}
            missing = [lab for lab in self.labels if lab not in found]
            self.retried = {}
            if missing:
                # the segmenter missed some labels: other names from the vision model, at a lower threshold
                cache = self.out / "alternatives.json"
                cached = json.loads(cache.read_text()) if cache.is_file() else {}
                if all(m in cached for m in missing):
                    alts = {m: cached[m] for m in missing}
                else:
                    alts = alternative_labels(self.model, self.rec.rgb[0], missing)
                    cache.write_text(json.dumps({**cached, **alts}))
                self.retried = alts
                names = sorted({a for v in alts.values() for a in v} - set(self.labels))
                back = {a: lab for lab, v in alts.items() for a in v}
                for a, m, sc in self.tools.segment(self.rec.rgb[0], names, threshold=0.3) if names else []:
                    if back.get(a) in missing:
                        raw.append((back[a], m, sc * 0.9))
            H, W = self.rec.rgb.shape[1:3]
            best: dict[str, tuple[str, np.ndarray, float]] = {}
            for lab, m, sc in raw:
                if m.mean() > self.cfg.max_mask_frac or m.sum() < 20:
                    continue
                if lab not in best or sc > best[lab][2]:
                    best[lab] = (lab, m.astype(bool), float(sc))
            chosen: list[tuple[str, np.ndarray, float]] = []
            for cand in sorted(best.values(), key=lambda x: -x[2]):
                if all((cand[1] & c[1]).sum() / max(1, min(cand[1].sum(), c[1].sum())) < 0.5 for c in chosen):
                    chosen.append(cand)
            self.masks = chosen
            del H, W
        ov = self._save_png(_overlay(self.rec.rgb[0], [(l, m) for l, m, _ in self.masks]), "masks",
                            "frame 0 with the masks (index:label)")
        return self._step("segment", t0, {"masks": [{"index": i, "label": l, "pixels": int(m.sum()), "score": round(s, 3)}
                                                    for i, (l, m, s) in enumerate(self.masks)], "overlay": ov,
                                          "missing": [l for l in self.labels if l not in {x[0] for x in self.masks}],
                                          "retried_with": getattr(self, "retried", {})})

    # -- 3. lift ------------------------------------------------------------------------------------------------
    def lift_to_3d(self) -> dict[str, Any]:
        t0 = time.monotonic()
        rec = self.rec
        rng = np.random.default_rng(0)
        info: dict[str, Any] = {}
        if rec.depth is not None and rec.K is not None:
            depth, K, info["depth"] = rec.depth.astype(np.float32), np.asarray(rec.K, float), "recorded"
        else:
            cache = self.out / "depth_cache.npz"
            if cache.is_file():
                with np.load(cache) as z:
                    d = {"depth": z["depth"], "K": z["K"], "fov_x_deg": float(z["fov_x_deg"])}
                info["depth_cached"] = True
            if not cache.is_file() or d["depth"].shape != rec.rgb.shape[:3]:
                d = self.tools.depth(rec.rgb)
                np.savez_compressed(cache, depth=d["depth"].astype(np.float16), K=d["K"], fov_x_deg=d.get("fov_x_deg") or 0.0)
            moving = np.zeros(rec.rgb.shape[1:3], bool)
            for _, m, _ in self.masks:
                moving |= ndimage.binary_dilation(m, iterations=6)
            depth, ratios = normalise_depth_scale(d["depth"], ~moving)
            K = np.asarray(d["K"], float)
            info.update(depth="MoGe-2 per frame", per_frame_scale=[round(float(x), 3) for x in
                                                                   np.percentile(ratios, [5, 50, 95])],
                        fov_x_deg=round(float(d.get("fov_x_deg") or 0), 2))
        if rec.T_world_cam is not None:
            T = np.asarray(rec.T_world_cam, float)
            tz, ext = table_in_world(depth[0], K, T, rng, hint=rec.meta.get("table_height"))
            self.lift = Lift(depth, K, T, tz, ext, "world", info)
            info["camera"] = "recorded pose"
        else:
            g = self.tools.gravity(rec.rgb[0])
            T, tz, ext, linfo = level_frame(depth[0], K, g.get("R_level_cam"), rng)
            info.update(camera="gravity-levelled (GeoCalib) and table-refined", **linfo)
            if g.get("roll") is not None:
                info.update(roll_deg=round(math.degrees(g["roll"]), 2), pitch_deg=round(math.degrees(g["pitch"]), 2))
            self.lift = Lift(depth, K, T, tz, ext, "level", info)
        info.update(table_z=round(self.lift.table_z, 4), table_extent=np.round(self.lift.table_extent, 3).tolist(),
                    frame=self.lift.frame)
        return self._step("lift", t0, info)

    def _rescale(self, s: float, S: np.ndarray | None = None) -> None:
        """Scale the lifted geometry by s (and map the frame by the similarity S = s R | t when given)."""
        L = self.lift
        R = np.eye(3) if S is None else S[:3, :3] / s
        tvec = np.zeros(3) if S is None else S[:3, 3]
        T = L.T_frame_cam.copy()
        Tn = np.eye(4)
        Tn[:3, :3] = R @ T[:3, :3]
        Tn[:3, 3] = s * R @ T[:3, 3] + tvec
        L.depth = (L.depth * s).astype(np.float32)
        L.T_frame_cam = Tn
        corners = np.array([[L.table_extent[0, i], L.table_extent[1, j], L.table_z] for i in (0, 1) for j in (0, 1)])
        c = (s * corners @ R.T) + tvec
        L.table_extent = np.array([[c[:, 0].min(), c[:, 0].max()], [c[:, 1].min(), c[:, 1].max()]])
        L.table_z = float(s * L.table_z + tvec[2]) if S is not None else float(s * L.table_z)
        for tr in self.tracks:
            P = tr.poses.copy()
            tr.poses[:, :3, :3] = R @ P[:, :3, :3]
            tr.poses[:, :3, 3] = s * P[:, :3, 3] @ R.T + tvec
            tr.model = tr.model * s

    # -- 3b. hands ----------------------------------------------------------------------------------------------
    def hands(self) -> dict[str, Any]:
        """Human video: hand keypoints -> scale from the hand's metric size -> gripper TCP and opening in the world."""
        t0 = time.monotonic()
        L = self.lift
        h = self.tools.hands(self.rec.rgb, L.K)
        sc = hand_scale(h["kp3d_cam"], h["kp2d"], L.depth, L.K, h["valid"])
        s = float(np.clip(sc["scale"], 0.2, 5.0))
        self._rescale(s)
        L.frame = "world"
        g = hand_to_gripper(h["kp3d_cam"], h["kp2d"], h["valid"], L.depth, L.K, L.T_frame_cam)
        self.tcp, self.gripper = g["tcp"], g["gripper"]
        self.hand = {**h, **g}
        self._anchored = False
        summary = {"backend": h["backend"], "valid_frames": int(h["valid"].sum()), "frames": int(len(h["valid"])),
                   "depth_scale_from_hand": sc,
                   "aperture_m": [round(float(x), 3) for x in np.percentile(g["aperture"], [5, 50, 95])],
                   "fallback_reason": h.get("fallback_reason"), "side": h.get("side")}
        if self.cfg.hand:
            summary["retarget"] = self._retarget(h)
        return self._step("hands", t0, summary)

    def _retarget(self, h: dict[str, Any]) -> dict[str, Any]:
        """The human hand in the world -> the robot hand: finger joints and palm (TCP) pose per frame. The hand's
        model and HandSpec come from the floating-hand environment (mujoco.dex_table), its side from the human's."""
        from hylo.core.registry import registry
        from hylo.manip.handmodel import HandModel
        from hylo.manip.retarget import Retargeter

        from .replay_track import hand_keypoints_world

        L = self.lift
        side = h.get("side") or "right"
        name = self.cfg.hand if self.cfg.hand.endswith(("_left", "_right")) else f"{self.cfg.hand}_{side}"
        env = registry.build("envs", {"type": "mujoco.dex_table", "params": {"hand": name, "cameras": []}})
        arm = env.robot_spec().arms[0]
        hm = HandModel.from_arm(arm)
        kw = hand_keypoints_world(h["kp3d_cam"], h["kp2d"], h["valid"], L.depth, L.K, L.T_frame_cam)
        res = Retargeter(hm).run(kw["kp"], valid=kw["valid"], side=side)
        self.dex = {"name": name, "model": hm, "model_file": arm.model, "ret": res, "kp": kw["kp"],
                    "tcp": res.tcp.copy(), "fingers": res.q.copy(), "side": side, "grasps": []}
        return {"hand": name, "human_side": side, **res.summary()}

    # -- 4. track -----------------------------------------------------------------------------------------------
    def track(self, mode: str | None = None, gate: float | None = None) -> dict[str, Any]:
        t0 = time.monotonic()
        L = self.lift
        self.tracks = init_tracks(L, self.rec.rgb[0], [(l, m) for l, m, _ in self.masks])
        if self.gripper is not None:
            gap = int(round(0.5 * self.rec.fps)) if self.rec.agent == "hand" else 0
            self.intervals = closed_intervals(self.gripper, merge_gap=gap)
        prior = None
        if self.tcp is not None and L.frame == "world":
            tcp = self.tcp
            near: dict[int, str] = {}
            for a, b in self.intervals:
                d = {tr.name: float(np.linalg.norm(tr.poses[0][:3, 3] - tcp[a][:3, 3])) for tr in self.tracks}
                if d and min(d.values()) < 0.25:
                    near[a] = min(d, key=d.get)

            def prior(name: str, t: int) -> np.ndarray | None:
                for a, b in self.intervals:
                    if a < t < b and near.get(a) == name:
                        tr = next(x for x in self.tracks if x.name == name)
                        if np.linalg.norm(tr.poses[t - 1][:3, 3] - tcp[t - 1][:3, 3]) < 0.08:
                            return tcp[t] @ inv(tcp[t - 1]) @ tr.poses[t - 1]
                return None
        track_objects(L, self.rec.rgb, self.tracks, mode=mode or self.cfg.track_mode, prior=prior,
                      gate=gate or self.cfg.gate, fps=self.rec.fps, log=self.say)
        for tr in self.tracks:
            tr.poses = smooth_poses(tr.poses, tr.lost)
        self._track_overlay()
        return self._step("track", t0, {"objects": [tr.summary() for tr in self.tracks], "frame": L.frame,
                                        "closed_intervals": self.intervals, "mode": mode or self.cfg.track_mode})

    def _track_overlay(self) -> None:
        from .replay_track import project
        L = self.lift
        frames = np.linspace(0, self.rec.T - 1, 4).astype(int)
        tiles = []
        for f in frames:
            pts = {}
            for tr in self.tracks:
                cam = apply(inv(L.T_frame_cam), apply(tr.poses[f], tr.model[:: max(1, len(tr.model) // 150)]))
                pts[tr.name] = project(cam, L.K)[0]
            tiles.append(_overlay(self.rec.rgb[f], [], pts))
        self._save_png(np.concatenate(tiles, 1), "tracks", "tracked models projected at 4 frames")

    # -- 5. register --------------------------------------------------------------------------------------------
    def register(self) -> dict[str, Any]:
        """Robot videos without a camera pose: similarity from F to the robot frame (carried object vs TCP)."""
        t0 = time.monotonic()
        L = self.lift
        res: dict[str, Any] = {"frame_before": L.frame}
        if L.frame == "level" and self.rec.tcp is not None:
            obj, tcp, names = carried_pairs(self.tracks, self.rec.tcp, self.intervals)
            ok_track = len(obj) >= 5 and np.ptp(obj, axis=0).max() > 0.25 * np.ptp(tcp, axis=0).max()
            if len(obj) >= 5 and not ok_track:
                res["rejected"] = (f"the carried object ({names[0] if names else '?'}) moved "
                                   f"{np.ptp(obj, axis=0).max():.3f} (frame units) while the TCP moved "
                                   f"{np.ptp(tcp, axis=0).max():.3f} m: its track did not follow the gripper")
            if ok_track and np.ptp(tcp, axis=0).max() > 0.05:
                reg = register_similarity(obj, tcp)
                res.update(method="carried object path vs TCP path", objects=names,
                           path_extent_m=round(float(np.ptp(tcp, axis=0).max()), 3))
                if np.ptp(tcp, axis=0).max() < self.cfg.min_scale_path:
                    # too short a path to tell scale from noise: keep the depth model's metric scale
                    res["fitted_scale"] = round(reg["scale"], 4)
                    reg = register_similarity(obj, tcp, scale=False)
                    res["scale_from"] = "depth model (carried path shorter than min_scale_path)"
            else:
                # fallback: unit scale, no yaw, the first grasp's TCP onto the nearest object
                a = self.intervals[0][0] if self.intervals else 0
                tr = min(self.tracks, key=lambda x: x.poses[a][2, 3]) if self.tracks else None
                t = self.rec.tcp[a][:3, 3] - (tr.poses[a][:3, 3] if tr else np.zeros(3))
                reg = {"scale": 1.0, "yaw": 0.0, "t": t, "rmse": None, "pairs": 0}
                res.update(method="fallback: no carried object; unit scale, TCP at the first grasp onto the object")
            if res.get("method", "").startswith("carried") and self.intervals:
                # anchor at the first grasp: the object centre where the TCP closed on it (the path fit spreads
                # its error over the whole carry; the grasp moment is where an error breaks the physics)
                tr = next(x for x in self.tracks if x.name == names[0])
                pos = tr.positions()
                a = max(self.intervals, key=lambda iv: float(np.ptp(pos[iv[0]:iv[1]], axis=0).max()))[0]
                S0 = similarity_matrix(reg)
                k = min(range(a, min(a + 4, self.rec.T)), key=lambda i: tr.lost[i])
                shift = self.rec.tcp[k][:3, 3] - apply(S0, tr.poses[k][:3, 3])[0]
                reg["t"] = reg["t"] + shift
                res["grasp_anchor_shift_m"] = np.round(shift, 4).tolist()
            S = similarity_matrix(reg)
            self._pre_reg = (self.lift.depth[0].copy(), self.lift.T_frame_cam.copy(), S)
            self._rescale(reg["scale"], S)
            L.frame = "world"
            self.tcp = self.rec.tcp
            res.update(scale=round(reg["scale"], 4), yaw_deg=round(math.degrees(reg["yaw"]), 2),
                       t=np.round(reg["t"], 4).tolist(), rmse_m=None if reg.get("rmse") is None else round(reg["rmse"], 4),
                       pairs=reg["pairs"], table_z=round(L.table_z, 4))
        elif L.frame == "level":
            res["method"] = "none (no robot data): the levelled frame is the world"
            L.frame = "world"
        else:
            res["method"] = "none needed (camera pose known)"
        if self.rec.agent == "panda" and self.q is None and self.tcp is not None:
            res["ik"] = self._ik()
        return self._step("register", t0, res)

    def _robot_spec(self) -> Any:
        from hylo.core.registry import registry

        env = registry.build("envs", {"type": "mujoco.panda_table", "params": {"cameras": []}})
        return env.spec().robot

    def _ik(self) -> dict[str, Any]:
        from hylo.core.robot import RobotSpec
        from hylo.manip.kinematics import Kinematics

        robot = self._robot_spec()
        arm = robot.arms[0]
        tcp = self.tcp
        sample = np.linspace(0, len(tcp) - 1, min(12, len(tcp))).astype(int)

        def kin_for(base: np.ndarray) -> Any:
            spec = RobotSpec(arms=(arm.model_copy(update={"base_in_world": base.tolist()}),), cameras=(),
                             table_height=robot.table_height, notes="")
            return Kinematics(spec)["arm"]

        cands = []
        b0 = self.base if self.base is not None else np.eye(4)
        cands.append(b0)
        zmin, xmed = float(tcp[:, 2, 3].min()), float(np.median(tcp[:, 0, 3]))
        for bz in (zmin - 0.02, zmin - 0.12, zmin - 0.22):
            for bx in (xmed - 0.45, xmed - 0.55, xmed - 0.65):
                cands.append(pose(None, [bx, float(np.median(tcp[:, 1, 3])) * 0.0, bz]))
        scored = []
        for b in cands:
            km = kin_for(b)
            q, err = km.home.copy(), 0.0
            for i in sample:
                r = km.ik(tcp[i], q, n_random=4)
                q, err = r.q, err + r.pos_err + 0.1 * r.rot_err
            scored.append((err / len(sample), b))
        best = min(scored, key=lambda x: x[0])
        if scored[0][0] <= best[0] + 1e-3:
            best = scored[0]
        self.base = best[1]
        km = kin_for(self.base)
        qs, pe, re_ = [], [], []
        q = km.home.copy()
        for t in range(len(tcp)):
            r = km.ik(tcp[t], q, n_random=6)
            q = r.q
            qs.append(q)
            pe.append(r.pos_err)
            re_.append(r.rot_err)
        self.q = np.asarray(qs)
        self.arm_model = arm.model
        return {"base": np.round(self.base[:3, 3], 3).tolist(), "base_from": "profile" if best is scored[0] else "fitted",
                "pos_err_mean_m": round(float(np.mean(pe)), 4), "pos_err_max_m": round(float(np.max(pe)), 4),
                "rot_err_mean_deg": round(float(np.degrees(np.mean(re_))), 2)}

    # -- 6. events ----------------------------------------------------------------------------------------------
    def events(self) -> dict[str, Any]:
        t0 = time.monotonic()
        objs = {tr.name: tr.positions() for tr in self.tracks}
        self.ref_events = detect_events(self.rec.time, self.gripper, objs, table_z=self.lift.table_z,
                                        intervals=self.intervals)
        out: dict[str, Any] = {"events": self.ref_events}
        if self.rec.agent == "hand" and self.tcp is not None and not getattr(self, "_anchored", False):
            self.tcp, out["grasp_anchors"] = anchor_grasps(self.tcp, objs, self.ref_events, self.intervals)
            self._anchored = True
        if self.rec.gt is not None:
            self.gt_match = self._gt_match()
            gobjs = {self.gt_match.get(n, n): o["poses"][:, :3, 3] for n, o in self.rec.gt["objects"].items()}
            gobjs = {n: p for n, p in gobjs.items() if n in objs}
            out["gt_events"] = detect_events(self.rec.time, self.gripper, gobjs, table_z=self.lift.table_z,
                                             intervals=self.intervals)
        return self._step("events", t0, out)

    def _gt_match(self) -> dict[str, str]:
        """Ground-truth object name -> track name (nearest at frame 0, within 6 cm)."""
        out = {}
        for gname, o in self.rec.gt["objects"].items():
            p = o["poses"][0][:3, 3]
            best = min(self.tracks, key=lambda tr: np.linalg.norm(tr.poses[0][:3, 3] - p), default=None)
            if best is not None and np.linalg.norm(best.poses[0][:2, 3] - p[:2]) < 0.06:
                out[gname] = best.name
        return out

    # -- 7. objects ---------------------------------------------------------------------------------------------
    def build_objects(self) -> dict[str, Any]:
        t0 = time.monotonic()
        L = self.lift
        grasped = {e["object"] for e in getattr(self, "ref_events", []) if e["type"] == "grasp"}
        self.objects, self.body_poses, per = [], {}, []
        mesh_dir = self.out / "meshes_in"
        mesh_dir.mkdir(exist_ok=True)
        masks = {tr.name: tr.mask0 for tr in self.tracks}
        if hasattr(self.tools, "table_z"):
            self.tools.table_z = L.table_z
        for tr in self.tracks:
            info: dict[str, Any] = {"name": tr.name, "label": tr.label}
            # net displacement (start vs end, medians over a few frames): tracking noise and occlusions do not
            # make a fixture move; a grasped or pushed object does
            p = tr.positions()
            k = max(1, min(8, len(p) // 10))
            moved = float(np.linalg.norm(np.median(p[-k:], 0) - np.median(p[:k], 0)))
            eps = self.cfg.static_eps * (1.0 if L.info.get("depth") == "recorded" else 2.0)   # monocular: noisier
            dynamic = moved > eps or tr.name in grasped
            info["net_motion_m"] = round(moved, 4)
            try:
                rgb0, m0, d0, K0 = self.rec.rgb[0], masks[tr.name], L.depth[0], L.K
                f = int(min(4, math.ceil(math.sqrt(self.cfg.mesh_min_pixels / max(1, m0.sum())))))
                if f > 1:               # small in the image: upsample (nearest for mask and depth, K scaled)
                    rgb0, m0, d0 = (np.repeat(np.repeat(a, f, 0), f, 1) for a in (rgb0, m0, d0))
                    K0 = K0.copy()
                    K0[:2, :2] *= f
                    K0[:2, 2] = (K0[:2, 2] + 0.5) * f - 0.5
                    info["mesh_upsampled"] = f
                pre = getattr(self, "_pre_reg", None)
                if pre is not None:
                    # mesh at the depth model's own scale (the mesh tool's thresholds are metric), then map it
                    # into the world with the registration's similarity
                    d_pre = pre[0] if f == 1 else np.repeat(np.repeat(pre[0], f, 0), f, 1)
                    mres = self.tools.mesh(rgb0, m0, d_pre, K0, pre[1], tr.name, mesh_dir)
                    mres["V"] = apply(pre[2], mres["V"])
                    mres["obj"] = str(write_obj(mesh_dir / f"{tr.name}_world.obj", mres["V"], mres["F"]))
                    info["mesh_frame"] = "pre-registration, mapped by the similarity"
                else:
                    mres = self.tools.mesh(rgb0, m0, d0, K0, L.T_frame_cam, tr.name, mesh_dir)
                Vw, src = mres["V"], mres["obj"]
                info.update(mesh_backend=mres["backend"], mesh_fallback=mres.get("fallback_reason"),
                            mesh_extents_m=[round(float(x), 4) for x in np.ptp(Vw, axis=0)])
                if dynamic:
                    # a moving object needs a closed, supported collision body: a single view sees only its top
                    # and front, so the mesh is extruded down to the table (2.5D: hidden underside below the
                    # visible top) before the convex decomposition
                    Vg = ground_mesh(np.maximum(Vw, [-np.inf, -np.inf, L.table_z]), L.table_z, max_gap=0.04)
                    src = write_obj(mesh_dir / f"{tr.name}_grounded.obj", *hull_mesh(Vg))
                    info["collision"] = "mesh extruded to the table, then convex"
                cres = self.tools.convex(src, tr.name, self.cfg.mass)
                parts, visual, com = cres["parts"], cres["visual"], cres["body_pos"]
                info.update(parts=len(parts), convex_drop=cres["info"].get("drop"))
            except Exception as exc:  # noqa: BLE001 - a tool failure falls back to the observed points
                pts = apply(tr.poses[0], tr.model)
                Vw = pts
                if dynamic:             # a moving object: the hull of its points extruded to the table
                    V, F = hull_mesh(np.concatenate([pts, pts * [1, 1, 0] + [0, 0, L.table_z]]))
                    com = V.mean(0)
                    parts, visual = [(V - com, F)], (V - com, F)
                    fb = "points-hull"
                else:                   # a fixture: height-field columns keep a container's concavity
                    cols = column_parts(pts, L.table_z)
                    com = np.concatenate([V for V, _ in cols]).mean(0)
                    parts, visual = [(V - com, F) for V, F in cols], None
                    fb = "height-field columns"
                info.update(mesh_backend=f"{fb} (mesh tool failed)", error=str(exc)[:300], parts=len(parts))
            low = Vw[Vw[:, 2] < Vw[:, 2].min() + 0.012]
            gap = float(Vw[:, 2].min() - L.table_z)
            if not dynamic and visual is not None and 0.002 < gap < 0.04 and len(low) >= 4:
                sup = np.concatenate([low, low * [1, 1, 0] + [0, 0, L.table_z + 0.0005]])
                try:
                    V, F = hull_mesh(sup)
                    parts = [*parts, (V - com, F)]
                    info["support_part"] = round(gap, 4)
                except Exception:  # noqa: BLE001 - flat band: no support part
                    pass
            allv = np.concatenate([V + com for V, _ in parts])
            ext = box_extent(allv, L.table_z)
            info["size_m"] = [round(x, 4) for x in ext["size"]]
            P0 = pose(None, com)
            off = inv(tr.poses[0]) @ P0
            self.body_poses[tr.name] = np.einsum("tij,jk->tik", tr.poses, off)
            rgba = (*[c / 255 for c in tr.colour], 1.0)
            self.objects.append(SceneObject(tr.name, tr.label, parts, visual, P0, rgba, dynamic, self.cfg.mass, info))
            info["dynamic"] = dynamic
            if self.rec.gt is not None:
                inv_match = {v: k for k, v in getattr(self, "gt_match", {}).items()}
                g = self.rec.gt["objects"].get(inv_match.get(tr.name, ""), None)
                if g is not None:
                    hs = np.asarray(g["half_size"], float)
                    gt_size = 2 * hs if g["kind"] != "cyl" else np.array([2 * hs[0], 2 * hs[0], 2 * hs[2]])
                    info["size_error"] = size_error(ext["size"], gt_size)
                    info["gt_object"] = inv_match[tr.name]
            per.append(info)
        return self._step("objects", t0, {"objects": per})

    # -- 7b. grasps (multi-finger hand) ------------------------------------------------------------------------------
    def refine_grasps(self) -> dict[str, Any]:
        """Bend the retargeted hand motion onto planned grasps of the objects the recording grasps."""
        t0 = time.monotonic()
        if self.dex is None:
            return self._step("grasps", t0, {"skipped": "no multi-finger hand (--hand)"})
        from hylo.manip.dexgrasp import DexGraspPlanner, ObjectGeom, PlannerConfig, mesh_surface
        from hylo.manip.handpath import bend_onto_grasp

        dx, res = self.dex, self.dex["ret"]
        tcp, fingers, out = res.tcp.copy(), res.q.copy(), []
        planner = DexGraspPlanner(dx["model"], PlannerConfig(n_dirs=28, n_rolls=6, keep=6))
        objs = {o.name: o for o in self.objects}
        for e in getattr(self, "ref_events", []) if self.cfg.refine_grasps else []:
            if e["type"] != "grasp" or e.get("object") not in objs:
                continue
            a, name = int(e["frame"]), e["object"]
            b = next((b for a0, b in self.intervals if a0 == a), a + 1)
            P = self.body_poses[name][a]
            geom = ObjectGeom.from_parts([apply(P, mesh_surface(V, F)) for V, F in objs[name].parts],
                                         mass=objs[name].mass)
            gs = planner.plan(geom, table_z=self.lift.table_z, prefer={"tcp": res.tcp[a], "w_rot": 1.0, "w_pos": 0.6})
            if not gs:
                out.append({"object": name, "frame": a, "plan": None})
                continue
            tcp, fingers, info = bend_onto_grasp(tcp, fingers, gs[0], a, b, self.rec.fps, ramp_s=self.cfg.grasp_ramp_s)
            out.append({"object": name, "frame": a, "release_frame": int(b), "candidates": len(gs),
                        "epsilon": round(gs[0].epsilon, 3), "force_closure": gs[0].force_closure,
                        "style": gs[0].style, "contact_fingers": sorted(set(gs[0].contact_fingers)), **info})
        dx.update(tcp=tcp, fingers=fingers, grasps=out)
        return self._step("grasps", t0, {"hand": dx["name"], "refined": out,
                                         "note": None if self.cfg.refine_grasps else "pure retargeting"})

    # -- 8. scene -----------------------------------------------------------------------------------------------
    def build_scene(self) -> dict[str, Any]:
        t0 = time.monotonic()
        L = self.lift
        H, W = self.rec.rgb.shape[1:3]
        cam = {"K": L.K, "T_world_cam": L.T_frame_cam, "width": W, "height": H}
        rs = self.cfg.render_scale
        size = (int(W * rs), int(H * rs))
        self.sim = SimClient(self.out)
        if self.rec.agent == "panda":
            arm_model = getattr(self, "arm_model", None) or self._robot_spec().arms[0].model
            info = self.sim.build(table_z=L.table_z, table_extent=L.table_extent, objects=self.objects, agent="panda",
                                  arm_model=arm_model, base=self.base, camera=cam, image_size=size)
        elif self.dex is not None:
            hm = self.dex["model"]
            hand = {"xml": self.dex["model_file"], "joints": list(hm.hand.joints),
                    "tcp_in_ee": hm.tcp_in_ee, "open": hm.open, "closed": hm.closed, "ee": hm.ee_body,
                    "kp_scale": self.cfg.hand_kp_scale, "name": self.dex["name"]}
            info = self.sim.build(table_z=L.table_z, table_extent=L.table_extent, objects=self.objects, agent="hand",
                                  camera=cam, gripper_T0=self.dex["tcp"][0], image_size=size, hand=hand)
        else:
            info = self.sim.build(table_z=L.table_z, table_extent=L.table_extent, objects=self.objects,
                                  agent="gripper" if self.tcp is not None else "none", camera=cam,
                                  gripper_T0=None if self.tcp is None else self.tcp[0], image_size=size)
        self.scene = info
        return self._step("scene", t0, info)

    # -- 9. replay ----------------------------------------------------------------------------------------------
    def _hand_cmd(self) -> np.ndarray:
        if self.rec.gripper_cmd is not None:
            return np.asarray(self.rec.gripper_cmd, float)
        cmd = np.ones(self.rec.T)
        for a, b in self.intervals:
            cmd[a:b] = 0.0
        return cmd

    def replay(self) -> dict[str, Any]:
        t0 = time.monotonic()
        held = {}
        for e in getattr(self, "ref_events", []):
            if e["type"] == "grasp":
                for a, b in self.intervals:
                    if a == e["frame"]:
                        held.update({t: e["object"] for t in range(a, b)})
        opening = self.gripper if self.gripper is not None else np.ones(self.rec.T)
        if self.dex is not None:
            res = self.sim.replay(time=self.rec.time, q=None, tcp=self.dex["tcp"], opening=opening,
                                  hand_cmd=self._hand_cmd(), obj_poses=self.body_poses, held=held,
                                  fingers=self.dex["fingers"])
        else:
            res = self.sim.replay(time=self.rec.time, q=self.q, tcp=self.tcp, opening=opening,
                                  hand_cmd=self._hand_cmd(), obj_poses=self.body_poses, held=held)
        self.kin, self.phys = res["kinematic"], res["physics"]
        return self._step("replay", t0, {"kinematic": {k: v for k, v in self.kin.items() if k != "qpos"},
                                         "physics": {"sim_time_s": round(self.phys["sim_time"], 2),
                                                     "diverged": self.phys["diverged"]}})

    # -- 10. checks ---------------------------------------------------------------------------------------------
    def checks(self) -> dict[str, Any]:
        t0 = time.monotonic()
        cmd = self._hand_cmd()
        sim_ev = sim_events(self.rec.time, self.phys, self.lift.table_z, cmd)
        ref = getattr(self, "ref_events", [])
        out: dict[str, Any] = {"events": {"recorded": ref, "simulated": sim_ev,
                                          "match": match_events(ref, sim_ev, self.cfg.event_tol_s)}}
        dyn = [o.name for o in self.objects if o.dynamic]
        out["pose_error_sim_vs_tracked"] = {n: _brief(pose_errors(self.body_poses[n], self.phys["poses"][n]))
                                            for n in dyn}
        ref_final = {n: self.body_poses[n][-1] for n in dyn}
        if self.rec.gt is not None:
            inv_match = {v: k for k, v in getattr(self, "gt_match", {}).items()}
            gt_err, sim_gt = {}, {}
            for n in dyn:
                g = self.rec.gt["objects"].get(inv_match.get(n, ""))
                if g is None:
                    continue
                gp = np.asarray(g["poses"], float)
                # the body frame is the mesh centroid, the ground truth the object centre: compare positions after
                # removing the frame-0 offset (a constant perception offset is reported separately)
                off = self.body_poses[n][0][:3, 3] - gp[0][:3, 3]
                tr_shift = self.body_poses[n].copy()
                tr_shift[:, :3, 3] -= off
                sim_shift = self.phys["poses"][n].copy()
                sim_shift[:, :3, 3] -= off
                gt_err[n] = {"initial_offset_m": round(float(np.linalg.norm(off)), 4),
                             **_brief(pose_errors(gp, tr_shift), rot=False)}
                sim_gt[n] = _brief(pose_errors(gp, sim_shift), rot=False)
                ref_final[n] = pose(None, gp[-1][:3, 3] + off)
            out["pose_error_tracked_vs_gt"] = gt_err
            out["pose_error_sim_vs_gt"] = sim_gt
        sim_final = {n: P[-1] for n, P in self.phys["poses"].items()}
        moved = [n for n in dyn if np.linalg.norm(ref_final[n][:3, 3] - self.body_poses[n][0][:3, 3]) > 0.02]
        out["final_state"] = final_state(ref_final, sim_final, moved, self.cfg.final_tol)
        rec_success = self.rec.meta.get("success")
        out["recording_success"] = rec_success
        out["success_reproduced"] = None if rec_success is None else bool(rec_success) and out["final_state"]["success"]
        out["kinematic"] = {k: v for k, v in self.kin.items() if k != "qpos"}
        return self._step("checks", t0, out)

    # -- 11. video ----------------------------------------------------------------------------------------------
    def video(self, path: str | Path | None = None, which: str = "physics") -> dict[str, Any]:
        t0 = time.monotonic()
        H, W = self.rec.rgb.shape[1:3]
        rs = self.cfg.render_scale
        qpos = (self.phys if which == "physics" else self.kin)["qpos"]
        r = self.tools.render(self.scene["path"], qpos, self.rec.time, camera="replay_cam", width=int(W * rs) // 2 * 2,
                              height=int(H * rs) // 2 * 2, fps=self.rec.fps)
        res: dict[str, Any] = {"render_backend": r.get("backend"), "fallback_reason": r.get("fallback_reason")}
        if r.get("video"):
            out = Path(path) if path else self.out / "replay.mp4"
            res.update(side_by_side(self.rec.rgb, r["video"], out, self.rec.fps, height=self.cfg.video_height,
                                    labels=("recording", f"simulation ({which} replay)")))
        return self._step("video", t0, res)

    # -- all ----------------------------------------------------------------------------------------------------
    def run(self, labels: list[str] | None = None, video: bool = True) -> dict[str, Any]:
        self.propose_labels(labels)
        self.segment()
        self.lift_to_3d()
        if self.rec.agent == "hand":
            self.hands()
        self.track()
        self.register()
        self.events()
        self.build_objects()
        if self.dex is not None:
            self.refine_grasps()
        self.build_scene()
        self.replay()
        self.checks()
        if video:
            self.video()
        return self.save()

    def save(self) -> dict[str, Any]:
        self.report["tools"] = getattr(self.tools, "calls", [])
        arrays: dict[str, np.ndarray] = {"time": self.rec.time}
        for n, P in self.body_poses.items():
            arrays[f"tracked_{n}"] = P
        if self.phys:
            for n, P in self.phys["poses"].items():
                arrays[f"sim_{n}"] = P
        if self.tcp is not None:
            arrays["tcp"] = self.tcp
        if self.dex is not None:
            arrays["hand_tcp"], arrays["hand_fingers"] = self.dex["tcp"], self.dex["fingers"]
            arrays["hand_kp"] = self.dex["kp"]
            arrays["hand_tcp_retargeted"], arrays["hand_fingers_retargeted"] = self.dex["ret"].tcp, self.dex["ret"].q
            arrays["hand_tip_error_mm"] = self.dex["ret"].tip_error_mm
        if self.q is not None:
            arrays["q"] = self.q
        np.savez_compressed(self.out / "tracks.npz", **arrays)
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, default=_json))
        return self.report


def _brief(e: dict[str, Any], rot: bool = True) -> dict[str, Any]:
    keep = ["frames", "trans_mean_m", "trans_median_m", "trans_max_m", "trans_final_m"]
    if rot:
        keep += ["rot_mean_deg", "rot_final_deg"]
    return {k: e[k] for k in keep if k in e}


def _json(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# ==================================================================================================== CLI
def replay_command(argv: list[str]) -> int:
    """replay to sim: a recorded episode -> MuJoCo scene -> kinematic and physics replay -> checks and video"""
    ap = argparse.ArgumentParser(prog="hylo replay", description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", choices=["libero", "droid", "video"])
    ap.add_argument("--camera", help="image key of a hf: dataset")
    ap.add_argument("--labels", help="comma-separated object labels (skip the VLM)")
    ap.add_argument("--agent", action="store_true", help="let a chat model drive the steps")
    ap.add_argument("--model", help="chat model spec (JSON or shorthand); default Ollama qwen3.5:35b-a3b")
    ap.add_argument("--no-vlm", action="store_true", help="labels from the instruction, no chat model")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--step", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--fps", type=float, help="frame rate to fetch / read at")
    ap.add_argument("--stub", action="store_true", help="no tool workers (needs recorded depth and segmentation)")
    ap.add_argument("--track-mode", default="yaw", choices=["xyz", "yaw", "full"])
    ap.add_argument("--render-backend", default="mujoco", choices=["auto", "blender", "mujoco"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--hand", help="human videos: a floating multi-finger robot hand instead of the parallel gripper "
                                   "(g1, allegro, or a full name such as allegro_right)")
    ap.add_argument("--hand-kp-scale", type=float, default=1.0, help="finger servo stiffness multiplier")
    ap.add_argument("--no-grasp-refine", action="store_true", help="--hand: pure retargeting, no planned grasps")
    ap.add_argument("--video-path")
    a = ap.parse_args(argv)

    def log(m: str) -> None:
        print(f"[replay] {m}", file=sys.stderr, flush=True)

    out = Path(a.out)
    rec = load_source(a.source, work=out, profile=a.profile, camera=a.camera, fps=a.fps, max_frames=a.max_frames)
    if a.step > 1 or (a.max_frames and rec.T > a.max_frames):
        rec = rec.subsample(a.step, a.max_frames)
    from .replay_tools import RealTools, StubTools, gt_segmenter

    if a.stub:
        if rec.gt is None or rec.gt.get("seg") is None:
            raise SystemExit("--stub needs a recording with ground-truth segmentation")
        tools: Any = StubTools(gt_segmenter(rec.gt["seg"][0], rec.gt["segments"]))
    else:
        tools = RealTools(log=log, render_backend=a.render_backend)
    model = None
    if a.agent or not (a.no_vlm or a.labels):
        from hylo.reasoners.llm.chat import build_chat

        spec = (json.loads(a.model) if a.model.lstrip().startswith("{") else a.model) if a.model else DEFAULT_MODEL
        model = build_chat(spec)
    cfg = ReplayConfig(track_mode=a.track_mode, hand=a.hand, hand_kp_scale=a.hand_kp_scale,
                       refine_grasps=not a.no_grasp_refine)
    pipe = ReplayPipeline(rec, out, tools, model=model, config=cfg, log=log)
    try:
        if a.agent:
            report = run_agent(pipe, model, turns=a.turns, log=log)
        else:
            labels = [s for s in a.labels.split(",")] if a.labels else None
            if a.no_vlm and not labels:
                labels = labels_from_text(rec.instruction) or None
            report = pipe.run(labels=labels, video=not a.no_video)
            if a.video_path and "video" in report["steps"] and report["steps"]["video"].get("path"):
                import shutil

                shutil.copy2(report["steps"]["video"]["path"], a.video_path)
    finally:
        tools.close()
    c = report["steps"].get("checks", {})
    print(json.dumps({"out": str(out), "events": c.get("events", {}).get("match", {}).get("matched"),
                      "of": c.get("events", {}).get("match", {}).get("of"), "final_state": c.get("final_state"),
                      "success_reproduced": c.get("success_reproduced"), "timing": report["timing"]}, indent=2,
                     default=_json))
    return 0


__all__ = ["AGENT_SYSTEM", "DEFAULT_MODEL", "ReplayAPI", "ReplayConfig", "ReplayPipeline", "replay_command", "run_agent"]
