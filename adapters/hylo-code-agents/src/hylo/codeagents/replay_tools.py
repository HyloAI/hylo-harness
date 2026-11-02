"""The perception and rendering tools the replay workflow calls, behind one small interface.

`RealTools` runs the hylo-tools workers (each in its own uv environment, on the CPU): `segment` (SAM 3), `depth`
(MoGe-2, a persistent worker for whole videos), `gravity` (GeoCalib), `mesh` (SAM 3D Objects when access is
granted, else depth carving), `convex` (CoACD + MJCF), `hands` (WiLoR, else MediaPipe) and `render` (Blender Cycles
or MuJoCo offscreen). `StubTools` gives the same interface without workers (tests, and runs that must not start
models): masks from a callback or the recording's segmentation, the recording's own depth, convex hulls for meshes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .replay_geom import box_mesh, hull_mesh, read_obj, write_obj
from .replay_track import apply, backproject, box_extent


class RealTools:
    name = "hylo-tools"

    def __init__(self, threads: int | None = None, log: Callable[[str], None] | None = None,
                 render_backend: str = "mujoco") -> None:
        self.threads, self.say = threads, log or (lambda m: None)
        self.render_backend = render_backend
        self._seg = None
        self._depth = None
        self.calls: list[dict[str, Any]] = []
        self.mesh_backend = "auto"

    def _note(self, tool: str, t0: float, **info: Any) -> None:
        self.calls.append({"tool": tool, "wall_s": round(time.monotonic() - t0, 2), **info})

    def segment(self, rgb: np.ndarray, labels: Sequence[str], threshold: float = 0.5) -> list[tuple[str, np.ndarray, float]]:
        from hylo.tools.segment import Sam3Segmenter

        t0 = time.monotonic()
        if self._seg is None:
            self._seg = Sam3Segmenter(threads=self.threads)
        self._seg.threshold = threshold
        out = self._seg(np.asarray(rgb), list(labels))
        self._note("segment", t0, labels=list(labels), masks=len(out), threshold=threshold, model="facebook/sam3")
        return out

    def depth(self, frames: np.ndarray, log_every: int = 20) -> dict[str, Any]:
        """Metric depth for every frame (one persistent MoGe-2 worker; the field of view of frame 0 is reused)."""
        from hylo.tools.depth import depth as run_depth
        from hylo.tools.runner import ToolServer

        t0 = time.monotonic()
        if self._depth is None:
            self._depth = ToolServer("depth", threads=self.threads)
            self._depth.start()
        out, K, fov = [], None, None
        for t, f in enumerate(frames):
            d = run_depth(np.asarray(f), fov_x=fov, preview=False, server=self._depth)
            z = np.where(d.mask & np.isfinite(d.depth), d.depth, 0.0).astype(np.float32)
            out.append(z)
            if K is None:
                K = np.asarray(d.K, float)
                fov = float(np.degrees(2 * np.arctan(0.5 * f.shape[1] / K[0, 0])))
            if t % log_every == 0:
                self.say(f"depth: frame {t + 1}/{len(frames)}")
        self._note("depth", t0, frames=len(frames), model="Ruicheng/moge-2-vitl-normal", fov_x_deg=round(fov or 0, 2))
        return {"depth": np.stack(out), "K": K, "fov_x_deg": fov}

    def gravity(self, rgb: np.ndarray) -> dict[str, Any]:
        from hylo.tools.gravity import gravity

        t0 = time.monotonic()
        g = gravity(np.asarray(rgb), preview=False, threads=self.threads)
        self._note("gravity", t0, roll_deg=round(float(np.degrees(g.roll)), 2), pitch_deg=round(float(np.degrees(g.pitch)), 2))
        return {"R_level_cam": np.asarray(g.R_level_cam, float), "roll": g.roll, "pitch": g.pitch, "K": g.K,
                "vfov_deg": g.vfov_deg}

    def mesh(self, rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
             name: str, out_dir: Path) -> dict[str, Any]:
        from hylo.tools.mesh import mesh as run_mesh

        t0 = time.monotonic()
        m = run_mesh(np.asarray(rgb), mask, depth=np.asarray(depth, float), K=K, backend=self.mesh_backend,
                     T_world_cam=T_world_cam, threads=self.threads)
        if m.fallback_reason:
            self.mesh_backend = "depth"          # access is checked once per run
        V, F = read_obj(m.obj)
        self._note("mesh", t0, object=name, backend=m.backend, fallback=m.fallback_reason, extents=m.info.get("extents_m"))
        return {"V": V, "F": F, "obj": str(m.obj), "backend": m.backend, "info": m.info,
                "fallback_reason": m.fallback_reason}

    def convex(self, mesh_obj: str | Path, name: str, mass: float) -> dict[str, Any]:
        from hylo.tools.convex import convex as run_convex

        t0 = time.monotonic()
        c = run_convex(mesh_obj, name=name, mass=mass, preview=False, threads=self.threads)
        parts = [read_obj(p) for p in c.parts]
        visual = read_obj(c.mjcf.parent / "meshes" / f"{name}_visual.obj")
        self._note("convex", t0, object=name, parts=len(parts), drop=c.info.get("drop"))
        return {"parts": parts, "visual": visual, "body_pos": np.asarray(c.info["body_pos"], float), "mjcf": str(c.mjcf),
                "info": {k: v for k, v in c.info.items() if k != "mjcf_snippet"}}

    def hands(self, frames: np.ndarray, K: np.ndarray | None) -> dict[str, Any]:
        from hylo.tools.hands import hands as run_hands

        t0 = time.monotonic()
        h = run_hands(np.asarray(frames), K=K, max_hands=1, overlay=False, threads=self.threads)
        self._note("hands", t0, backend=h.backend, frames=len(frames), valid=int(np.asarray(h.valid).reshape(len(frames), -1)[:, 0].sum()),
                   fallback=h.fallback_reason)
        valid = np.asarray(h.valid).reshape(len(frames), -1)[:, 0].astype(bool)
        hd = np.asarray(h.handedness).reshape(len(frames), -1)[:, 0]
        hd = hd[valid & (hd >= 0)]
        side = None if not len(hd) else ("right" if float(np.mean(hd)) >= 0.5 else "left")
        return {"backend": h.backend, "kp2d": np.asarray(h.kp2d)[:, 0], "kp3d_cam": np.asarray(h.kp3d_cam)[:, 0],
                "valid": valid, "side": side, "fallback_reason": h.fallback_reason}

    def render(self, scene: str | Path, qpos: np.ndarray, times: np.ndarray, *, camera: str, width: int, height: int,
               fps: float) -> dict[str, Any]:
        from hylo.tools.render import render

        t0 = time.monotonic()
        r = render(scene, {"qpos": np.asarray(qpos), "time": np.asarray(times)}, backend=self.render_backend,
                   width=width, height=height, fps=fps, camera=camera, max_frames=2000, samples=16,
                   threads=self.threads)
        self._note("render", t0, backend=r.backend, frames=r.frames, fallback=r.fallback_reason)
        return {"video": str(r.video), "still": str(r.still), "backend": r.backend, "fallback_reason": r.fallback_reason}

    def close(self) -> None:
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        if self._depth is not None:
            self._depth.close()
            self._depth = None


class StubTools:
    """No workers: deterministic stand-ins with the same interface (tests; dry runs)."""
    name = "stub"

    def __init__(self, segmenter: Callable[[np.ndarray, Sequence[str]], list[tuple[str, np.ndarray, float]]] | None = None,
                 depth_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
                 render_fn: Callable[..., dict[str, Any]] | None = None, table_z: float = 0.0) -> None:
        self.segmenter, self.depth_fn, self.render_fn = segmenter, depth_fn, render_fn
        self.table_z = table_z
        self.calls: list[dict[str, Any]] = []

    def segment(self, rgb: np.ndarray, labels: Sequence[str], threshold: float = 0.5) -> list[tuple[str, np.ndarray, float]]:
        if self.segmenter is None:
            raise RuntimeError("StubTools has no segmenter")
        self.calls.append({"tool": "segment(stub)", "labels": list(labels)})
        return self.segmenter(rgb, labels)

    def depth(self, frames: np.ndarray, log_every: int = 20) -> dict[str, Any]:
        if self.depth_fn is None:
            raise RuntimeError("StubTools has no depth function (the recording should carry depth)")
        return self.depth_fn(frames)

    def gravity(self, rgb: np.ndarray) -> dict[str, Any]:
        return {"R_level_cam": None}

    def mesh(self, rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
             name: str, out_dir: Path) -> dict[str, Any]:
        p, _ = backproject(depth, K, mask)
        pts = apply(T_world_cam, p)
        pts = pts[pts[:, 2] > self.table_z + 0.004]
        ext = box_extent(pts, self.table_z)
        V, F = box_mesh(ext["size"], ext["yaw"], ext["centre"])
        obj = write_obj(Path(out_dir) / f"{name}_mesh.obj", V, F)
        self.calls.append({"tool": "mesh(stub box)", "object": name})
        return {"V": V, "F": F, "obj": str(obj), "backend": "stub-box", "info": {"extents_m": ext["size"]},
                "fallback_reason": None}

    def convex(self, mesh_obj: str | Path, name: str, mass: float) -> dict[str, Any]:
        V, F = read_obj(mesh_obj)
        V, F = hull_mesh(V)
        c = V.mean(0)
        self.calls.append({"tool": "convex(stub hull)", "object": name})
        return {"parts": [(V - c, F)], "visual": (V - c, F), "body_pos": c, "mjcf": None, "info": {"parts": 1}}

    def hands(self, frames: np.ndarray, K: np.ndarray | None) -> dict[str, Any]:
        raise RuntimeError("StubTools has no hand tracker")

    def render(self, scene: str | Path, qpos: np.ndarray, times: np.ndarray, **kw: Any) -> dict[str, Any]:
        if self.render_fn is None:
            return {"video": None, "backend": "none"}
        return self.render_fn(scene, qpos, times, **kw)

    def close(self) -> None:
        pass


def gt_segmenter(seg0: np.ndarray, segments: dict[str, str]) -> Callable[[np.ndarray, Sequence[str]], list]:
    """A segmenter from a ground-truth instance image: each label gets the instance whose name contains its words."""
    def run(rgb: np.ndarray, labels: Sequence[str]) -> list[tuple[str, np.ndarray, float]]:
        out = []
        for lab in labels:
            words = set(lab.lower().split())
            best = max(segments.items(), key=lambda kv: len(words & set(kv[1].lower().split())), default=None)
            if best and words & set(best[1].lower().split()):
                m = seg0 == int(best[0])
                if m.sum() > 10:
                    out.append((lab, m, 1.0))
        return out
    return run


__all__ = ["RealTools", "StubTools", "gt_segmenter"]
