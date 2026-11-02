# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["coacd>=1.0.5", "trimesh>=4.4", "numpy>=1.26", "scipy>=1.10", "rtree>=1.2"]
# ///
"""Collision geometry for a visual mesh (RFC-0008 part 10), in the file format a stage's `collision.parts` reads.

params: strategy (auto | box | hull | parts | voxels), scale (1.0), density (kg/m^3, 500), mass (kg, optional),
        concavity_threshold (0.25: auto uses convex parts above it), box_fit (0.9: auto uses one box when the mesh
        fills at least this fraction of its oriented bounding box), max_parts (16: auto turns to voxels when a
        watertight mesh needs more), coacd_threshold (0.05), voxel (metres, default: longest side / 32),
        max_boxes (256)
inputs: mesh (.obj, .stl, .glb, .ply)
outputs: collision.json: {"strategy", "parts": [{"vertices": [[x,y,z]...]}]} (hull, parts) or
         {"strategy", "boxes": [[cx, cy, cz, hx, hy, hz], ...]} (box, voxels), in the mesh's own frame
         JSON: strategy, why, volume_m3, hull_volume_m3, concavity, box_fill, watertight, pieces, mass_kg
"""

from __future__ import annotations

import json

import numpy as np

from _common import main, threads, versions

LICENCE = "MIT (CoACD, trimesh)"


def greedy_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Cover the filled cells of a 3D grid with axis-aligned boxes: from each uncovered cell grow along x, then y,
    then z as long as every cell is filled and uncovered. Returns (x0, y0, z0, x1, y1, z1) half-open index boxes."""
    todo = grid.copy()
    out = []
    nx, ny, nz = grid.shape
    for x, y, z in zip(*np.nonzero(grid)):
        if not todo[x, y, z]:
            continue
        x1 = x + 1
        while x1 < nx and todo[x1, y, z]:
            x1 += 1
        y1 = y + 1
        while y1 < ny and todo[x:x1, y1, z].all():
            y1 += 1
        z1 = z + 1
        while z1 < nz and todo[x:x1, y:y1, z1].all():
            z1 += 1
        todo[x:x1, y:y1, z:z1] = False
        out.append((x, y, z, x1, y1, z1))
    return out


def voxel_boxes(mesh, pitch: float, max_boxes: int):
    vox = mesh.voxelized(pitch).fill()
    grid = np.asarray(vox.matrix, bool)
    boxes = greedy_boxes(grid)
    while len(boxes) > max_boxes:                        # too many: coarser voxels
        pitch *= 1.3
        vox = mesh.voxelized(pitch).fill()
        grid = np.asarray(vox.matrix, bool)
        boxes = greedy_boxes(grid)
    T = np.asarray(vox.transform)                         # index -> mesh frame (cell centres)
    out = []
    for x0, y0, z0, x1, y1, z1 in boxes:
        lo = (T @ np.array([x0 - 0.5, y0 - 0.5, z0 - 0.5, 1.0]))[:3]
        hi = (T @ np.array([x1 - 0.5, y1 - 0.5, z1 - 0.5, 1.0]))[:3]
        c, h = (lo + hi) / 2, np.abs(hi - lo) / 2
        out.append([*np.round(c, 6).tolist(), *np.round(h, 6).tolist()])
    return out, pitch, int(grid.sum())


def handle(state, req):
    import trimesh

    threads()
    mesh = trimesh.load(req.input("mesh"), force="mesh", process=True)
    if req.get("scale", 1.0) != 1.0:
        mesh.apply_scale(float(req.get("scale")))
    hull = mesh.convex_hull
    watertight = bool(mesh.is_watertight)
    vol = float(abs(mesh.volume)) if watertight else float(hull.volume)
    hull_vol = float(hull.volume)
    concavity = max(0.0, 1.0 - vol / hull_vol) if hull_vol > 0 else 0.0
    obb = mesh.bounding_box_oriented
    box_fill = vol / float(obb.volume) if obb.volume > 0 else 0.0
    strategy = req.get("strategy", "auto")
    why = f"chosen: {strategy}"
    if strategy == "auto":
        if watertight and box_fill >= float(req.get("box_fit", 0.9)):
            strategy, why = "box", f"fills {box_fill:.0%} of its oriented box"
        elif concavity < float(req.get("concavity_threshold", 0.25)):
            strategy, why = "hull", f"nearly convex (concavity {concavity:.2f})"
        else:
            strategy, why = "parts", f"concave (concavity {concavity:.2f})"
    doc: dict = {"strategy": strategy}
    if strategy == "box":
        T = np.asarray(obb.primitive.transform)
        ext = np.asarray(obb.primitive.extents) / 2
        if not np.allclose(T[:3, :3], np.eye(3), atol=1e-6):        # rotated box: write it as its 8 corners' hull
            doc = {"strategy": "hull", "parts": [{"vertices": np.round(obb.vertices, 6).tolist()}]}
            strategy = "hull"
        else:
            doc["boxes"] = [[*np.round(T[:3, 3], 6).tolist(), *np.round(ext, 6).tolist()]]
    if strategy == "hull":
        doc = {"strategy": "hull", "parts": [{"vertices": np.round(hull.vertices, 6).tolist()}]}
    if strategy == "parts":
        import coacd

        coacd.set_log_level("warn")
        cm = coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32))
        parts = coacd.run_coacd(cm, threshold=float(req.get("coacd_threshold", 0.05)), seed=0)
        if watertight and len(parts) > int(req.get("max_parts", 16)) and req.get("strategy", "auto") == "auto":
            strategy, why = "voxels", f"{len(parts)} convex parts would be needed: voxel boxes instead"
        else:
            doc = {"strategy": "parts", "parts": [{"vertices": np.round(np.asarray(v), 6).tolist()} for v, _ in parts]}
    if strategy == "voxels":
        pitch = float(req.get("voxel") or max(mesh.extents) / 32.0)
        boxes, pitch, cells = voxel_boxes(mesh, pitch, int(req.get("max_boxes", 256)))
        doc = {"strategy": "voxels", "boxes": boxes, "voxel_m": round(pitch, 6), "cells": cells}
    pieces = len(doc.get("parts") or doc.get("boxes") or [])
    mass = float(req.get("mass")) if req.get("mass") else vol * float(req.get("density", 500.0))
    doc.update(mass_kg=round(mass, 6), volume_m3=round(vol, 9), source_mesh=str(req.input("mesh")))
    out = req.path("collision.json", "collision")
    out.write_text(json.dumps(doc))
    return {"strategy": doc["strategy"], "why": why, "volume_m3": round(vol, 9), "hull_volume_m3": round(hull_vol, 9),
            "concavity": round(concavity, 4), "box_fill": round(box_fill, 4), "watertight": watertight,
            "pieces": pieces, "mass_kg": round(mass, 6), "versions": versions("trimesh", "coacd"),
            "licence": LICENCE}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle))
