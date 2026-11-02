# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.10", "trimesh>=4.4", "scikit-image>=0.22", "matplotlib>=3.8", "pillow>=10",
#                 "huggingface_hub>=0.24"]
# ///
"""Image + mask -> object mesh.

backend "sam3d": SAM 3D Objects (facebook/sam-3d-objects). Checks access to the gated weights first; while access is
    pending (403) it reports the block. The upstream pipeline needs CUDA extensions (pytorch3d, kaolin, gsplat) and a
    32 GB GPU; a CPU port is the next step once the weights can be downloaded.
backend "depth" (fallback, CPU, seconds): back-project the masked pixels of a metric depth map (from the `depth` tool
    or given), clean them (mask erosion, depth outliers), complete the unseen back by reflecting the visible surface
    across the plane of the silhouette points (the widest cross-section of a convex-ish object seen from one side),
    then reconstruct a closed surface: "carve" (default: voxels inside the mask's viewing frustum between the visible
    surface and its reflection, smoothed, marching cubes; watertight, keeps concavities seen in the silhouette),
    "hull" (convex hull of the completed points) or "relief" (the visible surface only, a height field; not closed).

params: backend (depth | sam3d), method (carve | hull | relief; default carve), erode (px, 2), scale (1.0),
        voxels (along the longest side, 64), max_points (20000), T_world_cam (4x4, optional: output in world coordinates), preview (true)
inputs: image, mask ((H, W) bool .npy/.png), depth (.npz with depth (H, W) and K (3, 3); or .npy depth with params K)
outputs: mesh.glb (vertex colours), mesh.obj, points.npz (front (N, 3), back (M, 3), colours), preview.png
         JSON: vertices, faces, watertight, volume_m3, extents_m, centre (camera or world frame), method, notes
"""

from __future__ import annotations

import numpy as np

from _common import Blocked, hf_access, load_array, load_rgb, main, versions

SAM3D = "facebook/sam-3d-objects"
LICENCE_FALLBACK = "Apache-2.0 (Hylo code); scikit-image BSD-3; depth from the depth tool (MoGe-2, MIT)"


def load_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    if path.endswith((".npy", ".npz")):
        m = load_array(path)
    else:
        from PIL import Image
        m = np.asarray(Image.open(path).convert("L")) > 127
    m = np.asarray(m)
    if m.ndim == 3:     # (N, H, W): union
        m = m.any(0)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} != image shape {shape}")
    return m.astype(bool)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    from scipy import ndimage
    return ndimage.binary_erosion(mask, iterations=r) if r > 0 else mask


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return p, np.stack([v, u], 1)


def clean(points: np.ndarray, pix: np.ndarray, k: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """Drop depth outliers (median absolute deviation along the depth axis) and isolated points."""
    z = points[:, 2]
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-6
    keep = np.abs(z - med) < k * 1.4826 * mad + 0.02 * med
    points, pix = points[keep], pix[keep]
    if len(points) > 50:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(points).query(points, k=9)
        m = d[:, 1:].mean(1)
        keep = m <= m.mean() + 2.5 * m.std()
        points, pix = points[keep], pix[keep]
    return points, pix


def silhouette_plane(points: np.ndarray, pix: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A plane (point, unit normal towards the camera) through the points on the mask boundary."""
    from scipy import ndimage
    border = mask & ~ndimage.binary_erosion(mask, iterations=3)
    on = border[pix[:, 0], pix[:, 1]]
    P = points[on] if on.sum() >= 10 else points
    c = P.mean(0)
    view = c / np.linalg.norm(c)                       # camera -> object
    if len(P) >= 10:
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[2]
        if abs(n @ view) < 0.5:                        # degenerate (boundary points nearly collinear): face the camera
            n = view
    else:
        n = view
    n = -n if n @ view > 0 else n                      # normal points towards the camera
    return c, n / np.linalg.norm(n)


def complete(points: np.ndarray, plane: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Reflect the visible surface across the silhouette plane (points already behind it are mirrored forward)."""
    c, n = plane
    d = (points - c) @ n
    return points - 2.0 * d[:, None] * n[None, :]


def carve(depth: np.ndarray, K: np.ndarray, mask: np.ndarray, plane: tuple[np.ndarray, np.ndarray],
          voxels: int = 64):
    """Voxels whose projection falls in the mask and whose depth lies between the visible surface and its reflection
    across the silhouette plane (measured along each pixel's ray), smoothed, then marching cubes."""
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes
    H, W = depth.shape
    Kinv = np.linalg.inv(K)
    c, n = plane
    z_f = np.where(mask & np.isfinite(depth), depth, np.nan)
    med = ndimage.median_filter(np.nan_to_num(z_f, nan=np.nanmedian(z_f)), size=5)
    z_f = np.where(np.isfinite(z_f), z_f, med)                      # holes inside the mask: local median
    v, u = np.nonzero(mask)
    rays = (Kinv @ np.stack([u + 0.5, v + 0.5, np.ones_like(u, float)])).T          # z = 1
    zf = z_f[v, u]
    z_p = (c @ n) / (rays @ n)                                      # ray-plane intersection depth
    thick = np.maximum(z_p - zf, 0.0)
    min_t = 0.15 * np.percentile(thick, 90) + 1e-4                  # objects are at least a little thick
    z_b = zf + np.maximum(2.0 * thick, min_t)
    back = np.full((H, W), np.nan)
    back[v, u] = z_b
    P = np.concatenate([rays * zf[:, None], rays * z_b[:, None]])
    lo, hi = P.min(0), P.max(0)
    step = (hi - lo).max() / voxels
    lo, hi = lo - 2 * step, hi + 2 * step
    axes = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    q = np.stack([X, Y, Z], -1).reshape(-1, 3)
    pix = (K @ q.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uu, vv = np.floor(pix[:, 0] / pix[:, 2]).astype(int), np.floor(pix[:, 1] / pix[:, 2]).astype(int)
    ok = (q[:, 2] > 0) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    occ = np.zeros(len(q), bool)
    i = np.flatnonzero(ok)
    inm = mask[vv[i], uu[i]]
    i = i[inm]
    occ[i] = (q[i, 2] >= z_f[vv[i], uu[i]]) & (q[i, 2] <= back[vv[i], uu[i]])
    occ = occ.reshape(X.shape)
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, nlab = ndimage.label(occ)
    if nlab > 1:
        sizes = ndimage.sum(occ, lab, range(1, nlab + 1))
        occ = lab == (1 + int(np.argmax(sizes)))
    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    field = np.pad(field, 1)
    verts, faces, _, _ = marching_cubes(field, 0.5, spacing=(step, step, step))
    verts += lo - step
    m = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=10)    # voxel stair-steps; keeps volume
    if m.volume < 0:
        m.invert()
    return m


def reconstruct(points: np.ndarray, method: str, front: np.ndarray, front_pix: np.ndarray, carve_args: tuple):
    import trimesh
    if method == "hull":
        return trimesh.convex.convex_hull(points), []
    if method == "relief":
        from scipy.spatial import Delaunay
        tri = Delaunay(front_pix[:, ::-1].astype(float))
        f = tri.simplices
        # drop triangles across depth discontinuities
        e = np.max([np.linalg.norm(front[f[:, i]] - front[f[:, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
        f = f[e < 4 * np.median(e)]
        return trimesh.Trimesh(front, f, process=True), []
    if method != "carve":
        raise ValueError(f"method {method!r}: carve, hull or relief")
    notes = []
    mesh = carve(*carve_args)
    if not mesh.is_watertight or len(mesh.faces) < 20:
        notes.append("carved surface not watertight: convex hull used")
        return trimesh.convex.convex_hull(points), notes
    return mesh, notes


def colour_vertices(mesh, points: np.ndarray, colours: np.ndarray) -> None:
    from scipy.spatial import cKDTree
    _, idx = cKDTree(points).query(mesh.vertices)
    rgba = np.concatenate([colours[idx], np.full((len(idx), 1), 255, np.uint8)], 1)
    mesh.visual.vertex_colors = rgba


def preview(rgb, mask, mesh, front, back, T_flip) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(9.6, 3.2), dpi=100)
    ax0 = fig.add_subplot(1, 3, 1)
    shown = rgb.copy()
    shown[~mask] = (shown[~mask] * 0.35).astype(np.uint8)
    ax0.imshow(shown)
    ax0.set_title("input + mask", fontsize=9)
    ax0.axis("off")
    m = mesh
    if len(m.faces) > 6000:
        try:
            m = m.simplify_quadric_decimation(face_count=6000)
        except Exception:  # noqa: BLE001
            pass
    V = m.vertices @ T_flip.T
    F = m.faces
    cols = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0 if m.visual.kind == "vertex" else np.full((len(V), 3), 0.7)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    shade = 0.45 + 0.55 * np.clip(np.abs(fn @ light), 0, 1)
    fc = np.clip(cols[F].mean(1) * shade[:, None], 0, 1)
    lo, hi = V.min(0), V.max(0)
    r = (hi - lo).max() / 2
    mid = (hi + lo) / 2
    for k, (elev, azim, title) in enumerate([(20, -60, "mesh"), (20, 120, "mesh, back")]):
        ax = fig.add_subplot(1, 3, k + 2, projection="3d")
        ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="none"))
        for a, (s, e) in zip("xyz", zip(mid - r, mid + r)):
            getattr(ax, f"set_{a}lim")(s, e)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def handle_sam3d(req):
    status = hf_access(SAM3D, "checkpoints/pipeline.yaml")
    if status == "gated":
        raise Blocked(f"{SAM3D}: access to the gated weights is not granted yet (403)",
                      f"request access at https://huggingface.co/{SAM3D} (logged-in user) and wait for approval; "
                      f"meanwhile backend 'depth' runs")
    if status != "ok":
        raise Blocked(f"{SAM3D}: weights not reachable ({status})", "check the network and `hf auth login`")
    raise Blocked(f"{SAM3D}: weights accessible, but the CPU port of the SAM 3D Objects pipeline (upstream needs "
                  "pytorch3d, kaolin and gsplat CUDA builds and a 32 GB GPU) is not implemented yet",
                  "port sam3d_objects inference to CPU (next step in hylo-tools); meanwhile backend 'depth' runs")


def handle(state, req):
    backend = req.get("backend", "depth")
    if backend == "sam3d":
        return handle_sam3d(req)
    import trimesh
    rgb = load_rgb(req.input("image"))
    H, W = rgb.shape[:2]
    mask = load_mask(req.input("mask"), (H, W))
    dpath = req.input("depth")
    if dpath.endswith(".npz"):
        with np.load(dpath) as z:
            depth = z["depth"].astype(np.float64)
            K = np.asarray(req.get("K") if req.get("K") is not None else z["K"], float)
    else:
        depth, K = load_array(dpath).astype(np.float64), np.asarray(req.get("K"), float)
    if depth.shape != (H, W):
        raise ValueError(f"depth shape {depth.shape} != image shape {(H, W)}")
    scale = float(req.get("scale", 1.0))
    method = req.get("method", "carve")
    m_used = erode(mask, int(req.get("erode", 2)))
    if m_used.sum() < 30:
        m_used = mask
    front, pix = backproject(depth * scale, K, m_used)
    if len(front) < 30:
        raise ValueError(f"only {len(front)} valid depth pixels inside the mask")
    front, pix = clean(front, pix)
    nmax = int(req.get("max_points", 20000))
    if len(front) > nmax:
        sel = np.random.default_rng(0).choice(len(front), nmax, replace=False)
        front, pix = front[sel], pix[sel]
    colours = rgb[pix[:, 0], pix[:, 1]]
    plane = silhouette_plane(front, pix, m_used)
    back = complete(front, plane)
    pts = np.concatenate([front, back]) if method != "relief" else front
    cols = np.concatenate([colours, colours]) if method != "relief" else colours
    mesh, notes = reconstruct(pts, method, front, pix,
                              (depth * scale, K, m_used, plane, int(req.get("voxels", 64))))
    colour_vertices(mesh, pts, cols)
    T = np.asarray(req.get("T_world_cam"), float) if req.get("T_world_cam") is not None else None
    if T is not None:
        mesh.apply_transform(T)
        frame = "world"
    else:
        frame = "camera (OpenCV: x right, y down, z forward)"
    mesh.export(req.path("mesh.glb", "glb"))
    mesh.export(req.path("mesh.obj", "obj"))
    req.save_npz("points", front=front, back=back, colours=colours, plane_point=plane[0], plane_normal=plane[1])
    if req.get("preview", True):
        # show camera-frame meshes with y up for the 3D views
        flip = np.diag([1.0, -1.0, -1.0]) if T is None else np.eye(3)
        req.save_png("preview", preview(rgb, mask, mesh, front, back, flip))
    req.provenance.update({"model": "depth back-projection + silhouette-plane completion + "
                                    + {"carve": "frustum carving + marching cubes", "hull": "convex hull",
                                       "relief": "height field"}[method],
                           "licence": LICENCE_FALLBACK, "backend": "depth",
                           "packages": versions("trimesh", "scikit-image", "scipy")})
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
            "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
            "extents_m": [round(float(x), 4) for x in mesh.extents], "centre": [round(float(x), 4) for x in
                                                                                mesh.bounds.mean(0)],
            "frame": frame, "method": method, "points": int(len(front)), "notes": notes,
            "sam3d_access": hf_access(SAM3D, "checkpoints/pipeline.yaml") if req.get("check_sam3d") else None}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle))
