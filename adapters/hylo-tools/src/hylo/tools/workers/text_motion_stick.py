# hylo: worker-side (runs in a tool worker's own environment; never imported by Hylo packages)
"""Stick-figure rendering of skeleton motion (PIL + imageio; no OpenGL): an oblique orthographic view that follows
the root, a floor grid, the root's trail and per-side colours."""

from __future__ import annotations

import numpy as np

LEFT, RIGHT, MID = (40, 110, 220), (225, 80, 40), (60, 60, 70)


def _side_colour(name_or_idx, names):
    n = names[name_or_idx] if names is not None else ""
    return LEFT if n.startswith("Left") or n.startswith("l_") else RIGHT if n.startswith("Right") or n.startswith("r_") else MID


def _project(p, centre, azim=np.radians(-60), elev=np.radians(15), scale=140.0, size=(480, 360)):
    """World (z up) -> image: rotate about z by azim, tilt by elev, orthographic."""
    W, H = size
    q = p - centre
    ca, sa = np.cos(azim), np.sin(azim)
    x = ca * q[..., 0] - sa * q[..., 1]
    y = sa * q[..., 0] + ca * q[..., 1]
    ce, se = np.cos(elev), np.sin(elev)
    up = ce * q[..., 2] - se * y
    depth = se * q[..., 2] + ce * y
    return np.stack([W / 2 + scale * x, H * 0.5 - scale * up], -1), depth


def stick_frames(joints, parents, names=None, size=(480, 360), stride=1, title=""):
    from PIL import Image, ImageDraw
    J = np.asarray(joints, float)
    T = len(J)
    W, H = size
    root = J[:, 0]
    frames = []
    for t in range(0, T, stride):
        centre = np.array([root[t, 0], root[t, 1], 0.9])
        img = Image.new("RGB", size, (246, 246, 243))
        d = ImageDraw.Draw(img)
        # floor grid around the root
        g0 = np.floor(root[t, :2] / 0.5) * 0.5
        for k in range(-6, 7):
            for a, b in (((g0[0] + k * 0.5, g0[1] - 3, 0), (g0[0] + k * 0.5, g0[1] + 3, 0)),
                         ((g0[0] - 3, g0[1] + k * 0.5, 0), (g0[0] + 3, g0[1] + k * 0.5, 0))):
                (pa, _), (pb, _) = _project(np.array(a), centre, size=size), _project(np.array(b), centre, size=size)
                d.line([tuple(pa), tuple(pb)], fill=(222, 222, 216), width=1)
        trail, _ = _project(np.c_[root[: t + 1, :2], np.zeros(t + 1)], centre, size=size)
        if len(trail) > 1:
            d.line([tuple(p) for p in trail], fill=(150, 190, 150), width=2)
        p2, depth = _project(J[t], centre, size=size)
        bones = [(j, int(parents[j])) for j in range(len(parents)) if parents[j] >= 0]
        bones.sort(key=lambda b: -(depth[b[0]] + depth[b[1]]))       # far first
        for j, pj in bones:
            col = _side_colour(j, names)
            fine = names is not None and ("Hand" in names[j] and names[j] not in ("LeftHand", "RightHand"))
            d.line([tuple(p2[pj]), tuple(p2[j])], fill=col, width=1 if fine else 4)
        hx, hy = p2[min(6, len(p2) - 1)] if names is None or "Head" not in names else p2[names.index("Head")]
        d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], outline=MID, width=3)
        if title:
            d.text((8, 6), title[:70], fill=(30, 30, 30))
        d.text((8, H - 16), f"frame {t}/{T - 1}", fill=(90, 90, 90))
        frames.append(np.asarray(img))
    return frames


def stick_video(joints, parents, out_mp4, fps, names=None, still=None, stride=1, title="", size=(480, 360)):
    import imageio.v2 as imageio
    frames = stick_frames(joints, parents, names, size=size, stride=stride, title=title)
    with imageio.get_writer(out_mp4, fps=fps / stride, codec="libx264", quality=6,
                            macro_block_size=8) as wr:   # imageio-ffmpeg writes yuv420p
        for f in frames:
            wr.append_data(f)
    if still:
        from PIL import Image
        pick = np.linspace(0, len(frames) - 1, 5).astype(int)
        Image.fromarray(np.concatenate([frames[i] for i in pick], axis=1)).save(still, optimize=True)
