"""Retargeting evaluation: a human hand video -> 21 MediaPipe keypoints per frame (the `hands` tool) -> each robot
hand (hylo.manip.retarget) -> per-frame fingertip error (robot scale, palm frame) and solve time per frame. With
--video, a side-by-side MP4 (at most 4 MB): the recording with the detected skeleton | the human hand in its palm frame
| each robot hand in the same view.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/dexterous/retarget_eval.py \\
        hf:rli14/10132025_human_demonstrations_hurricanemug_plaintable:0 --profile video \\
        --camera observation.image.ego_global --out runs/dex-retarget [--hands g1 allegro] [--max-frames N] [--video]

SOURCE is anything `hylo replay` reads (hf:REPO:EPISODE, a video file); the keypoints are cached in OUT/hands.npz.
Writes OUT/results.json ({hand: summary with tip_error_mm mean / median / p95 / max, ms_per_frame}, per-frame
errors) and OUT/<hand>.npz (finger joints, palm poses, errors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
         (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
PANEL = (320, 240)
VIEW_DIST = 0.45                       # camera distance from the palm (m), robot views and human drawing alike


def keypoints(rec, out: Path) -> dict:
    cache = out / "hands.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if len(z["kp3d"]) == rec.T:
            return {"kp3d": z["kp3d"], "kp2d": z["kp2d"], "valid": z["valid"], "side": str(z["side"]),
                    "backend": str(z["backend"])}
    from hylo.codeagents.replay_tools import RealTools

    tools = RealTools()
    try:
        h = tools.hands(rec.rgb, None)
    finally:
        tools.close()
    res = {"kp3d": h["kp3d_cam"], "kp2d": h["kp2d"], "valid": h["valid"], "side": h.get("side") or "right",
           "backend": h["backend"]}
    np.savez_compressed(cache, **res)
    return res


def robot_hand(family: str, side: str):
    """HandModel and model file of a library hand (built by the floating-hand environment)."""
    from hylo.core.registry import registry
    from hylo.manip.handmodel import HandModel

    name = family if family.endswith(("_left", "_right")) else f"{family}_{side}"
    env = registry.build("envs", {"type": "mujoco.dex_table", "params": {"hand": name, "cameras": []}})
    arm = env.robot_spec().arms[0]
    return name, HandModel.from_arm(arm), arm.model


class HandView:
    """Renders a floating-hand model at finger joints q, seen from the palm side (fingers up)."""

    def __init__(self, model_file: str, hm, F_r: np.ndarray, size=PANEL) -> None:
        import mujoco

        spec = mujoco.MjSpec.from_file(model_file)
        m0 = spec.compile()
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        ee = m0.body(hm.ee_body).id
        R, p = d0.xmat[ee].reshape(3, 3), d0.xpos[ee]
        x, y, z = (R @ F_r[:3, i] for i in range(3))
        centre = R @ F_r[:3, 3] + p + x * 0.06          # 6 cm up the palm from the wrist
        cam = spec.worldbody.add_camera(name="palm_view", pos=list(centre + z * VIEW_DIST), fovy=40)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [*(-y), *x]
        spec.worldbody.add_light(pos=list(centre + z * 0.6 + x * 0.2), dir=list(-z), diffuse=[0.9, 0.9, 0.9])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = size
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in hm.hand.joints]
        self.r = mujoco.Renderer(self.m, size[1], size[0])

    def __call__(self, q: np.ndarray) -> np.ndarray:
        import mujoco

        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="palm_view")
        return self.r.render().copy()


def draw_skeleton(img: np.ndarray, uv: np.ndarray, ok: bool, size=PANEL) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(img, np.uint8))
    s = size[0] / im.width
    im = im.resize(size)
    if ok:
        dr = ImageDraw.Draw(im)
        for a, b in EDGES:
            dr.line([tuple(uv[a] * s), tuple(uv[b] * s)], fill=(255, 210, 40), width=2)
    return np.asarray(im)


def draw_palm_frame(k: np.ndarray, scale: float, size=PANEL) -> np.ndarray:
    """The human hand in its palm frame, as the robot views show it (fingers up, palm side towards the viewer),
    scaled by the robot/human size ratio."""
    from PIL import Image, ImageDraw

    from hylo.manip.retarget import human_frame

    F = human_frame(k)
    loc = (k - F[:3, 3]) @ F[:3, :3] * scale
    loc[:, 0] -= 0.06
    px = size[1] / (2 * VIEW_DIST * np.tan(np.radians(20)))           # same projection scale as the robot view
    u = size[0] / 2 - loc[:, 1] * px
    v = size[1] / 2 - loc[:, 0] * px
    im = Image.new("RGB", size, (235, 236, 240))
    dr = ImageDraw.Draw(im)
    for a, b in EDGES:
        dr.line([(u[a], v[a]), (u[b], v[b])], fill=(40, 90, 200), width=3)
    for i in range(21):
        dr.ellipse([u[i] - 3, v[i] - 3, u[i] + 3, v[i] + 3], fill=(200, 60, 40))
    return np.asarray(im)


def label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([3, 3, 7 * len(text) + 9, 17], fill=(0, 0, 0))
    dr.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="video")
    ap.add_argument("--camera")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--hands", nargs="*", default=["g1", "allegro"])
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    from hylo.codeagents.replay_data import load_source
    from hylo.manip.retarget import Retargeter

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = load_source(a.source, work=out / "source", profile=a.profile, camera=a.camera, fps=a.fps,
                      max_frames=a.max_frames)
    if a.step > 1 or (a.max_frames and rec.T > a.max_frames):
        rec = rec.subsample(a.step, a.max_frames)
    kp = keypoints(rec, out)
    results = {"source": a.source, "frames": int(rec.T), "fps": round(rec.fps, 3), "keypoints": kp["backend"],
               "human_side": kp["side"], "valid_frames": int(np.sum(kp["valid"])), "hands": {}}
    runs = {}
    for fam in a.hands:
        name, hm, model_file = robot_hand(fam, kp["side"])
        rt = Retargeter(hm)
        res = rt.run(kp["kp3d"], valid=kp["valid"], side=kp["side"])
        np.savez_compressed(out / f"{name}.npz", q=res.q, ee=res.ee, tip_error_mm=res.tip_error_mm,
                            pair_error_mm=res.pair_error_mm, valid=res.valid)
        results["hands"][name] = {**res.summary(), "per_frame_tip_error_mm": np.round(res.tip_error_mm, 2).tolist()}
        runs[name] = (hm, model_file, rt, res)
        print(name, json.dumps(res.summary()), flush=True)
    if a.video:
        from hylo.codeagents.replay_geom import write_video

        views = {n: HandView(mf, hm, rt.F_r) for n, (hm, mf, rt, _) in runs.items()}
        scale = next(iter(runs.values()))[3].scale if runs else 1.0
        frames = []
        for t in range(rec.T):
            row = [label(draw_skeleton(rec.rgb[t], kp["kp2d"][t], bool(kp["valid"][t])), "recording"),
                   label(draw_palm_frame(kp["kp3d"][t], scale), "human hand (palm frame)")]
            for n, (_, _, _, res) in runs.items():
                row.append(label(views[n](res.q[t]), f"{n}  {res.tip_error_mm[t]:.1f} mm"))
            frames.append(np.concatenate(row, 1))
        results["video"] = write_video(frames, out / "retarget.mp4", fps=rec.fps)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
