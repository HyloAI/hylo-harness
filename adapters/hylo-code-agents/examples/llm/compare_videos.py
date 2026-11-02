"""Side-by-side video of two recorded episodes of the same task and seed (e.g. a campaign's parent and the promoted
program on one held-out seed): both re-simulated from their traces and rendered with Blender (or MuJoCo), the shorter
clip padded by holding its last frame, each panel labelled and ending with its outcome.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/llm/compare_videos.py \\
        LEFT_RUN_DIR LEFT_EPISODE "left label" RIGHT_RUN_DIR RIGHT_EPISODE "right label" OUT.mp4 [--mujoco] [--max-mb 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_videos import (F_HEAD, F_TEXT, FPS, blender_frames, encode, font, instance_for,  # noqa: E402
                         read_trace, replay, task_instruction)

F_MID = font("InterDisplay-Bold.ttf", 36)

PW, PH = 640, 480          # panel size
TOP, BOTTOM = 92, 44       # title band, status band


def episode_frames(run_dir: Path, episode_id: str, *, blender: bool, every: int, samples: int,
                   cache: Path | None = None) -> tuple[list, dict, str]:
    from hylo.core.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(json.loads((run_dir / "manifest.json").read_text())["config"])
    ep = read_trace(run_dir, episode_id)
    inst = instance_for(cfg, ep["start"])
    frames, qpos, res = replay(cfg, inst, ep["actions"], render=not blender)
    if blender:
        key = cache / f"{episode_id}-e{every}-s{samples}.npz" if cache else None
        if key is not None and key.exists():
            with np.load(key) as z:
                bf = list(z["frames"])
        else:
            with tempfile.TemporaryDirectory(prefix="hylo-cmp-") as tmp:
                bf = blender_frames(res["mjb"], qpos, every, samples, Path(tmp))
            if key is not None:
                key.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(key, frames=np.asarray(bf))
        frames = [bf[min(i // every, len(bf) - 1)] for i in range(res["steps"] + 1)]
    recorded = bool((ep["end"] or {}).get("success"))
    out = {"success": recorded, "steps": res["steps"], "replay_matches_trace": res["success"] == recorded,
           "seed": inst.seed, "task": inst.spec.id}
    return frames, out, task_instruction(cfg, inst)


def panel(img: np.ndarray, step: int, steps: int, done: bool, ok: bool) -> Image.Image:
    im = Image.fromarray(img).convert("RGB").resize((PW, PH), Image.LANCZOS).convert("RGBA")
    if done:
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        colour = (34, 139, 84) if ok else (178, 48, 48)
        d.rectangle([PW // 2 - 130, PH - 78, PW // 2 + 130, PH - 18], fill=(*colour, 225))   # low: the scene stays
        d.text((PW // 2, PH - 48), "SUCCESS" if ok else "FAILURE", font=F_MID, fill=(255, 255, 255, 255), anchor="mm")
        im = Image.alpha_composite(im, layer)
    return im.convert("RGB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("left_run")
    p.add_argument("left_episode")
    p.add_argument("left_label")
    p.add_argument("right_run")
    p.add_argument("right_episode")
    p.add_argument("right_label")
    p.add_argument("out")
    p.add_argument("--mujoco", action="store_true", help="MuJoCo frames instead of Blender")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--max-mb", type=float, default=4.0)
    p.add_argument("--title", default="")
    p.add_argument("--cache", help="directory to keep rendered frames in (re-composing then skips rendering)")
    a = p.parse_args()
    sides = []
    for run, eid, label in ((a.left_run, a.left_episode, a.left_label), (a.right_run, a.right_episode, a.right_label)):
        frames, info, instr = episode_frames(Path(run), eid, blender=not a.mujoco, every=a.every, samples=a.samples,
                                             cache=Path(a.cache) if a.cache else None)
        sides.append((frames, info, label))
        print(json.dumps({"label": label, **info}))
    n = max(len(s[0]) for s in sides)
    hold = int(2.5 * FPS)
    W, H = 2 * PW, TOP + PH + BOTTOM
    renderer = "MuJoCo" if a.mujoco else "Blender Cycles (CPU)"
    title = a.title or f"{sides[0][1]['task'].split('/')[-1].replace('manip_', '')}, held-out seed {sides[0][1]['seed']}"
    video = []
    for i in range(n + hold):
        canvas = Image.new("RGB", (W, H), (14, 18, 26))
        d = ImageDraw.Draw(canvas)
        d.text((16, 8), title, font=F_HEAD, fill=(255, 255, 255))
        sub = f"“{instr}”"
        while d.textlength(sub, font=F_TEXT) > W - 32:
            sub = sub[:-4].rstrip() + "…"
        d.text((16, 34), sub, font=F_TEXT, fill=(200, 210, 225))
        for k, (frames, info, label) in enumerate(sides):
            x = k * PW
            size = 22
            while size > 12 and d.textlength(label, font=font("Inter-Bold.ttf", size)) > PW - 28:
                size -= 1                                  # the whole label, in the size that fits the panel
            d.text((x + 14, 62), label, font=font("Inter-Bold.ttf", size), fill=(255, 196, 92) if k else (170, 200, 255))
            j = min(i, len(frames) - 1)
            done = i >= len(frames) - 1
            canvas.paste(panel(frames[j], j, info["steps"], done, info["success"]), (x, TOP))
            status = f"step {j}/{info['steps']}" + ("  (finished: last frame held)" if done and i < n else "")
            d.text((x + 16, TOP + PH + 12), status, font=F_TEXT, fill=(200, 210, 225))
        d.text((W - 16, TOP + PH + 12), f"{renderer}, re-simulated from the traces", font=F_TEXT,
               fill=(140, 150, 170), anchor="ra")
        video.append(np.asarray(canvas))
    import make_videos

    make_videos.W, make_videos.H = W, H                    # encode() reads the frame size from the module
    size = encode(video, Path(a.out), a.max_mb)
    print(json.dumps({"video": a.out, "bytes": size, "seconds": round(len(video) / FPS, 1), "frames": len(video)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
