"""Grasp benchmark: the camera-only program dex_lift.py on mujoco/dexterous-v0 for every hand and object, through
`hylo run` (agent code.program, the kernel and the safety shield, cameras with segmentation, no privileged state).
Success = the object held 8 cm above its start. Writes OUT/results.json and OUT/results.md (success per hand and
object); with --videos, one re-simulated video per hand showing a successful lift of each object (from the traces,
by examples/llm/make_videos.py), at most 4 MB each.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/dexterous/grasp_benchmark.py --out runs/dex-grasps \\
        [--hands g1_right allegro_right] [--objects box cylinder sphere mug rock] [--episodes 10] [--videos]

--episodes N runs validation seeds 0..N-1 of each task; --finger-gain scales the finger servos (Menagerie gains: 1).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
HANDS = ("g1_right", "allegro_right")
OBJECTS = ("box", "cylinder", "sphere", "mug", "rock")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def config(hand: str, obj: str, episodes: int, out: Path, finger_gain: float, image_every: int) -> dict:
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"dex-lift-{hand}-{obj}".replace("_", "-"),
        "benchmark": {"type": "mujoco/dexterous-v0", "params": {"tasks": [f"lift_{obj}"], "episodes": episodes},
                      "split": "validation"},
        "env": {"type": "mujoco.dex_table", "params": {"hand": hand, "cameras": ["front", "side", "back"],
                                                       "segmentation": True, "finger_gain": finger_gain}},
        "agent": {"type": "code.program", "params": {"source": (HERE / "dex_lift.py").read_text(), "timeout_s": 900}},
        "trace": {"record_privileged": False, "image_every": image_every},
        "output_dir": str(out),
        "tags": ["dexterous-grasps", hand, obj],
    }


def concat_videos(paths: list[Path], out: Path, max_mb: float = 4.0) -> int:
    """Clips one after the other (same frame size) as one MP4 under max_mb."""
    from hylo.codeagents.replay_geom import read_video, write_video

    frames = [f for p in paths for f in read_video(p)]
    return write_video(frames, out, fps=20.0, max_bytes=int(max_mb * 1_000_000))["bytes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/dex-grasps")
    ap.add_argument("--hands", nargs="*", default=list(HANDS))
    ap.add_argument("--objects", nargs="*", default=list(OBJECTS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4, help="episode workers per run")
    ap.add_argument("--parallel", type=int, default=4, help="runs at once")
    ap.add_argument("--finger-gain", type=float, default=1.0)
    ap.add_argument("--videos", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    rp = _load(EXAMPLES / "programs" / "run_programs.py", "hylo_run_programs")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    pending = [(h, o) for h in a.hands for o in a.objects]
    running, results, t0 = {}, {}, time.time()
    while pending or running:
        while pending and len(running) < a.parallel:
            h, o = pending.pop(0)
            key = f"{h}/{o}"
            cfg = out / "configs" / f"{h}-{o}.yaml"
            cfg.write_text(yaml.safe_dump(config(h, o, a.episodes, out, a.finger_gain, 0), sort_keys=False))
            log = open(cfg.with_suffix(".log"), "w")
            cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(cfg), "--set", f"workers={a.workers}"]
            running[key] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env), log)
            print(f"started {key}", flush=True)
        for key, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            stdout = proc.stdout.read() if proc.stdout else ""
            log.close()
            del running[key]
            try:
                info = json.loads(stdout[stdout.index("{"):])
            except ValueError:
                results[key] = {"error": f"hylo run exit {proc.returncode}"}
                print(f"{key}: failed (see configs/)", flush=True)
                continue
            eps = rp.episode_stats(Path(info["run_dir"]))
            results[key] = {"run_dir": info["run_dir"], "success": info.get("success_mean"), "n": len(eps),
                            "episodes": eps}
            print(f"{key}: success {info.get('success_mean'):.2f} over {len(eps)} episodes", flush=True)
        time.sleep(2)
    if a.videos:
        mv = _load(EXAMPLES / "llm" / "make_videos.py", "hylo_make_videos")
        for h in a.hands:
            clips = []
            for o in a.objects:
                r = results.get(f"{h}/{o}", {})
                ok = [e for e in r.get("episodes", []) if e["success"]]
                if not ok:
                    continue
                log = json.loads((Path(r["run_dir"]) / "evallog.json").read_text())
                sample = next(s for s in log["samples"] if s["seed"] == ok[0]["seed"])
                ep_id = Path(sample["trace"]).stem
                clip = out / "clips" / f"{h}-{o}.mp4"
                mv.make(Path(r["run_dir"]), ep_id, clip, workflow="dex_lift (code.program)", model="no model",
                        max_mb=0.75)
                clips.append(clip)
            if clips:
                size = concat_videos(clips, out / f"grasps_{h}.mp4")
                results.setdefault("videos", {})[h] = {"path": str(out / f"grasps_{h}.mp4"), "bytes": size}
    (out / "results.json").write_text(json.dumps({"benchmark": "mujoco/dexterous-v0", "episodes": a.episodes,
                                                  "finger_gain": a.finger_gain, "wall_s": round(time.time() - t0, 1),
                                                  "results": results}, indent=1))
    lines = ["# Dexterous lift (mujoco/dexterous-v0, validation)", "", "| hand | " + " | ".join(a.objects) + " |",
             "|---|" + "---|" * len(a.objects)]
    for h in a.hands:
        cells = []
        for o in a.objects:
            r = results.get(f"{h}/{o}", {})
            cells.append(f"{sum(e['success'] for e in r['episodes'])}/{r['n']}" if "episodes" in r else "error")
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
