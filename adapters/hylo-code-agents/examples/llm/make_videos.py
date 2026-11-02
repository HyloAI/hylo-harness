"""Showcase videos of recorded episodes: re-simulated from the trace's actions, captioned from the trace.

The episode is replayed bit for bit (a fresh simulator, reset with the recorded task and seed, stepped with the
recorded `action` events only; the replayed outcome is checked against the recorded one). The front camera is rendered
at 960x720 with MuJoCo (default) or, with --blender, by the `render` tool (Blender Cycles on the CPU) from the replayed
joint positions. Captions come from the trace: for `code.llm` the model's turn number, a short excerpt of its thought
and code, the toolkit verbs it called and what the block printed; each turn without motion is held for a moment so
the viewer can read it. For `code.program` (a frozen program, no model) the program's motions and its printed output.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python harness/adapters/hylo-code-agents/examples/llm/make_videos.py RUN_DIR EPISODE_ID OUT.mp4 \\
        [--workflow "code.llm zero-shot"] [--model qwen3.5:35b-a3b] [--blender] [--max-mb 6]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hylo.core import codec

W, H, FPS = 960, 720, 20
FONTS = Path.home() / ".cache" / "hylo" / "fonts"
VERB = re.compile(r"robot\.(\w+)\(")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for f in (FONTS / name, Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSansMono.ttf" if "Mono" in name
                                                                         else "DejaVuSans.ttf")):
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


F_HEAD, F_TEXT, F_BOLD = font("Inter-SemiBold.ttf", 19), font("Inter-Regular.ttf", 16), font("Inter-Bold.ttf", 22)
F_CODE, F_BIG = font("JetBrainsMono-Regular.ttf", 14), font("InterDisplay-Bold.ttf", 54)


# ---------------------------------------------------------------- reading the trace

def read_trace(run_dir: Path, episode_id: str) -> dict:
    ep: dict = {"actions": [], "turns": [], "chunks": [], "program": None, "start": None, "end": None}
    with open(run_dir / "episodes" / f"{episode_id}.jsonl") as f:
        for line in f:
            head = line[:48]
            if '"observation"' in head or '"safety.review"' in head or '"step.result"' in head:
                continue
            ev = json.loads(line)
            t, p = ev["type"], ev["payload"]
            if t == "action":
                ep["actions"].append(codec.decode(p["data"]))
            elif t == "agent.chunk":
                ep["chunks"].append(len(ep["actions"]))
            elif t == "agent.code.turn":
                ep["turns"].append({k: p.get(k) for k in ("turn", "step", "reply", "code", "output", "error",
                                                          "latency_s", "tokens_in", "tokens_out", "model_error")})
            elif t == "agent.code.program":
                ep["program"] = p
            elif t == "agent.tool.turn":                  # tools.llm: one caption per model call
                calls = [f"{c['name']}({_args(c.get('arguments'))})" for c in p.get("tool_calls") or []]
                ep["turns"].append({"turn": p.get("call"), "step": p.get("step"), "reply": p.get("text"),
                                    "code": "\n".join(calls), "output": "", "error": None,
                                    "latency_s": p.get("latency_s"), "tokens_out": p.get("tokens_out"),
                                    "model_error": p.get("model_error")})
            elif t == "agent.tool.result" and ep["turns"]:
                ep["turns"][-1]["output"] = (ep["turns"][-1]["output"] + "\n" + _condense(p)).strip()
            elif t == "episode.start":
                ep["start"] = p | {"wall": ev["wall_time"]}
            elif t == "episode.end":
                ep["end"] = p | {"wall": ev["wall_time"]}
    return ep


def _args(a) -> str:
    if not isinstance(a, dict):
        return str(a)
    return ", ".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in a.items())


def _condense(p: dict) -> str:
    """A tool result in one line (the last line printed is what the caption shows)."""
    try:
        from hylo.codeagents.watch import condense

        text, ok = condense(p.get("name", ""), p.get("result", ""))
        return f"{p.get('name')}: {text}" if ok else f"{p.get('name')} failed: {text}"
    except Exception:  # noqa: BLE001
        return str(p.get("result", ""))[:110]


def instance_for(cfg, start: dict):
    from hylo.core.registry import registry

    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    for inst in bench.instances(start["split"]):
        if inst.spec.id == start["task_id"] and inst.seed == start["seed"]:
            return inst
    raise LookupError(f"no instance {start['task_id']} seed {start['seed']} in split {start['split']}")


def replay(cfg, inst, actions: list, *, every: int = 1, render: bool = True) -> tuple[list, list, dict]:
    """Frames (front camera, W x H) and qpos after reset and after every step; the replayed outcome."""
    from hylo.core.registry import registry
    from hylo.core.types import Action

    params = {**cfg.env.params, "cameras": [], "segmentation": False, "width": W, "height": H}
    env = registry.build("envs", {"type": cfg.env.type, "params": params})
    frames, qpos = [], []
    try:
        env.reset(inst)
        frames.append(env.render("front") if render else None)
        qpos.append(env.data.qpos.copy())
        res = None
        for i, a in enumerate(actions):
            res = env.step(Action(data=np.asarray(a, dtype=float)))
            qpos.append(env.data.qpos.copy())
            if render and ((i + 1) % every == 0 or i + 1 == len(actions)):
                frames.append(env.render("front"))
            if res.terminated or res.truncated:
                break
        out = {"success": bool(res.success) if res is not None else False, "steps": len(qpos) - 1}
        mjb = None
        if not render:                                  # the episode's model for an offline renderer
            mjb = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False).name
            env.reset(inst)
            m = env.model
            for name, info in env.items.items():       # fixtures are mocap bodies: bake their episode pose
                if "mocap" in info:
                    m.body_pos[info["body"]] = env.data.mocap_pos[info["mocap"]]
                    m.body_quat[info["body"]] = env.data.mocap_quat[info["mocap"]]
            import mujoco

            mujoco.mj_saveModel(m, mjb, None)
        out["mjb"] = mjb
        return frames, qpos, out
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------- captions

def thought(reply: str | None) -> str:
    text = (reply or "").split("```")[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text.replace("Thought:", "").replace("**", "")).strip()
    return text


def code_lines(code: str | None, n: int = 4) -> list[str]:
    lines = [ln.rstrip() for ln in (code or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    key = [ln for ln in lines if re.search(r"robot\.(run|plan|place_at|grasps|grasp_at|Goal)|finish\(|def solve", ln)]
    picked = (key or lines)[:n]
    indent = min((len(ln) - len(ln.lstrip()) for ln in picked), default=0)
    picked = [ln[indent:] for ln in picked]
    return [ln if len(ln) <= 104 else ln[:101] + "..." for ln in picked]


def last_line(text: str | None) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1][:110] if lines else ""


def turn_caption(t: dict, n_turns: int) -> dict:
    verbs = list(dict.fromkeys(VERB.findall(t.get("code") or "")))
    out = last_line(t.get("output"))
    res = ("error: " + last_line(t.get("error"))) if t.get("error") else (f"printed: {out}" if out else "")
    if t.get("model_error"):
        res = "model error: " + str(t["model_error"])[:90]
    return {"title": f"Turn {t['turn']}/{n_turns}", "thought": thought(t.get("reply")), "code": code_lines(t.get("code")),
            "verbs": verbs, "result": res,
            "meta": f"model reply after {t.get('latency_s') or 0:.0f} s (incl. queue), {t.get('tokens_out') or 0} tokens"}


def overlay(img: np.ndarray, header: str, sub: str, cap: dict | None, step: int, steps: int, *,
            banner: tuple[str, tuple] | None = None) -> np.ndarray:
    im = Image.fromarray(img).convert("RGBA")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, 0, W, 62], fill=(14, 18, 26, 200))
    d.text((16, 9), header, font=F_HEAD, fill=(255, 255, 255, 255))
    while sub and d.textlength(sub, font=F_TEXT) > W - 150:          # keep clear of the step counter
        sub = sub[:-4].rstrip() + "…"
    d.text((16, 35), sub, font=F_TEXT, fill=(200, 210, 225, 255))
    d.text((W - 16, 37), f"step {step}/{steps}", font=F_TEXT, fill=(200, 210, 225, 255), anchor="ra")
    if cap is not None:
        n_thought = min(2, len(textwrap.wrap(cap.get("thought") or "", 112)))
        n_code = len(cap.get("code", [])[:4])
        has_tail = bool(cap.get("verbs") or cap.get("result"))
        top = H - (48 + 21 * n_thought + 18 * n_code + (30 if has_tail else 6))
        d.rectangle([0, top, W, H], fill=(14, 18, 26, 214))
        d.text((16, top + 10), cap["title"], font=F_BOLD, fill=(255, 196, 92, 255))
        if cap.get("meta"):
            d.text((W - 16, top + 14), cap["meta"], font=F_TEXT, fill=(160, 170, 190, 255), anchor="ra")
        y = top + 40
        if cap.get("thought"):
            for ln in textwrap.wrap(cap["thought"], 112)[:2]:
                d.text((16, y), ln, font=F_TEXT, fill=(235, 238, 245, 255))
                y += 21
        for ln in cap.get("code", [])[:4]:
            d.text((24, y), ln, font=F_CODE, fill=(140, 220, 160, 255))
            y += 18
        tail = []
        if cap.get("verbs"):
            tail.append("tools: " + ", ".join(cap["verbs"][:9]))
        if cap.get("result"):
            tail.append(cap["result"])
        if tail:
            d.text((16, H - 26), "   ".join(tail)[:150], font=F_TEXT, fill=(170, 200, 255, 255))
    if banner is not None:
        text, colour = banner
        d.rectangle([W // 2 - 260, H // 2 - 60, W // 2 + 260, H // 2 + 60], fill=(*colour, 225))
        d.text((W // 2, H // 2), text, font=F_BIG, fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


# ---------------------------------------------------------------- assembling

def timeline(ep: dict, steps: int, hold_s: float = 1.6) -> list[tuple[int, dict | None, int]]:
    """(frame index, caption, repeat count): turns without motion are held so they can be read."""
    out: list[tuple[int, dict | None, int]] = []
    turns = [t for t in ep["turns"] if t.get("turn")]
    if turns:
        n = len(turns)
        for i, t in enumerate(turns):
            s0 = int(t.get("step") or 0)
            s1 = int(turns[i + 1].get("step") or s0) if i + 1 < n else steps
            cap = turn_caption(t, n)
            out.append((min(s0, steps), cap, int(hold_s * FPS)))
            out.extend((s, cap, 1) for s in range(s0 + 1, min(s1, steps) + 1))
        return out
    prog = ep.get("program") or {}
    marks = sorted(set(ep["chunks"])) or [0]
    printed = [ln for ln in (prog.get("output") or "").strip().splitlines() if ln.strip()]
    for k, s0 in enumerate(marks):
        s1 = marks[k + 1] if k + 1 < len(marks) else steps
        cap = {"title": f"Motion {k + 1}/{len(marks)}", "thought": "frozen program, no model: solve(robot) runs "
               "the toolkit (look, find, objects, grasps, plan, run) and every motion goes through the kernel",
               "code": printed[:4], "verbs": [], "result": "", "meta": f"program {prog.get('program', '')}"}
        out.extend((s, cap, 1) for s in range(s0, min(s1, steps) + 1))
    return out


def encode(frames: list[np.ndarray], path: Path, max_mb: float) -> int:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    seconds = len(frames) / FPS
    kbps = int(max_mb * 8 * 1024 * 0.92 / max(seconds, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    for crf in (23, 27, 31, 35):
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
               str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-maxrate",
               f"{kbps}k", "-bufsize", f"{2 * kbps}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
        if path.stat().st_size <= max_mb * 1024 * 1024:
            break
    return path.stat().st_size


def blender_frames(mjb: str, qpos: list, every: int, samples: int, out_dir: Path) -> list[np.ndarray]:
    from hylo.tools.render import gl_env
    from hylo.tools.runner import run_worker

    traj = out_dir / "traj.npz"
    q = np.asarray(qpos)[::every]
    np.savez(traj, qpos=q, fps=FPS / every)
    req = {"backend": "blender", "width": W, "height": H, "fps": FPS / every, "camera": "front", "samples": samples,
           "max_frames": len(q), "keep_frames": True}
    res = run_worker("render", req, {"scene": Path(mjb), "trajectory": traj}, workdir=out_dir / "render", env=gl_env())
    files = sorted((out_dir / "render").rglob("frames/*.png"))
    if res.outputs.get("backend") != "blender" or len(files) < len(q):
        raise RuntimeError(f"blender render failed: {res.outputs}")
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def make(run_dir: Path, episode_id: str, out: Path, *, workflow: str, model: str, blender: bool = False,
         samples: int = 24, max_mb: float = 6.0, every: int = 1, cache: str | None = None) -> dict:
    from hylo.core.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(json.loads((run_dir / "manifest.json").read_text())["config"])
    ep = read_trace(run_dir, episode_id)
    inst = instance_for(cfg, ep["start"])
    frames, qpos, res = replay(cfg, inst, ep["actions"], render=not blender)
    rec = ep["end"] or {}
    if res["success"] != bool(rec.get("success")) or res["steps"] != rec.get("steps", res["steps"]):
        print(f"warning: replay gave {res}, the trace says success={rec.get('success')} steps={rec.get('steps')}")
    steps = res["steps"]
    if blender:
        key = Path(cache) / f"{episode_id}-e{every}-s{samples}.npz" if cache else None
        if key is not None and key.exists():                     # frames rendered before (e.g. by compare_videos)
            with np.load(key) as z:
                bf = list(z["frames"])
        else:
            with tempfile.TemporaryDirectory(prefix="hylo-vid-") as tmp:
                bf = blender_frames(res["mjb"], qpos, every, samples, Path(tmp))
            if key is not None:
                key.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(key, frames=np.asarray(bf))
        frames = [bf[min(i // every, len(bf) - 1)] for i in range(steps + 1)]
    header = f"{workflow}  ·  {model}  ·  {inst.spec.id.split('/')[-1].replace('manip_', '')} (seed {inst.seed})"
    instr = ep["start"].get("instruction")
    tl = timeline(ep, steps)
    turns = [t for t in ep["turns"] if t.get("turn")]
    sub = f"“{instr or task_instruction(cfg, inst)}”"
    video = []
    for idx, cap, rep in tl:
        img = overlay(frames[min(idx, len(frames) - 1)], header, sub, cap, idx, steps)
        video.extend([img] * rep)
    ok = bool(rec.get("success"))
    wall = (rec.get("wall", 0) - ep["start"].get("wall", 0)) if rec else 0
    stats = (f"{len(turns)} turns, {sum(t.get('tokens_out') or 0 for t in turns)} tokens out, "
             f"{wall / 60:.1f} min wall" if turns else f"{steps} steps, {wall:.0f} s wall, no model")
    end = {"title": "Outcome (judged by the task, never shown to the model)",
           "thought": f"{stats}. Video: {render_name(blender)}.", "code": [],
           "verbs": [], "result": str(rec.get("termination_reason", "")), "meta": ""}
    last = frames[-1]
    video.extend([overlay(last, header, sub, end, steps, steps,
                          banner=("SUCCESS", (34, 139, 84)) if ok else ("FAILURE", (178, 48, 48)))] * int(3 * FPS))
    size = encode(video, out, max_mb)
    info = {"video": str(out), "bytes": size, "seconds": round(len(video) / FPS, 1), "success": ok, "steps": steps,
            "turns": len(turns), "task": inst.spec.id, "seed": inst.seed, "workflow": workflow, "model": model,
            "renderer": render_name(blender), "replay": res | {"mjb": None}}
    print(json.dumps(info))
    with open(out.parent / "index.jsonl", "a") as f:          # one record per video, for the README index
        f.write(json.dumps({**info, "video": out.name}) + "\n")
    return info


def render_name(blender: bool) -> str:
    return "Blender Cycles (CPU), re-simulated from the trace" if blender else "MuJoCo, re-simulated from the trace"


def task_instruction(cfg, inst) -> str:
    """The episode's natural-language instruction (the environment realises it at reset)."""
    from hylo.core.registry import registry

    env = registry.build("envs", {"type": cfg.env.type, "params": {**cfg.env.params, "cameras": [],
                                                                    "segmentation": False}})
    try:
        return env.reset(inst).instruction or inst.spec.instruction
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("episode_id")
    p.add_argument("out")
    p.add_argument("--workflow", default="code.llm")
    p.add_argument("--model", default="qwen3.5:35b-a3b")
    p.add_argument("--blender", action="store_true")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--every", type=int, default=2, help="Blender: render every n-th step (frames repeated)")
    p.add_argument("--max-mb", type=float, default=6.0)
    p.add_argument("--cache", help="directory to keep Blender frames in (shared with compare_videos.py)")
    a = p.parse_args()
    make(Path(a.run_dir), a.episode_id, Path(a.out), workflow=a.workflow, model=a.model, blender=a.blender,
         samples=a.samples, max_mb=a.max_mb, every=a.every, cache=a.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
