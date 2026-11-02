"""Trace replay: record episodes, re-simulate them from the recorded actions alone, and compare bit for bit.

What it shows
    1. Recording: a language + memory agent (composable.object_reach: rule-based colour chooser, SQLite spatial
       memory) on the MuJoCo table, where the pads are visible only for the first 5 steps. Every episode is an
       event-sourced trace; the example counts what is in it (events per type, bytes on disk).
    2. Replay: for every episode, a fresh simulator is reset with the recorded task and seed and stepped with the
       recorded `action` events only (no agent). Every observation and step result is compared with the trace:
       the maximum absolute deviation is the measured result (0.0 means bit-exact).
    3. Sensitivity: the same replay with the first action changed by 1e-9, to show that a deviation of exactly
       0.0 is a real check and not a comparison that cannot fail.
    4. A replay video: one episode re-simulated and rendered from the trace (needs an OpenGL backend for MuJoCo;
       the example uses the user-space OSMesa install when there is one, and skips the GIF otherwise).

Command (from the harness repository root)
    uv run python examples/core/03_trace_replay.py [--quick] [--out DIR] [--no-render]

Typical runtime: about 8 s (--quick: about 3 s) on a 128-core CPU, no GPU (OSMesa rendering included).
Requires: hylo-env-mujoco (hylo-bench); hylo-agents with hylo-reasoners and hylo-memory; Pillow and an OpenGL
backend (MUJOCO_GL=egl or osmesa) for the GIF (optional).
Output: <out>/result.json, <out>/replay.gif, the recorded run under <out>/runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


def ensure_render_env() -> None:
    """MuJoCo offscreen rendering on a machine without a GPU: use the user-space OSMesa install if present.
    LD_LIBRARY_PATH is read only at process start, so re-exec once with the environment set."""
    if os.environ.get("MUJOCO_GL") or "--no-render" in sys.argv or not (USER_MESA / "libOSMesa.so.8").exists():
        return
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
           "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


ensure_render_env()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

from hylo.core.config import ExperimentConfig  # noqa: E402
from hylo.core.registry import registry  # noqa: E402
from hylo.core.trace import EpisodeTrace, TraceStore  # noqa: E402
from hylo.core.types import Action  # noqa: E402
from hylo.eval.experiment import run_experiment  # noqa: E402

SLUG = "trace_replay"
ROOT = Path(__file__).resolve().parents[2]
ENV = {"type": "mujoco.table_reach"}


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def max_dev(a: dict, b: dict) -> float:
    """Largest absolute difference between two observation states (inf if keys or shapes differ)."""
    if set(a) != set(b):
        return float("inf")
    worst = 0.0
    for k in a:
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        if x.shape != y.shape:
            return float("inf")
        if x.size:
            worst = max(worst, float(np.max(np.abs(x - y))))
    return worst


def replay(tr: EpisodeTrace, instance, *, perturb: float = 0.0, frames: list | None = None,
           frame_every: int = 2) -> dict:
    """Re-simulate one episode from its recorded actions and compare with the recorded observations."""
    env = registry.build("envs", ENV)
    try:
        rec_obs = [e["payload"]["state"] for e in tr.of_type("observation")]
        rec_res = list(tr.of_type("step.result"))
        actions = [np.asarray(e["payload"]["data"], dtype=float) for e in tr.of_type("action")]
        if perturb:
            actions[0] = actions[0] + np.array([perturb] + [0.0] * (len(actions[0]) - 1))
        obs = env.reset(instance)
        worst = max_dev(obs.state, rec_obs[0])
        reward_dev, outcome_same = 0.0, True
        decisions = iter(e for e in tr.of_type("agent.decision"))
        caption, next_dec = "", next(decisions, None)
        if frames is not None:
            frames.append((env.render("front", 360, 270), 0, caption))
        for i, a in enumerate(actions):
            r = env.step(Action(data=a))
            worst = max(worst, max_dev(r.observation.state, rec_obs[i + 1]))
            rp = rec_res[i]["payload"]
            reward_dev = max(reward_dev, abs((r.reward or 0.0) - (rp["reward"] or 0.0)))
            outcome_same &= (r.success, r.terminated, r.truncated) == (rp["success"], rp["terminated"], rp["truncated"])
            if frames is not None:
                while next_dec is not None and next_dec["seq"] < rec_res[i]["seq"]:
                    caption, next_dec = next_dec["payload"].get("rationale", ""), next(decisions, None)
                if (i + 1) % frame_every == 0 or i + 1 == len(actions):
                    frames.append((env.render("front", 360, 270), i + 1, caption))
        tip = np.asarray(r.observation.state["tip_pos"]) - np.asarray(rec_obs[len(actions)]["tip_pos"])
        return {"steps": len(actions), "max_obs_dev": worst, "max_reward_dev": reward_dev,
                "outcome_same": outcome_same, "final_tip_dev_m": float(np.linalg.norm(tip))}
    finally:
        env.close()


def write_gif(frames: list, title: str, status: str, path: Path) -> int:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
        except OSError:
            return ImageFont.load_default()

    imgs = []
    n = frames[-1][1]
    for k, (rgb, step, caption) in enumerate(frames):
        img = Image.fromarray(rgb)
        w, h = img.size
        canvas = Image.new("RGB", (w, h + 70), (24, 26, 30))
        canvas.paste(img, (0, 30))
        d = ImageDraw.Draw(canvas)
        d.text((8, 7), title, font=font(13, True), fill=(255, 255, 255))
        if caption:
            d.text((8, h + 34), ("reasoner: " + caption)[:64], font=font(11), fill=(120, 220, 150))
        d.text((8, h + 52), f"step {step}/{n}   re-simulated from the recorded actions only", font=font(11),
               fill=(190, 190, 190))
        if k == len(frames) - 1:
            d.text((w - 8, 8), status, font=font(13, True), fill=(120, 220, 150), anchor="ra")
        imgs.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs += [imgs[-1]] * 8                                  # hold the last frame
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    ap.add_argument("--no-render", action="store_true", help="skip the replay GIF")
    a = ap.parse_args()
    try:
        registry.reference("envs", "mujoco.table_reach")
    except KeyError:
        sys.exit("'mujoco.table_reach' is not registered: install hylo-env-mujoco (hylo-bench) into this venv")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_val, workers = (3, 3) if a.quick else (30, 8)

    print(f"== 1. record: language + memory agent on the MuJoCo table, {n_val} seeds x 3 tasks")
    cfg = ExperimentConfig(
        name="core-trace-replay", benchmark={"type": "mujoco/table-v0", "params": {"validation": n_val}},
        env=ENV, agent={"type": "composable.object_reach",
                        "params": {"kp": 0.3, "chooser": {"type": "chooser.rule_color"},
                                   "memory": {"type": "sqlite", "params": {"path": ":memory:"}}}},
        safety={"max_delta": 0.5}, workers=workers)
    t0 = time.perf_counter()
    res = run_experiment(cfg, run_root=out / "runs")
    rec_s = time.perf_counter() - t0
    store = TraceStore(res.run_dir)
    counts: Counter = Counter()
    for s in res.log["samples"]:
        counts.update(e["type"] for e in store.read(s["episode_id"]).events)
    size = sum(p.stat().st_size for p in (res.run_dir / "episodes").glob("*.jsonl"))
    n_ep, n_steps = len(res.log["samples"]), sum(s["steps"] for s in res.log["samples"])
    print(f"  {n_ep} episodes, {n_steps} steps, success {res.metrics['success_mean']:.2f}, recorded in {rec_s:.1f} s; "
          f"traces: {size / 1024:.0f} KiB ({size / n_steps:.0f} B per step)")
    table([[t, c, f"{c / n_ep:.1f}"] for t, c in counts.most_common()], ["event type", "total", "per episode"])

    print("\n== 2. replay every episode from its recorded actions")
    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    instances = {i.key: i for i in bench.instances("validation")}
    t0 = time.perf_counter()
    reps = [replay(store.read(s["episode_id"]), instances[s["key"]]) for s in res.log["samples"]]
    rep_s = time.perf_counter() - t0
    worst = max(r["max_obs_dev"] for r in reps)
    worst_rew = max(r["max_reward_dev"] for r in reps)
    same = all(r["outcome_same"] for r in reps)
    bit_exact = sum(r["max_obs_dev"] == 0.0 and r["max_reward_dev"] == 0.0 for r in reps)
    table([[n_ep, f"{bit_exact}/{n_ep}", f"{worst:g}", f"{worst_rew:g}", same, f"{n_steps / rep_s:.0f}"]],
          ["episodes", "bit-exact", "max |obs dev|", "max |reward dev|", "same outcomes", "replay steps/s"])

    print("\n== 3. sensitivity: first action changed by 1e-9")
    pert = [replay(store.read(s["episode_id"]), instances[s["key"]], perturb=1e-9) for s in res.log["samples"]]
    pworst = max(r["max_obs_dev"] for r in pert)
    ptip = float(np.median([r["final_tip_dev_m"] for r in pert]))
    print(f"  max |obs dev| = {pworst:.3g} (median final fingertip deviation {ptip:.3g} m); "
          f"episodes still bit-exact: {sum(r['max_obs_dev'] == 0.0 for r in pert)}/{n_ep}")

    gif = None
    if not a.no_render:
        print("\n== 4. replay video")
        pick = next((s for s in res.log["samples"] if s["task_id"] == "mujoco/table_push" and s["status"] == "success"),
                    res.log["samples"][0])
        tr = store.read(pick["episode_id"])
        try:
            frames: list = []
            replay(tr, instances[pick["key"]], frames=frames, frame_every=1)
            instr = tr.first("observation")["payload"]["instruction"]
            nbytes = write_gif(frames, f"“{instr}” (shoved at step 12)" if "push" in pick["task_id"] else
                               f"“{instr}”", pick["status"].upper(), out / "replay.gif")
            gif = {"path": str(out / "replay.gif"), "episode": pick["episode_id"], "frames": len(frames),
                   "bytes": nbytes}
            print(f"  {out / 'replay.gif'}: {len(frames)} frames, {nbytes / 1024:.0f} KiB")
        except Exception as exc:  # noqa: BLE001 - no OpenGL backend: the measured results stand without the GIF
            print(f"  skipped: cannot render ({type(exc).__name__}: {exc})")

    result = {"example": SLUG, "quick": a.quick, "run_dir": str(res.run_dir),
              "recording": {"episodes": n_ep, "steps": n_steps, "success_mean": res.metrics["success_mean"],
                            "trace_bytes": size, "bytes_per_step": size / n_steps, "events_by_type": dict(counts),
                            "wall_s": rec_s},
              "replay": {"bit_exact_episodes": bit_exact, "max_obs_dev": worst, "max_reward_dev": worst_rew,
                         "same_outcomes": same, "steps_per_s": n_steps / rep_s},
              "perturbed_1e-9": {"max_obs_dev": pworst, "median_final_tip_dev_m": ptip,
                                 "bit_exact_episodes": sum(r["max_obs_dev"] == 0.0 for r in pert)},
              "gif": gif}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {out / 'result.json'}")
    return 0 if bit_exact == n_ep and same else 1


if __name__ == "__main__":
    sys.exit(main())
