"""Running one recorded episode of a scene (fixed-format or free) and writing what it produced.

    env = FrameRecorder(registry.build("envs", {...}), "front")
    result = run_recorded_episode(env, agent, task, out_dir / "solve", seed=0, video=out_dir / "front.mp4")

`run_recorded_episode` runs the kernel's EpisodeRunner with a trace store, writes the turns of a code agent to
turns.jsonl, summarises the outcome (status, steps, turns, usage, model and wall time) and, with `video`, encodes the
recorded camera frames as an MP4. It closes the agent and the environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hylo.core.task import TaskInstance, TaskSpec

VIDEO_MAX_BYTES = 3_000_000


class FrameRecorder:
    """An Environment wrapper that keeps one camera's frames (reset and every step); everything else passes."""

    def __init__(self, env: Any, camera: str = "front") -> None:
        self._env, self.camera, self.frames = env, camera, []

    def _keep(self, obs: Any) -> None:
        img = obs.images.get(self.camera) if obs is not None else None
        if img is not None:
            self.frames.append(np.asarray(img).copy())

    def spec(self) -> Any:
        return self._env.spec()

    def reset(self, task: TaskInstance) -> Any:
        obs = self._env.reset(task)
        self._keep(obs)
        return obs

    def step(self, action: Any) -> Any:
        r = self._env.step(action)
        self._keep(r.observation)
        return r

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def write_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: float = 20.0,
              max_bytes: int = VIDEO_MAX_BYTES) -> dict[str, Any]:
    """H.264 MP4 of RGB frames (imageio-ffmpeg); re-encoded at lower quality / frame rate until <= max_bytes."""
    import imageio_ffmpeg

    path = Path(path)
    frames = list(frames)
    if not frames:
        return {"written": False, "reason": "no frames"}
    h, w = frames[0].shape[:2]
    for crf, stride in ((26, 1), (31, 1), (34, 2), (38, 3), (42, 4)):
        sel = frames[::stride]
        gen = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps / stride, codec="libx264", quality=None,
                                          pix_fmt_out="yuv420p", output_params=["-crf", str(crf), "-preset", "slow"],
                                          macro_block_size=8)
        gen.send(None)
        for f in sel:
            gen.send(np.ascontiguousarray(f[..., :3], dtype=np.uint8))
        gen.close()
        size = path.stat().st_size
        if size <= max_bytes:
            return {"written": True, "path": str(path), "bytes": size, "frames": len(sel), "fps": fps / stride,
                    "crf": crf}
    path.unlink(missing_ok=True)
    return {"written": False, "reason": f"over {max_bytes} bytes at every quality"}


def run_recorded_episode(env: FrameRecorder, agent: Any, task: TaskSpec, out: Path, *, seed: int = 0,
                         video: Path | None = None) -> dict[str, Any]:
    """Run one episode of `agent` on `env` (traced under out/trace, code turns in out/turns.jsonl) and return the
    outcome: status, success, steps, reason, turns, errors, usage, wall_s, model_s, trace, episode_id, and the
    `video` record when asked. Closes the agent and the environment."""
    from hylo.core.kernel import EpisodeRunner
    from hylo.core.trace import TraceStore

    store = TraceStore(out / "trace")
    t0 = time.monotonic()
    try:
        rec = EpisodeRunner(env, agent, trace=store).run(TaskInstance(spec=task, seed=seed, split="scene"))
    finally:
        agent.close()
    wall = time.monotonic() - t0
    o = rec.outcome
    tr = store.read(o.episode_id)
    turns = [e["payload"] for e in tr.of_type("agent.code.turn")]
    with open(out / "turns.jsonl", "w") as fh:
        for t in turns:
            fh.write(json.dumps({k: v for k, v in t.items() if k not in ("images", "prompt")}, default=str) + "\n")
    summary = (tr.last("agent.code.summary") or {}).get("payload", {})
    result: dict[str, Any] = {
        "status": o.status, "success": o.success, "steps": o.steps, "reason": o.termination_reason,
        "turns": len(turns), "errors": sum(1 for t in turns if t.get("error")), "usage": summary.get("usage"),
        "wall_s": round(wall, 1), "model_s": round(sum(float(t.get("latency_s") or 0) for t in turns), 1),
        "trace": str(out / "trace"), "episode_id": o.episode_id}
    program = (tr.last("agent.code.program") or {}).get("payload")
    if program is not None:                                 # a code.program run (a scripted solution)
        result["program_output"] = str(program.get("output", ""))[-2000:]
        result["program_error"] = program.get("error")
    if video is not None:
        try:
            result["video"] = write_mp4(env.frames, video, fps=env.spec().control_hz)
        except ImportError:
            result["video"] = {"written": False, "reason": "imageio-ffmpeg is not installed"}
    env.close()
    return result


__all__ = ["FrameRecorder", "VIDEO_MAX_BYTES", "run_recorded_episode", "write_mp4"]
