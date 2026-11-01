"""Episode replay data: one trace -> a compact, step-indexed timeline, and camera frames as arrays."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from worldcurrent.core.trace import EpisodeTrace, TraceStore


class TraceCache:
    """A few recently opened traces (reading resolves every image blob, so reuse matters for frame requests)."""

    def __init__(self, size: int = 6) -> None:
        self.size = size
        self._items: OrderedDict[tuple[str, str], EpisodeTrace] = OrderedDict()

    def get(self, run_dir: Path, episode_id: str) -> EpisodeTrace:
        key = (str(run_dir), episode_id)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        if not (run_dir / "episodes" / f"{episode_id}.jsonl").exists():
            raise KeyError(episode_id)
        tr = TraceStore(run_dir).read(episode_id)
        _ = tr.events
        self._items[key] = tr
        while len(self._items) > self.size:
            self._items.popitem(last=False)
        return tr


def to_json(v: Any, nd: int = 4) -> Any:
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), nd).tolist()
    if isinstance(v, (list, tuple)):
        return [to_json(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: to_json(x, nd) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        return round(float(v), nd)
    if isinstance(v, np.integer):
        return int(v)
    return v


def timeline(tr: EpisodeTrace) -> dict[str, Any]:
    start = (tr.first("episode.start") or {}).get("payload", {})
    end = (tr.last("episode.end") or {}).get("payload", {})
    step = 0
    obs_i = 0
    frames, state, actions, chunks, safety, memory, skills, notes = [], [], [], [], [], [], [], []
    instruction, cameras = None, []
    for e in tr.events:
        t, p = e["type"], e["payload"]
        if t == "observation":
            imgs = p.get("images") or {}
            if imgs:
                cameras = sorted(set(cameras) | set(imgs))
                frames.append({"obs": obs_i, "step": step, "cameras": sorted(imgs)})
            st = p.get("state") or {}
            state.append({"step": step, **{k: to_json(np.asarray(v).reshape(-1)) for k, v in st.items()
                                         if k in ("eef_pos", "gripper_qpos", "pos", "goal", "vel")}})
            instruction = instruction or p.get("instruction")
            obs_i += 1
        elif t == "agent.chunk":
            prov = p.get("provenance") or {}
            chunks.append({"step": step, "execution_steps": p.get("execution_steps"),
                           "latency_s": to_json(p.get("latency_s"), 3), "horizon": 0 if p.get("actions") is None else len(p["actions"]),
                           "provenance": to_json({k: v for k, v in prov.items() if k != "seeds"})})
        elif t == "action":
            actions.append(to_json(np.asarray(p.get("data")).reshape(-1)))
            step += 1
        elif t == "safety.review" and p.get("modified"):
            d = np.abs(np.asarray(p["approved"], float) - np.asarray(p["proposed"], float))
            safety.append({"step": step, "reasons": p.get("reasons"), "clipped": p.get("clipped"),
                           "max_change": round(float(d.max()), 4)})
        elif t == "memory.retrieval":
            memory.append({"step": step, **to_json({k: v for k, v in p.items() if k not in ("key", "prior")})})
        elif t == "skill.applied":
            skills.append({"step": step, **to_json(p)})
        elif t in ("agent.decision", "error"):
            notes.append({"step": step, "type": t, **to_json(p)})
    return {"episode_id": tr.episode_id, "start": to_json(start), "end": to_json(end), "instruction": instruction,
            "steps": step, "cameras": cameras, "frames": frames, "state": state, "actions": actions, "chunks": chunks,
            "safety": safety, "memory": memory, "skills": skills, "notes": notes}


def frame_array(tr: EpisodeTrace, obs_index: int, camera: str) -> np.ndarray:
    for i, e in enumerate(tr.of_type("observation")):
        if i == obs_index:
            img = (e["payload"].get("images") or {}).get(camera)
            if img is None:
                raise KeyError(f"no {camera} frame at observation {obs_index}")
            return np.asarray(img, dtype=np.uint8)
    raise KeyError(f"observation {obs_index} out of range")
