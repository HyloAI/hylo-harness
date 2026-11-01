"""Success judges trained on our own traces (RFC-0004 A3; the instrument RFC-0003 M5 needs).

A judge decides "is the task done?" from what a real robot has: camera frames, proprio state and the task. It never
sees privileged simulator state, the episode length or the termination reason (in LIBERO an episode ends the moment
it succeeds, so length alone would leak the answer).

Model: the critic's numpy MLP ensemble over proprio + task one-hot + grayscale thumbnails of each camera. Trained by
the "judge.trace" learner in hylo-learn (hylo.learn.judges).

    instrument: {type: judge.trace, params: {path: "artifact:l10-trace-v2@54a1efadf00d34d8"}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from hylo.core.contract import Contract
from hylo.core.instrument import Judgement
from hylo.core.trace import EpisodeTrace

from .critic import STATE_KEYS, ChunkCritic, _thumb


class JudgeConfig(Contract):
    state_keys: tuple[str, ...] = STATE_KEYS
    cameras: tuple[str, ...] = ("agentview", "wrist")
    image_px: int = Field(default=8, ge=2, le=32)
    tasks: tuple[str, ...] = ()


def judge_features(state: dict[str, Any], images: dict[str, Any], task_id: str, cfg: JudgeConfig) -> np.ndarray:
    parts = [np.asarray(state[k], dtype=np.float64).reshape(-1) for k in cfg.state_keys]
    onehot = np.zeros(len(cfg.tasks))
    if task_id in cfg.tasks:
        onehot[cfg.tasks.index(task_id)] = 1.0
    parts.append(onehot)
    for cam in cfg.cameras:
        img = images.get(cam)
        parts.append(_thumb(np.asarray(img), cfg.image_px) if img is not None else np.zeros(cfg.image_px ** 2))
    return np.concatenate(parts)


def imaged_frames(trace: EpisodeTrace, cameras: tuple[str, ...]) -> list[tuple[int, dict[str, Any]]]:
    """(observation index, payload) of every observation that carries all `cameras`."""
    out = []
    for i, ev in enumerate(trace.of_type("observation")):
        p = ev["payload"]
        if all(c in (p.get("images") or {}) for c in cameras):
            out.append((i, p))
    return out


class TraceSuccessJudge:
    """SuccessJudge instrument over a sealed judge.trace artifact."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.model = ChunkCritic.load(path)
        self.cfg = JudgeConfig.model_validate(self.model.meta["judge"])
        self.name = f"judge.trace[{Path(path).name}]@0.1"

    def judge(self, episode: EpisodeTrace) -> Judgement:
        task_id = episode.first("episode.start")["payload"]["task_id"]
        frames = imaged_frames(episode, self.cfg.cameras)
        if not frames:
            return Judgement(p_success=0.0, info={"error": "no frames with all cameras"})
        x = np.stack([judge_features(p["state"], p["images"], task_id, self.cfg) for _, p in frames])
        mu, sd = self.model.predict(x)
        return Judgement(p_success=float(np.clip(mu[-1], 0, 1)), progress=[round(float(v), 4) for v in mu],
                         info={"std": round(float(sd[-1]), 4), "frames": len(frames)})
