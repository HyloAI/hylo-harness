"""LeRobotDataset (v3) as an external DataSource (RFC-0004 W1): demonstrations or other robots' logs -> ingested runs.

State and action vectors come from the parquet table; camera frames are decoded only for kept frames
(every `image_every`, plus the last). The dataset's own `next.success` column is the outcome label when present;
otherwise episodes stay unlabeled unless the ingest spec says the source is a curated demonstration set
(`assume_success`). Runs in a venv with lerobot (envs/libero).

    {"name": "libero10-demos", "licence": "MIT", "image_every": 10, "out": "runs/external",
     "source": {"type": "lerobot.dataset", "params": {"root": "data/libero_10", "repo_id": "HuggingFaceVLA/libero",
                "cameras": {"observation.images.image": "agentview", "observation.images.wrist_image": "wrist"}}}}
    envs/libero/.venv/bin/python -m hylo.learn.data ingest spec.json
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from hylo.core.datasource import ExternalEpisode, ExternalFrame

from .export import DEFAULT_STATE, POLICY_CAMERAS

LIBERO_STATE = {"eef_pos": 3, "eef_axis_angle": 3, "gripper_qpos": 2}   # how export.py packs observation.state


class LeRobotDatasetSource:
    def __init__(self, root: str, repo_id: str = "local/dataset", cameras: dict[str, str] | None = None,
                 state_split: dict[str, int] | None = None, episodes: list[int] | None = None, max_episodes: int | None = None,
                 image_every: int = 1, task_prefix: str = "", licence: str = "unknown") -> None:
        self.root, self.repo_id = root, repo_id
        self.cameras = cameras or {v: k for k, v in POLICY_CAMERAS.items()}   # dataset key -> our camera name
        self.state_split = state_split or dict(LIBERO_STATE)
        self.only, self.max_episodes, self.image_every = episodes, max_episodes, max(1, image_every)
        self.task_prefix, self.licence = task_prefix, licence
        self.origin = f"lerobot:{repo_id}@{root}"
        self.name = f"lerobot.dataset[{repo_id}]"

    def _state(self, vec: np.ndarray) -> dict[str, np.ndarray]:
        out, i = {}, 0
        for k, n in self.state_split.items():
            if i + n <= len(vec):
                out[k] = vec[i:i + n].astype(np.float32)
            i += n
        if i != len(vec):
            out["state"] = vec.astype(np.float32)                  # layout unknown: keep the raw vector too
        return out

    def episodes(self) -> Iterator[ExternalEpisode]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(self.repo_id, root=self.root)
        table = ds.hf_dataset.with_format("numpy")
        cols = set(table.column_names)
        eps = ds.meta.episodes
        tasks = ds.meta.tasks
        done = 0
        for row in range(len(eps)):
            ep_idx = int(eps[row]["episode_index"])
            if self.only is not None and ep_idx not in self.only:
                continue
            if self.max_episodes is not None and done >= self.max_episodes:
                return
            lo, hi = int(eps[row]["dataset_from_index"]), int(eps[row]["dataset_to_index"])
            rows = table[lo:hi]
            ti = int(rows["task_index"][0]) if "task_index" in cols else 0
            instruction = str(tasks.index[ti]) if hasattr(tasks, "index") else str(eps[row]["tasks"][0])
            frames = []
            for k in range(hi - lo):
                keep = k % self.image_every == 0 or k == hi - lo - 1
                images = {}
                if keep:
                    item = ds[lo + k]
                    for key, cam in self.cameras.items():
                        if key in item:
                            images[cam] = (item[key].numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
                frames.append(ExternalFrame(state=self._state(np.asarray(rows["observation.state"][k])) if "observation.state" in cols else {},
                                            images=images, action=np.asarray(rows["action"][k], np.float32) if "action" in cols else None))
            success = bool(np.asarray(rows["next.success"]).any()) if "next.success" in cols else None
            slug = "_".join(instruction.lower().split())[:120]
            yield ExternalEpisode(episode_id=f"{self.repo_id.replace('/', '.')}-ep{ep_idx:05d}",
                                  task_id=f"{self.task_prefix}{slug}", instruction=instruction, frames=frames, success=success,
                                  meta={"episode_index": ep_idx, "frames": hi - lo, "fps": ds.fps})
            done += 1


__all__ = ["LeRobotDatasetSource", "LIBERO_STATE", "DEFAULT_STATE"]
