"""Export Hylo traces to a LeRobotDataset (v3) for training (imitation, RECAP-style advantage
conditioning, RL warm starts). Runs in a venv with `lerobot` (envs/libero).

Each frame pairs the observation seen *before* an action with the action actually executed (after the
SafetyShield), plus next.reward / next.done / next.success. Video features need traces recorded with
trace.image_every = 1; episodes without per-step images are skipped and reported.
A provenance file maps every exported episode back to its run, episode id, status and agent, and the dataset is
sealed as a `dataset` artifact (RFC-0004). Gate and held-out runs are refused whatever `--splits` says.

    python -m hylo.policies.lerobot.export --runs runs/<run_dir> ... --out data/<name> \
        [--only-success] [--state-keys eef_pos eef_axis_angle gripper_qpos] [--cameras agentview wrist]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hylo.core.artifact import Provenance, SourceRef, SplitLeak, check_splits, forbidden_split, write_manifest
from hylo.core.trace import EpisodeTrace, TraceStore

DEFAULT_STATE = ("eef_pos", "eef_axis_angle", "gripper_qpos")
DEFAULT_CAMERAS = {"agentview": "observation.images.image", "wrist": "observation.images.image2"}
POLICY_CAMERAS = {"agentview": "observation.images.camera1", "wrist": "observation.images.camera2"}  # smolvla_libero's keys


def episode_frames(tr: EpisodeTrace, state_keys: tuple[str, ...], cameras: dict[str, str]) -> Iterator[dict[str, Any]]:
    """Yield (observation before action, executed action, outcome) frames; raises if images are missing."""
    obs, last_result = None, None
    pending: dict[str, Any] | None = None
    for e in tr.events:
        t, p = e["type"], e["payload"]
        if t == "observation":
            if pending is not None and last_result is not None:
                pending["next.reward"] = np.array([last_result.get("reward") or 0.0], dtype=np.float32)
                pending["next.done"] = np.array([bool(last_result.get("terminated") or last_result.get("truncated"))])
                pending["next.success"] = np.array([bool(last_result.get("success"))])
                yield pending
                pending, last_result = None, None
            obs = p
        elif t == "action" and obs is not None:
            missing = [c for c in cameras if c not in obs["images"]]
            if missing:
                raise KeyError(f"no images for {missing} at this step (record with trace.image_every=1)")
            frame = {key: np.asarray(obs["images"][cam], dtype=np.uint8) for cam, key in cameras.items()}
            frame["observation.state"] = np.concatenate(
                [np.asarray(obs["state"][k], dtype=np.float32).reshape(-1) for k in state_keys])
            frame["action"] = np.asarray(p["data"], dtype=np.float32).reshape(-1)
            frame["task"] = obs.get("instruction") or ""
            pending = frame
        elif t == "step.result":
            last_result = p


def export(run_dirs: list[str], out: str, *, repo_id: str = "local/hylo", fps: int = 20,
           state_keys: tuple[str, ...] = DEFAULT_STATE, cameras: dict[str, str] | None = None,
           only_success: bool = False, robot_type: str = "panda",
           splits: tuple[str, ...] | None = None) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cameras = dict(cameras or DEFAULT_CAMERAS)
    if splits is not None:
        check_splits(splits)
    plan: list[tuple[TraceStore, dict]] = []
    sources: list[SourceRef] = []
    for rd in run_dirs:
        log = json.loads((Path(rd) / "evallog.json").read_text())
        split = json.loads((Path(rd) / "manifest.json").read_text()).get("benchmark", {}).get("split")
        if forbidden_split(split) or (splits is not None and split not in splits):
            raise SplitLeak(f"{rd} is a {split!r} run; only {list(splits or ('search', 'search_val'))} may be exported")
        sources.append(SourceRef(id=str(log["run_id"]), split=split, path=str(rd), n_episodes=len(log["samples"])))
        store = TraceStore(rd)
        for s in log["samples"]:
            if only_success and s["status"] != "success":
                continue
            plan.append((store, {**s, "run_id": log["run_id"], "agent": log["agent"]["name"],
                                 "components": log["agent"].get("components", {})}))
    if not plan:
        raise SystemExit("nothing to export")
    probe = next(episode_frames(plan[0][0].read(plan[0][1]["episode_id"]), state_keys, cameras))
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": probe["observation.state"].shape, "names": None},
        "action": {"dtype": "float32", "shape": probe["action"].shape, "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for key in cameras.values():
        features[key] = {"dtype": "video", "shape": probe[key].shape, "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out, robot_type=robot_type, use_videos=True)
    provenance, skipped = [], []
    for store, s in plan:
        try:
            frames = list(episode_frames(store.read(s["episode_id"]), state_keys, cameras))
        except KeyError as exc:
            skipped.append({"episode_id": s["episode_id"], "reason": str(exc)})
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        provenance.append({"episode_index": len(provenance), "episode_id": s["episode_id"], "run_id": s["run_id"],
                           "task_id": s["task_id"], "seed": s["seed"], "status": s["status"], "agent": s["agent"],
                           "components": s["components"], "frames": len(frames)})
    ds.finalize()
    summary = {"out": out, "episodes": len(provenance), "skipped": len(skipped), "frames": sum(p["frames"] for p in provenance),
               "features": sorted(features)}
    (Path(out) / "hylo_provenance.json").write_text(json.dumps({"summary": summary, "episodes": provenance,
                                                                        "skipped": skipped}, indent=2))
    summary["artifact"] = write_manifest(out, kind="dataset", role="data", metrics={k: summary[k] for k in ("episodes", "frames")},
                                         provenance=Provenance(builder="hylo.policies.lerobot.export@0.1", sources=sources,
                                                               params={"only_success": only_success, "fps": fps,
                                                                       "state_keys": list(state_keys), "cameras": cameras})).digest
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/hylo")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--state-keys", nargs="+", default=list(DEFAULT_STATE))
    ap.add_argument("--policy-cameras", action="store_true", help="name cameras camera1/camera2 (smolvla_libero)")
    ap.add_argument("--splits", nargs="+", help="refuse runs from any other split (e.g. search search_val)")
    args = ap.parse_args()
    print(json.dumps(export(args.runs, args.out, repo_id=args.repo_id, fps=args.fps, state_keys=tuple(args.state_keys),
                            only_success=args.only_success, cameras=POLICY_CAMERAS if args.policy_cameras else None,
                            splits=tuple(args.splits) if args.splits else None), indent=2))


if __name__ == "__main__":
    main()
