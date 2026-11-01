"""External data (RFC-0004): demonstrations, videos and other robots' logs, ingested as ordinary runs.

A DataSource yields ExternalEpisodes (frames with state and camera images, actions if the source has them, an
outcome if the source states one). `ingest` writes them into the trace store as a run whose split is
`external:<name>`, so every builder reads them through the same split guard, and seals the run as a `dataset`
artifact recording origin and licence. Nothing external enters a gate or held-out split.

Two checks guard what external data may be used for:
- labels say who produced them: "source" (the dataset says so), "judge:<digest>" (a calibrated instrument, see
  `label_run`), or none; an unlabeled episode is never treated as a success;
- `contamination_report` compares each external episode's initial state with the initial states of protected
  (gate / held-out) episodes of the same task; an external source that reproduces protected init states must not
  be used to build anything that is then gated on them.

    python -m worldcurrent.learn.data ingest SPEC.json      # {"source": {"type": ..., "params": ...}, "name": ..., "out": ...}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from worldcurrent.core.artifact import FORBIDDEN_SPLIT_PREFIXES, Provenance, SourceRef, find_runs, run_split, write_manifest
from worldcurrent.core.datasource import ExternalEpisode, external_split
from worldcurrent.core.events import Event
from worldcurrent.core.trace import TraceStore


def ingest(episodes: Iterable[ExternalEpisode], out_root: str | Path, *, name: str, origin: str, licence: str,
           image_every: int = 1, assume_success: bool = False, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write episodes as a run under split external:<name> and seal it. Returns the run summary.

    assume_success: label every episode a success on the source's authority (curated demonstrations); otherwise an
    episode is a success only if the source says so, and is left unlabeled when it says nothing."""
    split = external_split(name)
    run_id = f"external-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(out_root) / run_id
    store = TraceStore(run_dir)
    samples, n_frames, with_actions = [], 0, 0
    for idx, ep in enumerate(episodes):
        if ep.success is not None:
            status, label = ("success" if ep.success else "failure"), {"value": float(ep.success), "by": "source"}
        elif assume_success:
            status, label = "success", {"value": 1.0, "by": "source:assumed"}
        else:
            status, label = "unlabeled", {"value": None, "by": None}
        seq = 0

        def w(type_: str, payload: dict[str, Any]) -> None:
            nonlocal seq
            store.write(Event(type=type_, episode_id=ep.episode_id, seq=seq, payload=payload, source="ingest"))
            seq += 1

        w("episode.start", {"task_id": ep.task_id, "task_version": "external", "seed": idx, "split": split,
                            "episode_index": idx, "env": f"external:{origin}", "agent": f"source:{name}",
                            "agent_components": {}, "guards": [], "compat_warnings": []})
        for k, fr in enumerate(ep.frames):
            keep = image_every > 0 and k % image_every == 0 or k == len(ep.frames) - 1   # always keep the last frame
            w("observation", {"state": dict(fr.state), "images": dict(fr.images) if keep else {},
                              "instruction": ep.instruction, "time": float(k)})
            if fr.action is not None:
                w("action", {"data": fr.action})
                w("step.result", {"step": k + 1, "reward": 0.0, "terminated": False, "truncated": False,
                                  "success": False, "info": {}})
                with_actions += 1
        w("episode.end", {"status": status, "success": label["value"] == 1.0 if label["value"] is not None else None,
                          "steps": len(ep.frames), "termination_reason": "source", "safety_interventions": 0})
        store.close_episode(ep.episode_id)
        n_frames += len(ep.frames)
        samples.append({"key": f"{ep.task_id}#{idx}", "task_id": ep.task_id, "seed": idx, "episode_id": ep.episode_id,
                        "status": status, "steps": len(ep.frames), "trace": f"episodes/{ep.episode_id}.jsonl",
                        "scores": {"success": {"value": label["value"]}}, "labels": {"success": label},
                        "meta": ep.meta})
    if not samples:
        raise ValueError("the source yielded no episodes")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "schema_version": "wc.external/v0.1", "created_at": time.time(),
        "benchmark": {"id": f"external/{name}", "split": split},
        "source": {"name": name, "origin": origin, "licence": licence, "params": params or {}}}, indent=2, default=str))
    labeled = [s for s in samples if s["status"] in ("success", "failure")]
    (run_dir / "evallog.json").write_text(json.dumps({
        "version": 1, "run_id": run_id, "benchmark": {"id": f"external/{name}", "split": split},
        "agent": {"name": f"source:{name}"}, "samples": samples,
        "results": {"n_episodes": len(samples), "labeled": len(labeled), "frames": n_frames,
                    "frames_with_actions": with_actions,
                    "success_mean": (sum(s["status"] == "success" for s in labeled) / len(labeled)) if labeled else None}},
        indent=2, default=str))
    m = write_manifest(run_dir, kind="dataset", role="data", name=run_id, mutable=True,
                       metrics={"episodes": len(samples), "frames": n_frames, "frames_with_actions": with_actions},
                       provenance=Provenance(builder="worldcurrent.learn.data@0.1", params=params or {},
                                             sources=[SourceRef(kind="external", id=name, split=split, path=origin,
                                                                n_episodes=len(samples), licence=licence)]))
    return {"run_dir": str(run_dir), "run_id": run_id, "split": split, "episodes": len(samples), "frames": n_frames,
            "frames_with_actions": with_actions, "labeled": len(labeled), "artifact": m.digest}


# ---------------------------------------------------------------- contamination

def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split()).rstrip(".")


def _first(trace, keys: Sequence[str]) -> tuple[np.ndarray | None, str]:
    """Initial state vector and instruction of an episode (tasks are matched by instruction: external datasets
    name tasks their own way)."""
    for ev in trace.of_type("observation"):
        st = ev["payload"].get("state") or {}
        parts = [np.asarray(st[k], float).reshape(-1) for k in keys if k in st]
        return (np.concatenate(parts) if parts else None), _norm(ev["payload"].get("instruction") or "")
    return None, ""


def contamination_report(external_runs: Iterable[str | Path], protected_roots: Iterable[str | Path], *,
                         state_keys: Sequence[str] = ("eef_pos", "eef_axis_angle", "gripper_qpos"),
                         tolerance: float = 1e-3) -> dict[str, Any]:
    """External episodes whose initial state matches a protected (gate / held-out) episode of the same task.

    Reads only the first observation of protected episodes; nothing from them is written anywhere."""
    protected: dict[str, list[tuple[str, np.ndarray]]] = {}
    for rd in find_runs(protected_roots):
        sp = run_split(rd) or ""
        if not sp.startswith(FORBIDDEN_SPLIT_PREFIXES):
            continue
        store = TraceStore(rd)
        for s in json.loads((rd / "evallog.json").read_text())["samples"]:
            x, instr = _first(store.read(s["episode_id"]), state_keys)
            if x is not None:
                protected.setdefault(instr, []).append((f"{rd.name}/{s['episode_id']}", x))
    hits, checked = [], 0
    for rd in find_runs(external_runs):
        store = TraceStore(rd)
        for s in json.loads((rd / "evallog.json").read_text())["samples"]:
            x, instr = _first(store.read(s["episode_id"]), state_keys)
            if x is None:
                continue
            checked += 1
            for ref, y in protected.get(instr, []):
                if y.shape == x.shape and float(np.abs(x - y).max()) <= tolerance:
                    hits.append({"external": f"{rd.name}/{s['episode_id']}", "protected": ref})
                    break
    return {"checked": checked, "protected_episodes": sum(map(len, protected.values())), "matches": hits,
            "clean": not hits, "tolerance": tolerance, "state_keys": list(state_keys)}


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    from worldcurrent.core.registry import registry

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "ingest":
        print("usage: python -m worldcurrent.learn.data ingest SPEC.json", file=sys.stderr)
        return 2
    spec = json.loads(Path(args[1]).read_text())
    src = registry.build("datasources", spec["source"])
    rep = ingest(src.episodes(), spec.get("out", "runs/external"), name=spec["name"], origin=getattr(src, "origin", spec["source"]["type"]),
                 licence=spec.get("licence") or getattr(src, "licence", "unknown"), image_every=int(spec.get("image_every", 1)),
                 assume_success=bool(spec.get("assume_success", False)), params=spec["source"])
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

