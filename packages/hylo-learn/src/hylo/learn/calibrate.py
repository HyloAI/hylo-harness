"""Instrument calibration and labelling (RFC-0004): is a judge good enough to label data without ground truth?

    python -m hylo.learn.calibrate --judge '{"type": "judge.trace", "params": {"path": ...}}' --runs DIR ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from hylo.core.artifact import DEV_SPLITS, ArtifactManifest, development_runs, read_manifest, write_manifest
from hylo.core.instrument import CalibrationReport, SuccessJudge
from hylo.core.trace import TraceStore


def _auc(s: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                         # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration_report(scores: Sequence[float], labels: Sequence[int], *, target_precision: float = 0.95,
                       min_n: int = 50, min_recall: float = 0.3, runs: Sequence[str] = ()) -> CalibrationReport:
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    n = len(s)
    bins = np.minimum((s * 10).astype(int), 9)
    ece = float(sum(abs(s[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10)
                    if (bins == b).any()) / max(n, 1))
    thr = prec = rec = None
    for t in sorted(set(s.tolist())):            # lowest threshold that reaches the target precision
        pred = s >= t
        if pred.sum() == 0:
            break
        p = float(y[pred].mean())
        if p >= target_precision:
            thr, prec, rec = float(t), p, float(pred[y == 1].mean()) if (y == 1).any() else 0.0
            break
    ready = n >= min_n and prec is not None and (rec or 0.0) >= min_recall
    return CalibrationReport(n=n, base_rate=float(y.mean()) if n else 0.0, auc=_auc(s, y),
                             brier=float(np.mean((s - y) ** 2)) if n else 0.0, ece=round(ece, 4), threshold=thr,
                             precision=prec, recall=rec, target_precision=target_precision, ready=ready,
                             runs=list(runs), scores=s)


def calibrate_judge(judge: SuccessJudge, roots: Iterable[str | Path], *, splits: Sequence[str] = DEV_SPLITS,
                    judge_artifact: str | Path | None = None, target_precision: float = 0.95, min_n: int = 50,
                    limit: int | None = None, seed: int = 0, task_prefix: str | None = None,
                    records: list[dict[str, Any]] | None = None, log: Any = None) -> CalibrationReport:
    """Score `judge` against simulator outcomes. Runs the judge was built from are excluded (no self-calibration);
    with `limit`, a seeded random sample of episodes is used. Per-episode results are appended to `records`.
    With `judge_artifact`, the report is written into that artifact's manifest (sealed again)."""
    manifest: ArtifactManifest | None = read_manifest(judge_artifact) if judge_artifact else None
    trained_on = {s.id for s in manifest.provenance.sources} if manifest else set()
    pool, used = [], []
    for rd, ref in development_runs(roots, splits):
        if ref.id in trained_on:
            continue
        found = [(rd, smp) for smp in json.loads((rd / "evallog.json").read_text())["samples"]
                 if smp["status"] in ("success", "failure")
                 and (not task_prefix or str(smp.get("task_id", "")).startswith(task_prefix))]
        if found:
            used.append(ref.id)
            pool += found
    if limit is not None and len(pool) > limit:
        idx = np.random.default_rng(seed).choice(len(pool), size=limit, replace=False)
        pool = [pool[i] for i in sorted(idx)]
    scores, labels = [], []
    for k, (rd, smp) in enumerate(pool):
        j = judge.judge(TraceStore(rd).read(smp["episode_id"]))
        scores.append(j.p_success)
        labels.append(int(smp["status"] == "success"))
        if records is not None:
            records.append({"run": rd.name, "episode_id": smp["episode_id"], "task_id": smp.get("task_id"),
                            "success": labels[-1], "p_success": j.p_success, "info": j.info})
        if log:
            log(f"[calibrate] {k + 1}/{len(pool)} truth={labels[-1]} p={j.p_success:.2f}")
    rep = calibration_report(scores, labels, target_precision=target_precision, min_n=min_n, runs=used)
    if manifest is not None and judge_artifact is not None:
        from hylo.core.artifact import manifest_root

        root = manifest_root(judge_artifact)
        write_manifest(root, kind=manifest.kind, role=manifest.role, provenance=manifest.provenance,
                       name=manifest.name, metrics=manifest.metrics, mutable=manifest.mutable,
                       calibration={**summarize(rep), "judge": getattr(judge, "name", type(judge).__name__)})
    return rep


def summarize(rep: CalibrationReport) -> dict[str, Any]:
    return rep.model_dump(mode="json", exclude={"scores"})


def main(argv: Sequence[str] | None = None) -> int:
    """python -m hylo.learn.calibrate --judge '{"type": "judge.vlm", "params": {}}' --runs DIR ... [--limit N]"""
    import argparse

    from hylo.core.registry import registry

    ap = argparse.ArgumentParser(prog="hylo calibrate")
    ap.add_argument("--judge", required=True, help='instrument spec as JSON: {"type": ..., "params": {...}}')
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=list(DEV_SPLITS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-prefix")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--artifact", help="the judge's artifact (directory or store id): excludes its training runs, "
                                       "stores the report")
    ap.add_argument("--out", help="write report + per-episode records here (JSON)")
    a = ap.parse_args(argv)
    from hylo.core.store import resolve_refs

    a.artifact, a.runs = resolve_refs(a.artifact), resolve_refs(a.runs)    # store ids -> this machine's paths
    judge = registry.build("instruments", json.loads(a.judge))
    records: list[dict[str, Any]] = []
    rep = calibrate_judge(judge, a.runs, splits=a.splits, judge_artifact=a.artifact, limit=a.limit, seed=a.seed,
                          task_prefix=a.task_prefix, target_precision=a.target_precision, records=records, log=print)
    out = {"judge": getattr(judge, "name", a.judge), **summarize(rep)}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**out, "records": records}, indent=2, default=str))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def label_run(run_dir: str | Path, judge: SuccessJudge, judge_artifact: str | Path, *,
              threshold: float | None = None) -> dict[str, Any]:
    """Label the unlabeled episodes of an ingested external run with a calibrated success judge.

    Only a judge whose calibration is `ready` may label, at (at least) its calibrated high-precision threshold; an
    episode below it stays unlabeled (the judge's score is recorded, but a low score is not called a failure). A run
    the judge was built from is refused, and so is any run with simulator outcomes (it needs no judge)."""
    from hylo.core.artifact import verify

    run_dir = Path(run_dir)
    split = str(json.loads((run_dir / "manifest.json").read_text()).get("benchmark", {}).get("split"))
    if not split.startswith("external:"):
        raise ValueError(f"{run_dir.name} is a {split!r} run; only ingested external runs are labeled by a judge")
    m = verify(judge_artifact)
    cal = m.calibration or {}
    if not cal.get("ready"):
        raise ValueError(f"judge {m.digest} is not calibrated for labelling (run `hylo calibrate` with --artifact first)")
    log = json.loads((run_dir / "evallog.json").read_text())
    if log["run_id"] in {s.id for s in m.provenance.sources}:
        raise ValueError("a judge never labels data it was built from")
    thr = max(float(threshold or 0.0), float(cal["threshold"]))
    store, counts = TraceStore(run_dir), {"labeled_success": 0, "left_unlabeled": 0, "skipped": 0}
    for smp in log["samples"]:
        if smp.get("status") != "unlabeled":
            counts["skipped"] += 1
            continue
        j = judge.judge(store.read(smp["episode_id"]))
        by = f"judge:{m.digest}"
        if j.p_success >= thr:
            smp["status"] = "success"
            smp["labels"] = {"success": {"value": 1.0, "by": by, "p": round(j.p_success, 4), "threshold": thr}}
            smp["scores"] = {"success": {"value": 1.0}}
            counts["labeled_success"] += 1
        else:
            smp["labels"] = {"success": {"value": None, "by": by, "p": round(j.p_success, 4), "threshold": thr}}
            counts["left_unlabeled"] += 1
    labeled = [s for s in log["samples"] if s["status"] in ("success", "failure")]
    log["results"] = {**log.get("results", {}), "labeled": len(labeled),
                      "success_mean": (sum(s["status"] == "success" for s in labeled) / len(labeled)) if labeled else None,
                      "judge": {"digest": m.digest, "threshold": thr, "expected_precision": cal.get("precision")}}
    (run_dir / "evallog.json").write_text(json.dumps(log, indent=2, default=str))
    return {"run": log["run_id"], "judge": m.digest, "threshold": thr, **counts}
