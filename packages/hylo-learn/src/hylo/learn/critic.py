"""Train a ChunkCritic from Hylo traces.

    python -m hylo.learn.critic --runs runs/rsi-libero10/runs runs/evolve/libero10-llm \
        --out critics/l10-v1 [--image-camera agentview] [--task-prefix libero/libero_10/]

Protocol integrity (RFC-0002): only episodes whose split is in --splits (default: search, search_val) are read;
gate and held-out episodes never reach the critic, so a critic-steered candidate is still judged on unseen
init states. The critic is fit on `search` and evaluated on `search_val`, then refit on both. The output directory is
sealed as a `value_model` artifact (RFC-0004) listing the runs it was trained on; `CriticLearner` is the same
trainer behind the learner interface ("critic.chunk_value").

Reports the offline diagnostics that decide whether re-ranking can help at all:
- val AUC / Brier of the predicted value against the episode's final success;
- the same for a state-only model: if (state + chunk) is no better than state alone, the critic cannot tell
  chunks apart and steering with it is noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.artifact import ArtifactManifest, Provenance, check_splits, development_runs, write_manifest
from hylo.core.artifact import find_runs as _find_runs
from hylo.core.trace import TraceStore

from hylo.agents.critic import ChunkCritic, FeatureConfig, chunk_features, state_features

N_CHUNK_FEATURES = 15


def find_runs(roots: list[str]) -> list[Path]:
    return _find_runs(roots)


def episode_samples(run_dir: str, episode_id: str, fcfg: dict[str, Any], gamma: float,
                    splits: tuple[str, ...], task_prefix: str | None) -> dict[str, Any] | None:
    """(state+chunk features, target) for every policy chunk of one episode, or None if it is filtered out."""
    cfg = FeatureConfig.model_validate(fcfg)
    tr = TraceStore(run_dir).read(episode_id)
    start, end = tr.first("episode.start"), tr.last("episode.end")
    if start is None or end is None:
        return None
    sp = start["payload"]
    if sp.get("split") not in splits or (task_prefix and not sp["task_id"].startswith(task_prefix)):
        return None
    status = end["payload"]["status"]
    if status not in ("success", "failure"):
        return None                                   # errors / safety aborts say nothing about chunk quality
    T = int(end["payload"]["steps"])
    obs, step, rows, sig = None, 0, [], hashlib.sha256()
    for e in tr.events:
        t, p = e["type"], e["payload"]
        if t == "observation":
            obs = p
        elif t == "step.result":
            step += 1
        elif t == "agent.chunk" and obs is not None:
            prov = p.get("provenance") or {}
            if str(prov.get("model", "")).startswith("recovery."):
                continue                              # scripted recoveries are not sampler candidates
            acts = np.asarray(p["actions"], dtype=np.float64)
            ex = int(p.get("execution_steps") or len(acts))
            sig.update(acts[:ex].round(5).tobytes())
            rows.append((state_features(obs["state"], step, sp["task_id"], obs.get("images"), cfg),
                         chunk_features(acts, ex, cfg.prefix_steps), step))
    if not rows:
        return None
    success = status == "success"
    y = [gamma ** (T - s) if success else 0.0 for _, _, s in rows]
    return {"x": np.stack([np.concatenate([a, b]) for a, b, _ in rows]), "y": np.asarray(y),
            "success": success, "split": sp["split"], "task_id": sp["task_id"], "seed": int(sp["seed"]),
            "signature": sig.hexdigest()}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect(run_dirs: list[Path], cfg: FeatureConfig, *, gamma: float, splits: tuple[str, ...],
            task_prefix: str | None, workers: int) -> list[dict[str, Any]]:
    jobs = [(str(rd), eid) for rd in run_dirs for eid in TraceStore(rd).episode_ids()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(episode_samples, rd, eid, cfg.model_dump(mode="json"), gamma, splits, task_prefix)
                for rd, eid in jobs]
        eps = [f.result() for f in futs]
    seen, out = set(), []
    for ep in eps:                                    # identical configs replay identical episodes (CRN): dedupe
        if ep is not None and (ep["task_id"], ep["seed"], ep["signature"]) not in seen:
            seen.add((ep["task_id"], ep["seed"], ep["signature"]))
            out.append(ep)
    return out


def _stack(eps: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([e["x"] for e in eps])
    y = np.concatenate([e["y"] for e in eps])
    succ = np.concatenate([np.full(len(e["y"]), e["success"]) for e in eps])
    w = np.concatenate([np.full(len(e["y"]), 1.0 / len(e["y"])) for e in eps])  # every episode weighs the same
    return x, y, succ, w


def same_state_pairs(eps: list[dict[str, Any]]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Pairs of first chunks taken from the *identical* reset state (same task, same init seed) that led to
    different outcomes: the cases where re-ranking chunks could have changed the result."""
    by_state: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for e in eps:
        by_state.setdefault((e["task_id"], e["seed"]), []).append(e)
    pairs = []
    for group in by_state.values():
        for a in group:
            for b in group:
                if a["success"] and not b["success"] and not np.allclose(a["x"][0], b["x"][0]):
                    pairs.append((a["seed"], b["seed"], a["x"][0], b["x"][0]))
    return pairs


def cross_validate(eps: list[dict[str, Any]], cfg: FeatureConfig, kw: dict[str, Any], *, folds: int = 5,
                   log=print) -> dict[str, Any]:
    """Grouped k-fold over init states (an init state never sits in both train and test)."""
    seeds = sorted({e["seed"] % 50 for e in eps})
    fold_of = {s: i % folds for i, s in enumerate(seeds)}
    pairs = same_state_pairs(eps)
    pf_all, ps_all, lab_all, wins, ties, n_pairs = [], [], [], 0, 0, 0
    for k in range(folds):
        tr = [e for e in eps if fold_of[e["seed"] % 50] != k]
        te = [e for e in eps if fold_of[e["seed"] % 50] == k]
        if not tr or not te:
            continue
        xt, yt, _, wt = _stack(tr)
        xv, _, sv, _ = _stack(te)
        n_state = xt.shape[1] - N_CHUNK_FEATURES
        full = ChunkCritic.fit(xt, yt, cfg, weights=wt, **kw)
        state_only = ChunkCritic.fit(xt[:, :n_state], yt, cfg, weights=wt, **kw)
        pf_all.append(full.predict(xv)[0])
        ps_all.append(state_only.predict(xv[:, :n_state])[0])
        lab_all.append(sv)
        for sa, _, xa, xb in pairs:
            if fold_of[sa % 50] == k:
                va, vb = full.predict(np.stack([xa, xb]))[0]
                n_pairs += 1
                wins += va > vb
                ties += va == vb
    pf, ps, lab = np.concatenate(pf_all), np.concatenate(ps_all), np.concatenate(lab_all)
    out = {"folds": folds, "auc_state_chunk": round(auc(pf, lab), 4), "auc_state_only": round(auc(ps, lab), 4),
           "same_state_pairs": n_pairs,
           "same_state_pair_accuracy": round(float(wins) / n_pairs, 4) if n_pairs else None}
    if n_pairs:   # binomial standard error, for reading the accuracy against 0.5
        acc = wins / n_pairs
        out["same_state_pair_accuracy_se"] = round(float(np.sqrt(acc * (1 - acc) / n_pairs)), 4)
    log(f"[critic] grouped CV: {out}")
    return out


def train(run_roots: list[str], out: str, *, image_camera: str | None = None, task_prefix: str | None = None,
          splits: tuple[str, ...] = ("search", "search_val"), gamma: float = 0.995, horizon: int = 520,
          members: int = 5, hidden: int = 64, epochs: int = 60, workers: int = 16, seed: int = 0,
          holdout_seeds: tuple[int, ...] = (), log=print) -> dict[str, Any]:
    check_splits(splits)
    sources = development_runs(run_roots, splits)
    run_dirs = [rd for rd, _ in sources]
    tasks = tuple(sorted({s["task_id"] for rd in run_dirs for s in json.loads((rd / "evallog.json").read_text())["samples"]
                          if not task_prefix or s["task_id"].startswith(task_prefix)}))
    cfg = FeatureConfig(horizon=horizon, image_camera=image_camera, tasks=tasks)
    eps = collect(run_dirs, cfg, gamma=gamma, splits=splits, task_prefix=task_prefix, workers=workers)
    eps = [e for e in eps if e["seed"] % 50 not in set(holdout_seeds)]   # init states kept for an online test
    if not eps:
        raise SystemExit("no usable episodes")
    tasks = cfg.tasks
    tr = [e for e in eps if e["split"] == splits[0]]
    va = [e for e in eps if e["split"] != splits[0]]
    log(f"[critic] {len(run_dirs)} runs -> {len(eps)} distinct episodes ({len(tr)} train / {len(va)} val), "
        f"{sum(len(e['y']) for e in eps)} chunks, {len(tasks)} tasks")
    kw = dict(members=members, hidden=hidden, epochs=epochs, seed=seed)
    report: dict[str, Any] = {"episodes": len(eps), "train_episodes": len(tr), "val_episodes": len(va),
                              "chunks": int(sum(len(e["y"]) for e in eps)), "tasks": list(tasks), "splits": list(splits),
                              "gamma": gamma, "holdout_seeds": list(holdout_seeds), "runs": [str(r) for r in run_dirs]}
    report["cv"] = cross_validate(eps, cfg, kw, log=log)
    if tr and va:
        xt, yt, _, wt = _stack(tr)
        xv, yv, sv, _ = _stack(va)
        n_state = xt.shape[1] - N_CHUNK_FEATURES
        full = ChunkCritic.fit(xt, yt, cfg, weights=wt, **kw)
        state_only = ChunkCritic.fit(xt[:, :n_state], yt, cfg, weights=wt, **kw)
        pf, ps = full.predict(xv)[0], state_only.predict(xv[:, :n_state])[0]
        report["val"] = {"auc_state_chunk": round(auc(pf, sv), 4), "auc_state_only": round(auc(ps, sv), 4),
                         "brier_state_chunk": round(float(np.mean((pf - yv) ** 2)), 5),
                         "brier_state_only": round(float(np.mean((ps - yv) ** 2)), 5),
                         "brier_constant": round(float(np.mean((yt.mean() - yv) ** 2)), 5)}
        log(f"[critic] val: {report['val']}")
    x, y, _, w = _stack(eps)
    critic = ChunkCritic.fit(x, y, cfg, weights=w, **kw)
    critic.meta.update({"trained_on": report})
    digest = critic.save(out)
    report["weights_sha256_16"] = digest
    params = {"image_camera": image_camera, "task_prefix": task_prefix, "gamma": gamma, "horizon": horizon,
              "members": members, "hidden": hidden, "epochs": epochs, "seed": seed, "holdout_seeds": list(holdout_seeds)}
    report["artifact"] = write_manifest(out, kind="value_model", role="behaviour", metrics={
        k: report.get(k) for k in ("cv", "val", "episodes", "chunks")},
        provenance=Provenance(builder=CriticLearner.name, sources=[r for _, r in sources], params=params)).digest
    log(f"[critic] saved {out} ({digest}, artifact {report['artifact']})")
    return report


class CriticLearner:
    """Learner interface (core.learning) for the chunk critic: job.params are `train` keyword arguments."""

    name = "hylo.learn.critic@0.1"

    def fit(self, job) -> ArtifactManifest:
        from hylo.core.artifact import read_manifest

        kw = dict(job.params)
        if "holdout_seeds" in kw:
            kw["holdout_seeds"] = tuple(kw["holdout_seeds"])
        train(list(job.inputs), job.out, splits=tuple(job.splits), **kw)
        return read_manifest(job.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories, or roots to search for them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-camera")
    ap.add_argument("--task-prefix")
    ap.add_argument("--splits", nargs="+", default=["search", "search_val"], help="first = train, rest = validation")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--horizon", type=int, default=520)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--holdout-seeds", default="", help="init states to exclude, e.g. 25-29 (kept for an online test)")
    a = ap.parse_args()
    lo_hi = [int(v) for v in a.holdout_seeds.split("-")] if a.holdout_seeds else []
    holdout = tuple(range(lo_hi[0], lo_hi[-1] + 1)) if lo_hi else ()
    if any(s.startswith(("gate", "heldout")) for s in a.splits):
        ap.error("gate / heldout episodes must not be used to train a critic that will be gated on them")
    rep = train(a.runs, a.out, image_camera=a.image_camera, task_prefix=a.task_prefix, splits=tuple(a.splits),
                gamma=a.gamma, horizon=a.horizon, members=a.members, hidden=a.hidden, epochs=a.epochs, workers=a.workers,
                holdout_seeds=holdout)
    print(json.dumps({k: v for k, v in rep.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
