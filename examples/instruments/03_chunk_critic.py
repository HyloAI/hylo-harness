"""Chunk value model: can a critic learned from traces tell good action chunks from bad ones, and does steering help?

A chunk critic (hylo.instruments.critic.ChunkCritic, the value model behind CriticSteeredAgent) predicts the
discounted success-to-go of executing an action chunk in a state. Here it is trained and tested on MuJoCo planar
reach, end to end:

1. a flow-matching policy (policy.flow) is trained from IK-controller demonstrations;
2. behaviour data: the flow policy runs on the development splits with three noise scales (1.0, 1.4, 1.8), so the
   same init state is tried with different chunks and different outcomes; each `agent.chunk` event with the state
   before it is one sample (hylo.learn.critic.collect);
3. offline: grouped 5-fold cross-validation over init states, AUC of the predicted value against the episode outcome
   for the critic (state + chunk) and for a state-only model, and the accuracy on same-state pairs (two first
   chunks from the identical reset state, one led to success and one to failure): the cases where re-ranking chunks
   could change the outcome;
4. online: the flow policy steered with 4 seeded samples per decision, selecting the first sample, the medoid, or
   the sample the critic scores highest (CriticSteeredAgent, beta = 0.5), on held-out init states.

    uv run python examples/instruments/03_chunk_critic.py [--quick] [--out DIR]

Runtime: about 90 s on 16 processes (quick: about 12 s). Requires hylo-env-mujoco (bench), hylo-learn and
hylo-agents. Note: the critic's chunk summary was designed for 7-D end-effector actions with a gripper; on the 2-D
joint actions here it sees net motion of joint 1 and the mean command of joint 2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (first: it caps BLAS threads before numpy loads)

import numpy as np  # noqa: E402

SLUG = "chunk_critic"
STATE_KEYS = ("qpos", "qvel", "tip_pos", "target_pos")
NOISE = (1.0, 1.4, 1.8)
N_CHUNK = 5                  # chunk_features length for 2-D actions: prefix sum, prefix mean, run sum, run mean, flips


def grouped_cv(eps: list[dict], cfg, kw: dict, folds: int = 5) -> dict:
    from hylo.instruments.critic import ChunkCritic
    from hylo.learn.critic import _stack, auc, same_state_pairs

    seeds = sorted({e["seed"] for e in eps})
    fold_of = {s: i % folds for i, s in enumerate(seeds)}
    pairs = same_state_pairs(eps)
    pf, ps, lab, wins, n_pairs = [], [], [], 0, 0
    for k in range(folds):
        tr = [e for e in eps if fold_of[e["seed"]] != k]
        te = [e for e in eps if fold_of[e["seed"]] == k]
        xt, yt, _, wt = _stack(tr)
        xv, _, sv, _ = _stack(te)
        n_state = xt.shape[1] - N_CHUNK
        full = ChunkCritic.fit(xt, yt, cfg, weights=wt, **kw)
        state_only = ChunkCritic.fit(xt[:, :n_state], yt, cfg, weights=wt, **kw)
        pf.append(full.predict(xv)[0])
        ps.append(state_only.predict(xv[:, :n_state])[0])
        lab.append(sv)
        for sa, _, xa, xb in pairs:
            if fold_of[sa] == k:
                va, vb = full.predict(np.stack([xa, xb]))[0]
                n_pairs += 1
                wins += va > vb
    pf, ps, lab = np.concatenate(pf), np.concatenate(ps), np.concatenate(lab)
    acc = float(wins) / n_pairs if n_pairs else None
    return {"folds": folds, "chunks": int(len(lab)), "auc_state_chunk": round(auc(pf, lab), 3),
            "auc_state_only": round(auc(ps, lab), 3), "same_state_pairs": n_pairs,
            "same_state_pair_accuracy": None if acc is None else round(acc, 3),
            "pair_accuracy_se": None if acc is None else round(float(np.sqrt(acc * (1 - acc) / n_pairs)), 3)}


def main() -> int:
    a = C.parse_args(__doc__, SLUG)
    t0 = time.perf_counter()
    states = 50 if a.quick else 250
    bench = C.reach_bench(states)
    roots = C.demonstrations(a.out, bench, workers=a.workers)
    flow, info = C.train("flow", [roots["demos"]], a.out / "artifacts" / "flow", {"chunk": 8, "execution_steps": 4})
    C.log(f"policy.flow: {info['episodes']} demonstrations, {info['seconds']} s")
    pol = C.policy_ref("flow", flow)

    behaviour = {f"flow-noise{s}": {"type": "steered", "params": {"sampler": pol, "steering": {"noise_scale": s}}}
                 for s in NOISE}
    beh_runs: dict[str, list[Path]] = {}
    for split in ("search", "search_val"):
        for k, v in C.run_many(behaviour, bench, split, a.out / "behaviour" / split, procs=a.workers).items():
            beh_runs.setdefault(k, []).extend(v)
    beh = {k: C.summarize(v) for k, v in beh_runs.items()}
    C.log("behaviour data: " + ", ".join(f"{k} {100 * s['success']:.0f}% of {s['episodes']}" for k, s in beh.items()))

    from hylo.instruments.critic import ChunkCritic, FeatureConfig
    from hylo.learn.critic import collect

    run_dirs = [d for v in beh_runs.values() for d in v]
    cfg = FeatureConfig(state_keys=STATE_KEYS, horizon=150, prefix_steps=4,
                        tasks=("mujoco/planar_reach_fixed", "mujoco/planar_reach_random"))
    eps = collect(run_dirs, cfg, gamma=0.98, splits=("search", "search_val"), task_prefix=None, workers=a.workers)
    kw = {"members": 3 if a.quick else 5, "hidden": 64, "epochs": 20 if a.quick else 60, "seed": 0}
    t = time.perf_counter()
    cv = grouped_cv(eps, cfg, kw)
    C.log(f"offline ({len(eps)} episodes, {time.perf_counter() - t:.0f} s): {cv}")
    from hylo.learn.critic import _stack

    x, y, _, w = _stack(eps)
    critic = ChunkCritic.fit(x, y, cfg, weights=w, **kw)
    critic_dir = a.out / "artifacts" / "critic"
    digest = critic.save(critic_dir)

    steer = lambda sel: {"n_samples": 4, "selector": sel, "execution_steps": 4}  # noqa: E731
    online = {
        "1 sample": {"type": "steered", "params": {"sampler": pol, "steering": {"execution_steps": 4}}},
        "4 samples, first": {"type": "steered", "params": {"sampler": pol, "steering": steer("first")}},
        "4 samples, medoid": {"type": "steered", "params": {"sampler": pol, "steering": steer("medoid")}},
        "4 samples, critic": {"type": "hylo.agents.critic:CriticSteeredAgent", "params": {
            "sampler": pol, "steering": steer("medoid"),
            "critic": {"path": str(critic_dir), "beta": 0.5, "within_vote": False}}},
    }
    runs = C.run_many(online, bench, "heldout", a.out / "runs", procs=a.workers)
    base = C.per_key(runs["4 samples, medoid"])
    rows = []
    for name, dirs in runs.items():
        s = C.summarize(dirs)
        vs = C.paired(C.per_key(dirs), base)
        rows.append({"selector": name, "success_%": round(100 * s["success"], 1),
                     "ci95_%": f"{100 * s['ci95'][0]:.0f}-{100 * s['ci95'][1]:.0f}", "steps": s["steps"],
                     "policy_ms": s["policy_ms"], "episodes": s["episodes"], "vs_medoid": vs,
                     "vs medoid": "-" if name.endswith("medoid") else vs["text"]})
    print()
    C.table([{"model": "critic (state + chunk)", "auc": cv["auc_state_chunk"]},
             {"model": "state only", "auc": cv["auc_state_only"]}], ["model", "auc"],
            title=f"Offline, grouped 5-fold CV over init states ({cv['chunks']} chunks); same-state pairs: "
                  f"{cv['same_state_pair_accuracy']} accuracy +- {cv['pair_accuracy_se']} (n={cv['same_state_pairs']})")
    print()
    C.table(rows, ["selector", "success_%", "ci95_%", "steps", "policy_ms", "vs medoid"],
            title=f"Online, flow policy on the held-out split ({rows[0]['episodes']} episodes)")
    C.write_result(a.out, {"example": f"instruments/{SLUG}", "quick": a.quick, "benchmark": bench,
                           "behaviour": beh, "offline": cv, "critic": {"path": str(critic_dir), "weights": digest, **kw},
                           "online": rows, "seconds": round(time.perf_counter() - t0, 1)})
    C.log(f"\nresult: {a.out / 'result.json'}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
