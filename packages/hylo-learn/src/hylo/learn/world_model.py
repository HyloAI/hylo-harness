"""Learner "world_model.state" (RFC-0004 W2): fit agents.world_model.StateWorldModel on development traces.

    learner: {learner: world_model.state, params: {env: {type: testing.point_reach}, epochs: 80}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hylo.agents.world_model import StateWorldModel, _Reg, _vec
from hylo.core.artifact import ArtifactManifest, Provenance, check_splits, development_runs, write_manifest
from hylo.core.trace import TraceStore


def _transitions(run_dir: Path, episode_id: str, keys: list[str] | None):
    tr = TraceStore(run_dir).read(episode_id)
    obs, acts = [], []
    for e in tr.events:
        if e["type"] == "observation":
            obs.append(e["payload"]["state"])
        elif e["type"] == "action":
            acts.append(np.asarray(e["payload"]["data"], float).reshape(-1))
    if keys is None:
        keys = sorted(k for k, v in obs[0].items() if np.asarray(v).size <= 64)
    end = tr.last("episode.end")
    return keys, obs, acts, (end or {}).get("payload", {}).get("status"), (tr.first("episode.start") or {}).get("payload", {})


class StateWorldModelLearner:
    """Learner "world_model.state". params: env (spec of the real env, to copy its EnvSpec), state_keys, members,
    hidden, epochs, seed, max_episodes."""

    name = "hylo.learn.world_model@0.1"

    def fit(self, job) -> ArtifactManifest:
        from hylo.core.registry import registry

        p = dict(job.params)
        splits = check_splits(job.splits)
        sources = development_runs(job.inputs, splits)
        keys = p.get("state_keys")
        X, Y, S, D, inits, val = [], [], [], [], {}, []
        rng = np.random.default_rng(int(p.get("seed", 0)))
        n_eps = 0
        for rd, ref in sources:
            for smp in json.loads((rd / "evallog.json").read_text())["samples"]:
                if smp["status"] not in ("success", "failure"):
                    continue
                if p.get("max_episodes") and n_eps >= int(p["max_episodes"]):
                    break
                keys, obs, acts, status, start = _transitions(rd, smp["episode_id"], keys)
                vecs = [_vec(o, keys) for o in obs]
                n = min(len(acts), len(vecs) - 1)
                bucket = val if ref.split == splits[-1] and len(splits) > 1 else None
                for t in range(n):
                    (X if bucket is None else bucket).append((np.concatenate([vecs[t], acts[t]]), vecs[t + 1] - vecs[t]))
                for t, v in enumerate(vecs):
                    S.append(v)
                    D.append(1.0 if status == "success" and t == len(vecs) - 1 else 0.0)
                inits.setdefault(start.get("task_id", "?"), {})[str(start.get("seed"))] = vecs[0].tolist()
                n_eps += 1
        if not X:
            raise SystemExit("no transitions in the development runs")
        x = np.stack([a for a, _ in X])
        y = np.stack([b for _, b in X])
        s, d = np.stack(S), np.array(D)
        mean = {"x": x.mean(0), "dy": y.mean(0), "s": s.mean(0)}
        std = {"x": x.std(0) + 1e-6, "dy": y.std(0) + 1e-6, "s": s.std(0) + 1e-6}
        members, hidden, epochs = int(p.get("members", 3)), int(p.get("hidden", 64)), int(p.get("epochs", 60))
        dyn, done = [], []
        for i in range(members):
            r = np.random.default_rng(int(p.get("seed", 0)) + i)
            boot = r.integers(0, len(x), len(x))
            m = _Reg(x.shape[1], y.shape[1], hidden, r)
            m.fit(((x - mean["x"]) / std["x"])[boot], ((y - mean["dy"]) / std["dy"])[boot], epochs=epochs, rng=r)
            dyn.append(m.p)
            w = np.where(d > 0, max(1.0, (d == 0).sum() / max(1, (d > 0).sum())), 1.0)   # balance the rare positives
            c = _Reg(s.shape[1], 1, hidden, r)
            target = np.where(d > 0, 4.0, -4.0)[:, None]                                      # logit targets
            rep = np.repeat(np.arange(len(s)), np.minimum(w, 20).astype(int))
            c.fit(((s - mean["s"]) / std["s"])[rep], target[rep], epochs=max(10, epochs // 3), rng=r)
            done.append(c.p)
        first = _transitions(sources[0][0], json.loads((sources[0][0] / "evallog.json").read_text())["samples"][0]["episode_id"], keys)[1][0]
        dims = [int(np.asarray(first[k]).size) for k in keys]
        env_spec = None
        if p.get("env"):
            env = registry.build("envs", p["env"])
            try:
                env_spec = env.spec().model_dump(mode="json")
            finally:
                getattr(env, "close", lambda: None)()
        model = StateWorldModel(keys, dims, mean, std, dyn, done, {"env_spec": env_spec, "inits": inits})
        metrics: dict[str, Any] = {"transitions": len(x), "episodes": n_eps, "state_dim": int(s.shape[1])}
        if val:
            xv = np.stack([a for a, _ in val])
            yv = np.stack([b for _, b in val])
            pred = np.stack([model.predict(r[: s.shape[1]], r[s.shape[1]:])[0] - r[: s.shape[1]] for r in xv])
            metrics["val"] = {"transitions": len(xv), "mse": float(np.mean((pred - yv) ** 2)),
                              "mse_zero_baseline": float(np.mean(yv ** 2))}
        model.save(job.out)
        return write_manifest(job.out, kind="world_model", role="instrument", metrics=metrics,
                              provenance=Provenance(builder=self.name, sources=[r for _, r in sources],
                                                    params={k: v for k, v in p.items() if k != "env"} | {"env": p.get("env")}))


