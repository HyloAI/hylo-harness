"""Shared helpers for the memory examples: arguments, output, paired statistics, and a flow policy with prior sampling.

Not an example itself. Everything here uses public Hylo APIs; the only policy internals touched are those of the
reference flow policy (`PriorFlowSampler`), the same way a policy adapter implements the prior protocol
(RFC-0003 `PriorSampler`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # the harness repository root
DEFAULT_ROOT = REPO / "runs" / "examples" / "memory"


# ---------------------------------------------------------------- command line and output

def parse_args(slug: str, doc: str | None, extra: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="a small smoke-sized run (< 60 s)")
    ap.add_argument("--out", default=str(DEFAULT_ROOT / slug), help="output directory")
    ap.add_argument("--workers", type=int, default=16, help="parallel episode workers (threads)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.out = Path(a.out).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def require_plugin(group: str, name: str) -> None:
    """Exit with a clear message when an optional plugin (e.g. the MuJoCo envs of hylo-bench) is not installed."""
    from hylo.core.registry import registry

    try:
        registry.resolve(group, name)
    except Exception as exc:                          # noqa: BLE001 - missing package or a broken import
        raise SystemExit(f"this example needs the {group} plugin {name!r} (hylo-bench's hylo-env-mujoco): {exc}")


def write_result(out: Path, result: Mapping[str, Any]) -> Path:
    path = out / "result.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def print_table(rows: list[Mapping[str, Any]], cols: list[str] | None = None, title: str | None = None) -> None:
    if not rows:
        return
    cols = cols or list(rows[0])
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    width = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    if title:
        print(f"\n{title}")
    print("  ".join(c.ljust(w) for c, w in zip(cols, width)))
    print("  ".join("-" * w for w in width))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, width)))


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if 0 < abs(v) < 0.001:
            return f"{v:.1e}"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)


# ---------------------------------------------------------------- statistics

def wilson(k: float, n: int) -> list[float] | None:
    from hylo.core.scorecard import wilson_interval

    ci = wilson_interval(k, n)
    return list(ci) if ci else None


def mcnemar_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs (b: only A succeeded, c: only B succeeded)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def rate(values: Iterable[float | None]) -> dict[str, Any]:
    v = [float(x) for x in values if x is not None]
    k, n = sum(v), len(v)
    return {"success": round(k / n, 4) if n else None, "ci95": wilson(k, n), "n": n}


def paired(a: Mapping[str, float | None], b: Mapping[str, float | None]) -> dict[str, Any]:
    """Paired comparison of B against A on shared keys (task id + seed): B-only wins, A-only wins, two-sided p."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    b_wins = sum(1 for k in keys if b[k] and not a[k])
    a_wins = sum(1 for k in keys if a[k] and not b[k])
    diff = (sum(float(b[k]) for k in keys) - sum(float(a[k]) for k in keys)) / max(1, len(keys))
    return {"n_pairs": len(keys), "diff": round(diff, 4), "wins": b_wins, "losses": a_wins,
            "p_two_sided": float(f"{mcnemar_two_sided(b_wins, a_wins):.4g}")}


# ---------------------------------------------------------------- MuJoCo planar reach: demonstrations and a flow policy

def reach_config(agent: dict[str, Any], *, rsi_states: int, workers: int, name: str = "memory-example",
                 **bench_params: Any) -> Any:
    from hylo.core.config import ExperimentConfig

    params = {"protocol": "rsi", "rsi_states": rsi_states, **bench_params}
    return ExperimentConfig(name=name, benchmark={"type": "mujoco/reach-v0", "params": params},
                            env={"type": "mujoco.planar_reach"}, agent=agent, workers=workers)


def train_flow_policy(out: Path, *, rsi_states: int, epochs: int, workers: int) -> Path:
    """Expert (IK controller) demonstrations on the RFC-0002 search split -> a sealed policy.flow artifact."""
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.learn.jobs import run_job

    art = out / f"flow-r{rsi_states}-e{epochs}"          # training is deterministic: reuse an artifact of these settings
    if (art / "policy.json").exists():
        return art
    expert = reach_config({"type": "baseline.planar_ik", "params": {"kp": 0.3}}, rsi_states=rsi_states, workers=workers)
    run_experiment(expert, split="search", run_root=out / f"demos-r{rsi_states}", label="expert-search")
    run_job(TrainJob(learner="policy.flow", inputs=[str(out / f"demos-r{rsi_states}")], out=str(art),
                     params={"chunk": 8, "epochs": epochs, "integration_steps": 10}))
    return art


class PriorFlowSampler:
    """The reference flow policy (policy.flow) with the prior protocol of RFC-0003, so MemorySteeredAgent can drive it.

    `sample(obs, seeds, noise_scale, prior_fn)` computes the retrieval key (the policy's own normalised observation
    vector), asks `prior_fn` for a PriorDecision and, if it returns a prior, starts the flow integration at
    tau0 = 1 - t0 from x = tau0 * prior + (1 - tau0) * noise (the reference flow runs from noise at tau = 0 to data
    at tau = 1). With no prior, or t0 = 1, it is the unchanged policy. `salt` changes the noise seeds, e.g. per
    session, while keeping them deterministic.

    The planar arm has no gripper, but elite_prior treats the last action dimension as one; `GRIP` is a constant
    gripper channel appended to stored segments and dropped again here.
    """

    GRIP = -1.0

    def __init__(self, path: str, salt: int = 0) -> None:
        from hylo.core.policy import build_policy

        self.policy = build_policy({"type": "policy.flow", "params": {"path": str(path)}})
        self.cfg = SimpleNamespace(chunk_size=self.policy.chunk)
        self.salt = int(salt)

    def spec(self):
        return self.policy.spec()

    def reset(self, ctx) -> None:
        self.policy.reset(ctx)

    def default_execution_steps(self) -> int:
        return self.policy.default_execution_steps()

    def key(self, state: Mapping[str, Any]) -> np.ndarray:
        p = self.policy
        return p.obs_norm(np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1) for k in p.keys]))

    def sample(self, observation, seeds, noise_scale: float = 1.0, prior_fn=None) -> np.ndarray:
        from hylo.policies.reference.flow import tau_features

        p = self.policy
        x = self.key(observation.state)
        seeds = [(int(s) + 7919 * self.salt) % 2 ** 32 for s in seeds] if seeds else None
        rngs = [np.random.default_rng(s) for s in seeds] if seeds else [np.random.default_rng()]
        z = np.stack([r.standard_normal(p.chunk * p.action_dim) for r in rngs]) * noise_scale
        start = 0
        if prior_fn is not None:
            d = prior_fn(x)
            if d.prior is not None and d.t0 < 1.0 - 1e-6:
                prior = np.asarray(d.prior, dtype=np.float64)[:, : p.action_dim]
                if len(prior) < p.chunk:                       # pad a short prior with its last action
                    prior = np.concatenate([prior, np.repeat(prior[-1:], p.chunk - len(prior), axis=0)])
                start = min(p.steps - 1, int(round((1.0 - d.t0) * p.steps)))
                tau0 = start / p.steps
                z = tau0 * p.act_norm(prior[: p.chunk]).reshape(1, -1) + (1.0 - tau0) * z
        obs = np.repeat(x[None], len(rngs), axis=0)
        for i in range(start, p.steps):
            tau = np.full(len(rngs), i / p.steps)
            z = z + p.net(np.concatenate([z, tau_features(tau), obs], axis=1)) / p.steps
        return p.act_norm.inverse(z.reshape(len(rngs), p.chunk, p.action_dim))


def segments_from_run(run_dir: Path, sampler: PriorFlowSampler, *, splits: tuple[str, ...]) -> list[Any]:
    """Verified action segments (successful episodes of development splits only): key = the policy's normalised
    observation at each decision, actions = the chunk the agent produced there (plus the constant gripper channel)."""
    from hylo.core.experience import ActionSegment, EpisodeRef
    from hylo.core.trace import TraceStore

    log = json.loads((run_dir / "evallog.json").read_text())
    store = TraceStore(run_dir)
    segs = []
    for s in log["samples"]:
        if s["status"] != "success":
            continue
        trace = store.read(s["episode_id"])
        split = trace.first("episode.start")["payload"].get("split")
        if split not in splits:
            continue
        ref = EpisodeRef(run_id=log["run_id"], episode_id=s["episode_id"], split=split)
        total, step, state = max(1, int(s["steps"])), 0, None
        for e in trace.events:
            if e["type"] == "observation":
                state = e["payload"]["state"]
            elif e["type"] == "step.result":
                step += 1
            elif e["type"] == "agent.chunk" and state is not None:
                a = np.asarray(e["payload"]["actions"], dtype=np.float64)
                a = np.concatenate([a, np.full((len(a), 1), PriorFlowSampler.GRIP)], axis=1)
                segs.append(ActionSegment(key=sampler.key(state), actions=a, task_id=s["task_id"], step=step,
                                          progress=min(1.0, step / total), source=ref))
    return segs


def evaluate_agent(cfg: Any, split: str, agent_factory: Callable[[], Any], run_root: Path, label: str) -> Any:
    """Run one split with an agent built in code (one per worker); returns the EvalResult."""
    from hylo.core.registry import registry
    from hylo.eval.experiment import env_factory
    from hylo.eval.runner import evaluate

    import shutil

    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    shutil.rmtree(run_root / f"{label}-{split}", ignore_errors=True)      # deterministic re-runs replace the old run
    return evaluate(bench, split, env_factory=env_factory(cfg), agent_factory=agent_factory, run_root=run_root,
                    run_id=f"{label}-{split}", workers=cfg.workers, record_privileged=False)
