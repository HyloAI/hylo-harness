"""Build components from an ExperimentConfig through the plugin registry and run it.

Only core and eval are imported here; envs, agents and memory backends arrive as plugins.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable

from worldcurrent.core.agent import Agent
from worldcurrent.core.config import ExperimentConfig, build_manifest
from worldcurrent.core.env import Environment
from worldcurrent.core.registry import registry

from .runner import EvalResult, evaluate


def env_factory(cfg: ExperimentConfig) -> Callable[[], Environment]:
    spec = cfg.env
    if spec.remote is not None:
        from worldcurrent.core.remote.client import RemoteEnvironment

        ref = registry.reference("envs", spec.type)   # do not import: the env may only exist in its own venv
        return lambda: RemoteEnvironment(ref, spec.params, python=spec.remote.python, env=spec.remote.env,
                                                 runtime=spec.remote.runtime)
    return lambda: registry.build("envs", spec.model_dump())


def build_agent(cfg: ExperimentConfig) -> Agent:
    return registry.build("agents", cfg.agent.model_dump())


def run_experiment(cfg: ExperimentConfig, *, split: str | None = None, run_root: str | Path | None = None,
                   label: str | None = None, agent: Agent | None = None,
                   progress: Callable[[dict], None] | None = None) -> EvalResult:
    split = split or cfg.benchmark.split
    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    parallel = agent is None and cfg.workers > 1
    if parallel:
        probe = build_agent(cfg)
        aspec = probe.spec()
        close = getattr(probe, "close", None)
        if callable(close):
            close()
    else:
        agent = agent or build_agent(cfg)
        aspec = agent.spec()
    run_id = f"{cfg.name}-{label or split}-{cfg.digest()}-{uuid.uuid4().hex[:6]}"
    manifest = build_manifest(cfg, run_id=run_id, extra={
        "benchmark": {"id": bench.id, "version": bench.version, "digest": bench.digest(), "split": split},
        "agent_spec": aspec.model_dump(mode="json"),
    })
    return evaluate(bench, split, env_factory=env_factory(cfg), agent=None if parallel else agent,
                    agent_factory=(lambda: build_agent(cfg)) if parallel else None,
                    run_root=run_root or cfg.output_dir, safety=cfg.safety, manifest=manifest, run_id=run_id,
                    record_privileged=cfg.trace.record_privileged, workers=cfg.workers, progress=progress,
                    image_every=cfg.trace.image_every, timing=cfg.timing)
