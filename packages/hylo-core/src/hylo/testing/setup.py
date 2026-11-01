"""Where plugin checks run: a real simulator when one is installed, the test double otherwise.

`hylo plugin check` and the learning / generator conformance checks make their sample runs and environments from
`check_experiment()`: MuJoCo planar reach (hylo-bench's hylo-env-mujoco, with the analytic IK controller)
when it is registered, else the harness's test double, so a harness-only install can still check a plugin. Any
experiment config can be given instead (`hylo plugin check ... --experiment CONFIG.yaml`), e.g. your robot's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..core.config import ExperimentConfig
from ..core.registry import registry
from ..core.task import TaskInstance

MUJOCO = ExperimentConfig(name="conformance", benchmark={"type": "mujoco/reach-v0", "params": {"protocol": "rsi"}},
                          env={"type": "mujoco.planar_reach"}, agent={"type": "baseline.planar_ik", "params": {"kp": 0.3}},
                          workers=8)
DOUBLE = ExperimentConfig(name="conformance", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
                          env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 1.0}})


def _registered(group: str, name: str) -> bool:
    try:
        registry.reference(group, name)
        return True
    except KeyError:
        return False


def check_experiment(experiment: str | Path | ExperimentConfig | None = None,
                     prefer: Literal["auto", "mujoco", "double"] = "auto") -> ExperimentConfig:
    """The experiment checks run on: the given one, else MuJoCo planar reach if installed, else the test double."""
    if isinstance(experiment, ExperimentConfig):
        return experiment
    if experiment is not None:
        return ExperimentConfig.load(experiment)
    if prefer != "double" and _registered("envs", "mujoco.planar_reach"):
        return MUJOCO
    if prefer == "mujoco":
        raise RuntimeError("MuJoCo planar reach is not installed (hylo-bench: hylo-env-mujoco)")
    return DOUBLE


def env_and_task(cfg: ExperimentConfig) -> tuple[Any, TaskInstance]:
    """A fresh environment of the experiment and its first task instance (validation split, else the first split)."""
    if cfg.env.remote is not None:
        from ..core.remote.client import RemoteEnvironment

        env = RemoteEnvironment(registry.reference("envs", cfg.env.type), cfg.env.params, python=cfg.env.remote.python,
                                env=cfg.env.remote.env, runtime=cfg.env.remote.runtime)
    else:
        env = registry.build("envs", cfg.env.model_dump(exclude={"remote"}))
    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    split = cfg.benchmark.split if bench.splits.get(cfg.benchmark.split) else next(k for k, v in bench.splits.items() if v)
    return env, bench.instances(split)[0]


__all__ = ["DOUBLE", "MUJOCO", "check_experiment", "env_and_task"]
