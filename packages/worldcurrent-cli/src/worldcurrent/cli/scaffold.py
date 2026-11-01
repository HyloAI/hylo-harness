"""`wc plugin new GROUP NAME`: a working plugin package to start from, and `wc plugin check GROUP NAME`.

Every template passes its conformance check as generated; replace the body with your method. The package is an
ordinary Python distribution with an entry point, so worldcurrent finds it once it is installed
(`uv pip install -e <dir>` or add it to a workspace). See docs/extending.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from worldcurrent.core.plugins import CONTRACTS

# group -> (entry-point group, class name, module source, test source). {cls} / {name} / {module} are filled in.
TEMPLATES: dict[str, tuple[str, str, str, str]] = {
    "guard": ("guards", "{cls}Guard", '''"""Guard plugin: reviews every action between the shield's mandatory finite and bounds checks."""

from __future__ import annotations

import numpy as np

from worldcurrent.core.safety import GuardVerdict, SafetyContext


class {cls}Guard:
    name = "{name}"

    def __init__(self, max_abs: float = 0.5) -> None:
        self.max_abs = float(max_abs)

    def review(self, data: np.ndarray, ctx: SafetyContext) -> GuardVerdict:
        # ctx.state: latest observation state (may be None); ctx.last_action: previous approved action
        limited = np.clip(data, -self.max_abs, self.max_abs)
        changed = not np.array_equal(limited, data)
        return GuardVerdict(limited, changed, "clamped to max_abs" if changed else "")
''', '''from worldcurrent.testing.conformance import check_guard
from {module}.plugin import {cls}Guard


def test_conformance():
    rep = check_guard({cls}Guard())
    assert rep.ok, rep.summary()
'''),
    "gate_test": ("gate_tests", "{cls}Test", '''"""Gate-test plugin: the paired statistic the improvement protocol decides with (H1: the candidate is better)."""

from __future__ import annotations

from worldcurrent.core.scoring import PairedComparison
from worldcurrent.eval.scoring import one_sided_exact_mcnemar


class {cls}Test:
    name = "{name}"

    def __init__(self, metric: str = "success", direction: str = "higher") -> None:
        self.metric, self.sign = metric, 1.0 if direction == "higher" else -1.0

    def compare(self, candidate, parent) -> PairedComparison:
        keys = [k for k in candidate if k in parent and candidate[k] is not None and parent[k] is not None]
        d = [self.sign * (float(candidate[k]) - float(parent[k])) for k in keys]
        wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
        return PairedComparison(n_pairs=len(keys), candidate_wins=wins, parent_wins=losses,
                                p_value=one_sided_exact_mcnemar(wins, losses))
''', '''from worldcurrent.eval.conformance import check_gate_test
from {module}.plugin import {cls}Test


def test_conformance():
    rep = check_gate_test({cls}Test())
    assert rep.ok, rep.summary()
'''),
    "learner": ("learners", "{cls}Learner", '''"""Learner plugin: development runs in, one sealed artifact out."""

from __future__ import annotations

import json
from pathlib import Path

from worldcurrent.core.artifact import ArtifactManifest, Provenance, development_runs, write_manifest


class {cls}Learner:
    name = "{name}@0.1"

    def fit(self, job) -> ArtifactManifest:
        sources = development_runs(job.inputs, job.splits)           # gate / held-out are refused or skipped
        episodes = [s for rd, _ in sources for s in json.loads((rd / "evallog.json").read_text())["samples"]]
        out = Path(job.out)
        out.mkdir(parents=True, exist_ok=True)
        rate = sum(s["status"] == "success" for s in episodes) / max(1, len(episodes))
        (out / "model.json").write_text(json.dumps({{"success_rate": rate}}))   # your model here
        return write_manifest(out, kind="other", role="behaviour", metrics={{"episodes": len(episodes), "success_rate": rate}},
                              provenance=Provenance(builder=self.name, sources=[r for _, r in sources], params=dict(job.params)))
''', '''from worldcurrent.learn.conformance import check_learner


def test_conformance():
    rep = check_learner("{module}.plugin:{cls}Learner")
    assert rep.ok, rep.summary()
'''),
    "online_learner": ("learners", "{cls}Online", '''"""Online learner plugin: propose agent specs, see their rollouts on a development split, update; then seal."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from worldcurrent.core.artifact import ArtifactManifest, Provenance, write_manifest


class {cls}Online:
    name = "{name}@0.1"

    def start(self, job, agent) -> None:
        self.job, self.base = job, agent
        self.path = job.params.get("path", "params.kp")                # the agent parameter to search
        self.low, self.high = job.params.get("low", 0.1), job.params.get("high", 4.0)
        self.rng, self.best, self.tried = np.random.default_rng(0), None, []

    def _spec(self, v: float):
        spec = copy.deepcopy(self.base)
        node = spec
        *parents, leaf = self.path.split(".")
        for k in parents:
            node = node.setdefault(k, {{}})
        node[leaf] = float(v)
        return spec

    def policies(self):
        self.tried = list(self.rng.uniform(self.low, self.high, 3))  # random search; your method here
        return [self._spec(v) for v in self.tried]

    def update(self, results):
        for v, r in zip(self.tried, results):
            score = float(r["metrics"]["success_mean"])
            if self.best is None or score > self.best[0]:
                self.best = (score, v)
        return {{"best_score": self.best[0], "best": self.best[1]}}

    def finished(self) -> bool:
        return False

    def seal(self, out, sources) -> ArtifactManifest:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "best.json").write_text(json.dumps({{"value": self.best[1], "score": self.best[0]}}))
        return write_manifest(out, kind="policy_params", role="behaviour", metrics={{"best": self.best[1], "score": self.best[0]}},
                              provenance=Provenance(builder=self.name, sources=sources,
                                                    params={{k: v for k, v in self.job.params.items() if k != "experiment"}}))
''', '''from worldcurrent.learn.conformance import check_learner


def test_conformance():
    rep = check_learner("{module}.plugin:{cls}Online")
    assert rep.ok, rep.summary()
'''),
    "datasource": ("datasources", "{cls}Source", '''"""Data-source plugin: external episodes (demonstrations, videos, other robots) with origin and licence."""

from __future__ import annotations

import numpy as np

from worldcurrent.core.datasource import ExternalEpisode, ExternalFrame


class {cls}Source:
    name = "{name}"

    def __init__(self, n: int = 3, licence: str = "CC-BY-4.0") -> None:
        self.n, self.licence, self.origin = n, licence, "synthetic://{name}"   # where the data comes from

    def episodes(self):
        for i in range(self.n):                                     # read your files here
            frames = [ExternalFrame(state={{"pos": np.array([t / 10, 0.0])}}, images={{"cam": np.zeros((16, 16, 3), np.uint8)}})
                      for t in range(10)]
            yield ExternalEpisode(episode_id=f"{name}-{{i}}", task_id="{name}/task", instruction="do the task", frames=frames)
''', '''from worldcurrent.learn.conformance import check_datasource
from {module}.plugin import {cls}Source


def test_conformance():
    rep = check_datasource({cls}Source())
    assert rep.ok, rep.summary()
'''),
    "judge": ("instruments", "{cls}Judge", '''"""Instrument plugin: a success judge (labels episodes; trusted only after `wc calibrate`)."""

from __future__ import annotations

from worldcurrent.core.instrument import Judgement


class {cls}Judge:
    name = "{name}@0.1"

    def judge(self, episode) -> Judgement:
        last = None
        for ev in episode.of_type("observation"):                   # only what a real robot records
            last = ev["payload"]
        return Judgement(p_success=0.5, info={{"has_frame": bool(last and last.get("images"))}})   # your model here
''', '''from worldcurrent.learn.conformance import check_judge
from {module}.plugin import {cls}Judge


def test_conformance():
    rep = check_judge({cls}Judge())
    assert rep.ok, rep.summary()
'''),
    "generator": ("generators", "{cls}Proposer", '''"""Generator plugin: proposes config patches (candidates), each with a hypothesis; the gate decides."""

from __future__ import annotations

from worldcurrent.core.improvement import Candidate


class {cls}Proposer:
    name = "{name}@0.1"

    def __init__(self, path: str = "agent.params.kp", factors: tuple[float, ...] = (1.5,)) -> None:
        self.path, self.factors = path, factors

    def propose(self, diagnosis, cfg, parent_id, archive):
        node = cfg.model_dump(mode="json")
        for k in self.path.split("."):
            node = node[k]
        return [(Candidate(patch={{self.path: float(node) * f}}, rationale=f"scale by {{f}}", parent_id=parent_id,
                           generator=self.name), f"scaling {{self.path}} by {{f}} reduces the failures in the diagnosis")
                for f in self.factors]
''', '''from worldcurrent.evolve.conformance import check_generator
from {module}.plugin import {cls}Proposer


def test_conformance():
    rep = check_generator({cls}Proposer())
    assert rep.ok, rep.summary()
'''),
    "policy": ("policies", "{cls}Policy", '''"""Policy plugin (core.policy.ChunkPolicy): draws action chunks, one per noise seed.

Run it:      agent: {{type: policy, params: {{policy: {{type: {name}}}}}}}
Steer it:    agent: {{type: steered, params: {{sampler: {{type: {name}}}, steering: {{n_samples: 4, selector: medoid}}}}}}
ACT-style:   agent: {{type: temporal_ensemble, params: {{policy: {{type: {name}}}, m: 0.1}}}}
Draw all randomness from the seeds, so episodes replay and paired comparisons share noise. A learned policy loads
its weights in __init__ (e.g. from an artifact path) and declares weights_license in its spec.

As generated it drives MuJoCo planar reach (worldcurrent-bench) with analytic two-link IK. For your robot, change
required_state, action_semantics / action_dim and sample(), and check it with
`wc plugin check policy {name} --experiment your_experiment.yaml`.
"""

from __future__ import annotations

import numpy as np

from worldcurrent.core.agent import AgentSpec, EpisodeContext
from worldcurrent.core.types import ActionSemantics, Observation


class {cls}Policy:
    def __init__(self, gain: float = 0.3, horizon: int = 8, noise: float = 0.05, l1: float = 0.3, l2: float = 0.25,
                 max_step: float = 0.05) -> None:
        self.gain, self.horizon, self.noise, self.l1, self.l2, self.max_step = gain, horizon, noise, l1, l2, max_step

    def spec(self) -> AgentSpec:
        # what it reads and what its actions mean: checked against the environment before the first step
        return AgentSpec(name="{name}", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"), action_dim=2,
                         components={{"policy": "{name}@0.1"}})

    def reset(self, ctx: EpisodeContext) -> None:
        pass

    def default_execution_steps(self) -> int:
        return max(1, self.horizon // 2)

    def _ik(self, x: float, y: float) -> np.ndarray:
        c2 = np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        return np.array([np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)), q2])

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        s = observation.state
        err = (self._ik(*np.asarray(s["target_pos"], float)[:2]) - np.asarray(s["qpos"], float) + np.pi) % (2 * np.pi) - np.pi
        a = np.clip(self.gain * err / self.max_step, -1.0, 1.0)
        rngs = [np.random.default_rng(x) for x in seeds] if seeds else [np.random.default_rng()]
        return np.stack([np.clip(a + self.noise * noise_scale * r.standard_normal((self.horizon, 2)), -1.0, 1.0)
                         for r in rngs])                                    # (N, T, A)
''', '''import pytest

pytest.importorskip("worldcurrent.envs.mujoco", reason="checked on MuJoCo planar reach (worldcurrent-bench)")

from worldcurrent.agents.policy_agents import PolicyAgent  # noqa: E402
from worldcurrent.core.kernel import EpisodeRunner  # noqa: E402
from worldcurrent.testing.conformance import check_policy  # noqa: E402
from worldcurrent.testing.setup import check_experiment, env_and_task  # noqa: E402
from {module}.plugin import {cls}Policy  # noqa: E402


def test_conformance():
    env, task = env_and_task(check_experiment(prefer="mujoco"))
    rep = check_policy({cls}Policy(), env, task)
    assert rep.ok, rep.summary()


def test_it_reaches_the_target():
    env, task = env_and_task(check_experiment(prefer="mujoco"))
    assert EpisodeRunner(env, PolicyAgent({cls}Policy())).run(task).outcome.status == "success"
'''),
    "agent": ("agents", "{cls}Agent", '''"""Agent plugin: what the kernel talks to. spec / reset / act / on_step / on_episode_end.

Write an agent when the decision logic is more than drawing chunks from one model (planning, memory, switching
between policies, a remote service); otherwise write a policy plugin and run it with the "policy" agent.

As generated it drives MuJoCo planar reach (worldcurrent-bench) with analytic two-link IK. For your robot, change
required_state, action_semantics / action_dim and act(), and check it with
`wc plugin check agent {name} --experiment your_experiment.yaml`.
"""

from __future__ import annotations

import numpy as np

from worldcurrent.core.agent import AgentBase, AgentSpec, EpisodeContext
from worldcurrent.core.types import ActionChunk, ActionSemantics, Observation


class {cls}Agent(AgentBase):
    def __init__(self, gain: float = 0.3, horizon: int = 4, l1: float = 0.3, l2: float = 0.25, max_step: float = 0.05) -> None:
        self.gain, self.horizon, self.l1, self.l2, self.max_step = gain, horizon, l1, l2, max_step

    def spec(self) -> AgentSpec:
        return AgentSpec(name="{name}", version="0.1", required_state=("qpos", "target_pos"),
                         action_semantics=ActionSemantics(control_mode="joint_delta", frame="base"), action_dim=2,
                         components={{"policy": "{name}@0.1"}})

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx                                  # ctx.emit(type, payload) writes into the trace

    def act(self, observation: Observation) -> ActionChunk:
        s = observation.state
        x, y = np.asarray(s["target_pos"], float)[:2]
        c2 = np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        q = np.array([np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)), q2])
        err = (q - np.asarray(s["qpos"], float) + np.pi) % (2 * np.pi) - np.pi
        a = np.clip(self.gain * err / self.max_step, -1.0, 1.0)
        return ActionChunk(actions=np.repeat(a[None], self.horizon, axis=0), execution_steps=self.horizon,
                           provenance={{"model": "{name}"}})
''', '''import pytest

pytest.importorskip("worldcurrent.envs.mujoco", reason="checked on MuJoCo planar reach (worldcurrent-bench)")

from worldcurrent.testing.conformance import check_agent  # noqa: E402
from worldcurrent.testing.setup import check_experiment, env_and_task  # noqa: E402
from {module}.plugin import {cls}Agent  # noqa: E402


def test_conformance():
    env, task = env_and_task(check_experiment(prefer="mujoco"))
    rep = check_agent({cls}Agent(), env, task)
    assert rep.ok, rep.summary()
'''),
}

DEPS = {"guard": ["wc-core"], "gate_test": ["wc-core", "wc-eval"], "learner": ["wc-core", "wc-learn"],
        "online_learner": ["wc-core", "wc-learn"], "datasource": ["wc-core", "wc-learn"], "judge": ["wc-core", "wc-learn"],
        "generator": ["wc-core", "wc-evolve"], "policy": ["wc-core", "wc-agents"], "agent": ["wc-core"]}


def new_plugin(group: str, name: str, directory: str | Path = "plugins") -> Path:
    if group not in TEMPLATES:
        raise ValueError(f"unknown group {group!r}; one of {sorted(TEMPLATES)}")
    if not re.fullmatch(r"[a-z][a-z0-9_.-]*", name):
        raise ValueError("plugin names are lowercase: letters, digits, '_', '-', '.'")
    module = "wc_" + re.sub(r"[^a-z0-9]+", "_", name)
    cls = "".join(p.capitalize() for p in re.split(r"[^a-z0-9]+", name) if p)
    ep_group, cls_tpl, src, test = TEMPLATES[group]
    fill = lambda t: t.replace("{cls}", cls).replace("{name}", name).replace("{module}", module).replace("{{", "{").replace("}}", "}")  # noqa: E731
    root = Path(directory) / f"wc-plugin-{re.sub(r'[^a-z0-9]+', '-', name)}"
    if root.exists():
        raise FileExistsError(f"{root} exists")
    (root / "src" / module).mkdir(parents=True)
    (root / "tests").mkdir()
    deps = json.dumps([*DEPS[group], "numpy>=1.24"])
    (root / "pyproject.toml").write_text(f'''[project]
name = "wc-plugin-{re.sub(r'[^a-z0-9]+', '-', name)}"
version = "0.1.0"
description = "worldcurrent {group} plugin: {name}"
requires-python = ">=3.10"
dependencies = {deps}

[project.entry-points."worldcurrent.{ep_group}"]
"{name}" = "{module}.plugin:{fill(cls_tpl)}"

[project.entry-points."worldcurrent.meta"]
"wc-plugin-{re.sub(r'[^a-z0-9]+', '-', name)}" = "{module}:META"

[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{module}"]
''')
    (root / "src" / module / "__init__.py").write_text(
        "# the contract version this plugin implements and how far it has been verified (worldcurrent.core.plugins);\n"
        "# set maturity to implemented once it has run end to end on its target, and add `license` to pyproject.toml\n"
        f'META = {{"contracts": {{"{ep_group}": "{CONTRACTS[ep_group]}"}}, "maturity": "unverified"}}\n')
    (root / "src" / module / "plugin.py").write_text(fill(src))
    (root / "tests" / f"test_{module}.py").write_text(fill(test))
    return root


def declaration_warnings(group: str, name: str) -> list[str]:
    """What is missing from a plugin's declaration (contract version, maturity, licence)."""
    from importlib.metadata import entry_points

    from worldcurrent.core.plugins import describe

    ep_group = {"guard": "guards", "gate_test": "gate_tests", "learner": "learners", "online_learner": "learners",
                "datasource": "datasources", "judge": "instruments", "generator": "generators", "policy": "policies",
                "agent": "agents"}.get(group, group)
    eps = [ep for ep in entry_points(group=f"worldcurrent.{ep_group}") if ep.name == name]
    if not eps:
        return []                                       # a module:attr reference: nothing declared to check
    m = describe(ep_group, eps[0])
    out = []
    if m.contract is None:
        out.append(f"{m.distribution} declares no contract version for {ep_group} (a worldcurrent.meta entry point)")
    elif m.compatible is False:
        out.append(f"{name} implements {m.contract}; this core implements {m.expected}")
    if m.maturity == "undeclared":
        out.append(f"{name} declares no maturity (implemented / unverified / needs-hardware)")
    if not m.licence:
        out.append(f"{m.distribution} declares no licence (license in pyproject.toml)")
    return out


def check_plugin(group: str, name: str, params: dict[str, Any] | None = None, experiment: str | None = None):
    """Build a registered plugin (or module:attr) and run its group's conformance check. Environment-dependent checks
    run on `experiment` (a config path), else MuJoCo planar reach when installed, else the test double."""
    from worldcurrent.core.registry import registry
    from worldcurrent.testing.setup import check_experiment

    params = params or {}
    cfg = check_experiment(experiment)
    if group == "guard":
        from worldcurrent.testing.conformance import check_guard

        return check_guard(registry.build("guards", {"type": name, "params": params}))
    if group == "gate_test":
        from worldcurrent.eval.conformance import check_gate_test

        return check_gate_test(registry.build("gate_tests", {"type": name, "params": params}))
    if group in ("learner", "online_learner"):
        from worldcurrent.learn.conformance import check_learner

        return check_learner(name, params, experiment=cfg)
    if group == "datasource":
        from worldcurrent.learn.conformance import check_datasource

        return check_datasource(registry.build("datasources", {"type": name, "params": params}))
    if group == "judge":
        from worldcurrent.learn.conformance import check_judge

        return check_judge(registry.build("instruments", {"type": name, "params": params}), experiment=cfg)
    if group == "generator":
        from worldcurrent.evolve.conformance import check_generator

        return check_generator(registry.build("generators", {"type": name, "params": params}), cfg=cfg)
    if group in ("policy", "agent"):
        from worldcurrent.testing.conformance import check_agent, check_policy
        from worldcurrent.testing.setup import env_and_task

        env, task = env_and_task(cfg)
        try:
            if group == "policy":
                return check_policy(registry.build("policies", {"type": name, "params": params}), env, task)
            return check_agent(registry.build("agents", {"type": name, "params": params}), env, task)
        finally:
            getattr(env, "close", lambda: None)()
    raise ValueError(f"no check for {group!r} here; environments: wc conformance CONFIG")
