"""Demonstrations from runs, and the artifact format and runtime shared by the reference policies.

An artifact directory holds `policy.json` (method, observation keys and sizes, chunk length, the demonstrator's
action semantics, hyper-parameters) and `policy.npz` (normalisers and parameters), sealed by the learner
(core.artifact). Nothing is pickled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldcurrent.core.agent import AgentSpec, EpisodeContext
from worldcurrent.core.trace import TraceStore
from worldcurrent.core.types import ActionSemantics, Observation

SCHEMA = "wc.reference_policy/v0.1"


@dataclass
class Demos:
    obs: np.ndarray            # (N, S) observation vectors
    chunks: np.ndarray         # (N, T, A) the actions taken from each observation on, padded with the last one
    keys: list[str]
    dims: list[int]
    semantics: dict[str, Any] | None
    episodes: int


def _vec(state: dict[str, Any], keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1) for k in keys])


def load_demos(sources: list[tuple[Path, Any]], *, chunk: int, keys: list[str] | None = None,
               statuses: tuple[str, ...] = ("success",), max_episodes: int | None = None) -> Demos:
    """Every (observation, next `chunk` actions) pair of the episodes whose status is in `statuses`. `keys`
    defaults to every small (<= 64 values) state entry of the first observation."""
    obs_rows, chunk_rows, semantics, episodes, dims = [], [], None, 0, None
    for run_dir, _ in sources:
        manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
        semantics = semantics or (manifest.get("agent_spec") or {}).get("action_semantics")
        store = TraceStore(run_dir)
        for s in json.loads((run_dir / "evallog.json").read_text())["samples"]:
            if s["status"] not in statuses or (max_episodes and episodes >= max_episodes):
                continue
            states, actions = [], []
            for e in store.read(s["episode_id"]).events:
                if e["type"] == "observation":
                    states.append(e["payload"]["state"])
                elif e["type"] == "action":
                    actions.append(np.asarray(e["payload"]["data"], dtype=np.float64).reshape(-1))
            if not actions:
                continue
            if keys is None:
                keys = sorted(k for k, v in states[0].items() if 0 < np.asarray(v).size <= 64)
            if dims is None:
                dims = [int(np.asarray(states[0][k]).size) for k in keys]
            acts = np.stack(actions)
            pad = np.concatenate([acts, np.repeat(acts[-1:], chunk, axis=0)])
            for t in range(min(len(acts), len(states))):
                obs_rows.append(_vec(states[t], keys))
                chunk_rows.append(pad[t:t + chunk])
            episodes += 1
    if not obs_rows:
        raise ValueError(f"no demonstrations with status in {statuses} in the given runs")
    return Demos(obs=np.stack(obs_rows), chunks=np.stack(chunk_rows), keys=list(keys or []), dims=dims or [],
                 semantics=semantics, episodes=episodes)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(0), x.std(0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def save(out: str | Path, method: str, demos: Demos, arrays: dict[str, np.ndarray], hyper: dict[str, Any]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "policy.npz", **arrays)
    (out / "policy.json").write_text(json.dumps({
        "schema": SCHEMA, "method": method, "keys": demos.keys, "dims": demos.dims, "chunk": int(demos.chunks.shape[1]),
        "action_dim": int(demos.chunks.shape[2]), "semantics": demos.semantics, "hyper": hyper}, indent=2))


class ReferencePolicy:
    """Loads an artifact; subclasses implement `_sample(x, rngs, noise_scale) -> (N, T, A)` in action units."""

    method = "reference"

    def __init__(self, path: str, execution_steps: int | None = None) -> None:
        root = Path(path)
        self.meta = json.loads((root / "policy.json").read_text())
        if self.meta.get("schema") != SCHEMA or self.meta["method"] != self.method:
            raise ValueError(f"{path} is not a {self.method} policy ({self.meta.get('schema')}, {self.meta.get('method')})")
        with np.load(root / "policy.npz", allow_pickle=False) as z:
            self.a = {k: z[k] for k in z.files}
        self.keys, self.chunk, self.action_dim = self.meta["keys"], self.meta["chunk"], self.meta["action_dim"]
        self.obs_norm = Normalizer(self.a["obs_mean"], self.a["obs_std"])
        self.act_norm = Normalizer(self.a["act_mean"], self.a["act_std"])
        digest = json.loads((root / "artifact.json").read_text())["digest"] if (root / "artifact.json").exists() else "unsealed"
        self.name = f"{self.method}@{digest}"
        self.exec_steps = int(execution_steps or self.meta["hyper"].get("execution_steps") or max(1, self.chunk // 2))

    def spec(self) -> AgentSpec:
        sem = self.meta.get("semantics")
        return AgentSpec(name=f"policy.{self.method}", version="0.1", required_state=tuple(self.keys),
                         action_semantics=ActionSemantics.model_validate(sem) if sem else None,
                         action_dim=self.action_dim, components={"policy": self.name},
                         weights_license="self-trained from development runs")

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def default_execution_steps(self) -> int:
        return min(self.exec_steps, self.chunk)

    def sample(self, observation: Observation, seeds: list[int] | None, noise_scale: float = 1.0) -> np.ndarray:
        x = self.obs_norm(_vec(observation.state, self.keys))
        rngs = [np.random.default_rng(s) for s in seeds] if seeds else [np.random.default_rng()]
        return self._sample(x, rngs, noise_scale)

    def _sample(self, x: np.ndarray, rngs: list[np.random.Generator], noise_scale: float) -> np.ndarray:
        raise NotImplementedError
