# Experiment Schema (`wc.experiment/v0.1`)

> Implementation: wc-core `worldcurrent/core/config.py`; example configs in worldcurrent-bench `examples/configs/` **[Validated in prototype]**

## 1. Configuration file

```yaml
schema_version: wc.experiment/v0.1      # required; readers match strictly and reject if incompatible
name: mujoco-reach
seed: 0                                  # experiment-level seed (Agent randomness, etc.); episode seeds are determined by the benchmark split
tags: [baseline, reach]

benchmark:
  type: mujoco/reach-v0                  # plugin name or module:attr
  split: validation                      # train | validation | heldout; search | search_val | gate | heldout with protocol: rsi
  params: {protocol: standard}

env:
  type: mujoco.planar_reach
  params: {}
  remote:                                # optional: run out of process
    python: /opt/venvs/libero/bin/python

agent:
  type: baseline.planar_ik
  params: {kp: 0.3}

safety:                                  # can only be tightened, mandatory guards cannot be disabled; RSI candidates cannot modify this
  max_delta: 0.25

trace:
  record_privileged: true

output_dir: runs                         # not included in the digest
```

## 2. Identifiers and reproducibility

| Identifier | How it is computed | Purpose |
|---|---|---|
| `config_digest` | `sha256(canonical_json(config except output_dir))[:16]` | Unique identifier of an experiment; two runs with the same digest ⇒ the same object is measured **[Validated in prototype]** |
| `benchmark.digest` | `sha256(id, version, each TaskSpec.digest, splits)` | Changes whenever the task set or seed table changes |
| `TaskSpec.digest` | `sha256(all TaskSpec fields)` | Written into every `episode.start` |
| `TaskInstance.key` | `"{task_id}#{seed}"` | Primary key for paired comparison |
| `run_id` | `{name}-{label}-{config_digest}-{rand}` | One execution |

### Run Manifest (`<run>/manifest.json`)

```json
{
  "run_id": "...", "schema_version": "wc.experiment/v0.1", "config_digest": "5ed561fa3971c221",
  "config": {"...": "full normalized configuration"},
  "code": {"git_sha": "..."},
  "runtime": {"python": "3.12.3", "platform": "...", "packages": {"numpy": "2.5.4", "mujoco": "3.17.0"}},
  "benchmark": {"id": "testing/reach-v0", "version": "0.1", "digest": "...", "split": "validation"},
  "agent_spec": {"name": "...", "version": "...", "components": {"policy": "...", "reasoner": "...", "memory": "..."}}
}
```

Added in v0.2: all worldcurrent package versions and protocol versions, external benchmark commit and asset hashes, policy weight identifiers and `weights_license`, memory `snapshot()` hash, Candidate Registry `active` version, EnvServer-side Python/dependency versions.

### Reproducibility levels

| Level | Guarantee | Conditions |
|---|---|---|
| R0 | Configuration traceable | Manifest exists |
| R1 | Statistically reproducible (same distribution) | Same config_digest + same benchmark digest |
| R2 | Step-wise deterministic | R1 + deterministic env + deterministic agent (MuJoCo CPU single-threaded, the test double) **[Validated in prototype: two episodes with the same seed produce bit-identical actions]** |
| R3 | Bit-identical across machines | R2 + locked dependencies and platform — promised only for the test double |

GPU simulation, GPU inference and remote LLMs can generally only reach R1; the EvalLog records the level achieved.

## 3. Seeds

- Episode seeds = the benchmark split's seed table; **the same seed must not appear in two splits** (validated at construction).
- Prototype split partitioning: train `0..`, validation `10000..`, heldout `20000..`.
- Agent-internal randomness is derived from `EpisodeContext.seed` (v0.2: `derive_seed(experiment.seed, task_key, component)`, modeled on inspect-robots).

## 4. Overrides and Sweep

- Dot-path override: `cfg.with_overrides({"agent.params.kp": 2.0})` returns a new config (new digest) **[Validated in prototype]**. An RSI candidate is simply a set of overrides.
- Sweep (v0.2): `sweep: {agent.params.kp: [0.5, 1, 2]}` expands into multiple configs, each with its own digest.

## 5. Output layout

```text
runs/<run_id>/
├── manifest.json
├── evallog.json           # wc.evallog/v0.1: benchmark, agent, results, samples[{key, task_id, seed, episode_id, status, scores, trace}]
├── episodes/<episode_id>.jsonl   # wc.trace/v0.1 event stream
└── blobs/<sha256>.npy     # arrays > 64 KB (images, etc.), content-addressed
```

Exporters (v0.2): `wc trace export --format lerobot-v3` (training data), `--format rerun` (visualization), `--format inspect-robots` (reuse its view tools).
