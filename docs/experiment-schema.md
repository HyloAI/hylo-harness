# Experiment configuration

Fields of `hylo.core.config.ExperimentConfig`, using schema `hylo.experiment/v0.1`.
For a complete workflow, see [Run experiments](../user-guide/experiments.md).
The [Configuration API](../api/config.md) documents the Python models;
[Data formats & schemas](../api/formats.md) provides the generated JSON Schema.

## Fields { #1-configuration-file }

| Field | Default | Meaning |
|---|---|---|
| `schema_version` | `hylo.experiment/v0.1` | experiment format version |
| `name` | required | experiment name |
| `benchmark` | required | `{type, split, params}`; `split` defaults to `validation`, `params` to `{}` |
| `env` | required | `{type, params, remote}`; `params` defaults to `{}`, `remote` to `null` |
| `agent` | required | `{type, params}`; `params` defaults to `{}` |
| `seed` | `0` | experiment seed recorded with the configuration |
| `safety` | `{}` | safety shield settings |
| `trace.record_privileged` | `true` | record available privileged state in the trace |
| `trace.image_every` | `1` | record camera images every N steps; `0` disables image recording |
| `timing` | default `Timing` model | defaults to `mode: sync`, `clock: virtual` |
| `output_dir` | `runs` | root directory for run outputs |
| `workers` | `1` | parallel episode workers; must be at least 1 |
| `tags` | `[]` | experiment labels |

Component `type` values name a registered plugin or a `module:attr` reference.
Unknown model fields are rejected. Plugin-specific fields belong in `params` and are validated by the consuming plugin.
See the [plugin catalogue](../developer-guide/plugin-catalogue.md) for registered implementations.

```yaml
schema_version: hylo.experiment/v0.1
name: mujoco-reach
benchmark:
  type: mujoco/reach-v0
  split: validation
  params: {protocol: standard}
env: {type: mujoco.planar_reach}
agent:
  type: baseline.planar_ik
  params: {kp: 0.3}
safety: {max_delta: 0.25}
trace: {record_privileged: true, image_every: 1}
output_dir: runs
```

### Component placement

`env.remote` accepts `runtime`, `python` and `env`. A named `runtime` resolves through the machine's
[execution profile](../configuration/profiles.md); explicit `python` and `env` values override that runtime.
A `remote` agent carries placement settings in its `params`.

### Timing

`timing` controls how policy time relates to environment time. See
[experiment timing](../user-guide/experiments.md#timing-sync-and-realtime) for execution semantics and
[`Timing`](../api/kernel.md#hylo.core.kernel.Timing) for every field and default.

## Digest and run manifest { #2-identifiers-and-reproducibility }

`config_digest` is the first 16 hexadecimal characters of SHA-256 over the canonical JSON produced by
`ExperimentConfig.measured()`. The measured configuration follows these rules:

- Exclude `output_dir` and `workers`.
- Normalize `env.remote` to `null`, so environment placement does not change the digest.
- Remove `runtime`, `python` and `env` from the `params` of components whose `type` is `remote`, recursively.
- Omit `timing` only when it equals the complete default `Timing` model; non-default timing is measured.
- Retain other configuration fields, including `name`, `seed` and `tags`.

### Reproducibility levels

A matching digest identifies the measured configuration. It does not establish identical code, model weights,
assets or results. Use [Validation](../developer-guide/validation/index.md) for measured reproducibility evidence.

### Run Manifest (`<run>/manifest.json`)

The run manifest records `run_id`, configuration and trace schema versions, the measured
configuration and digest, creation time, Git SHA, Python/platform information and selected package versions.
Store references add pinned artifact records. Callers may add further manifest fields.

## Seeds { #3-seeds }

The benchmark supplies each episode's task and seed. Split definitions belong to the benchmark; do not infer
held-out seeds from the experiment-level `seed`. See
[evaluation protocols](../benchmarking/protocols.md) for the fixed conditions used to compare agents.

## Overrides { #4-overrides-and-sweep }

`cfg.with_overrides({"agent.params.kp": 2.0})` applies dotted-path values and validates a new
`ExperimentConfig`. Its digest changes when the measured configuration changes.
Use [Run experiments](../user-guide/experiments.md) for command-line overrides.

## Output layout { #5-output-layout }

```text
runs/<run_id>/
├── manifest.json
├── evallog.json
├── episodes/<episode_id>.jsonl
└── blobs/
```

`manifest.json` records configuration and runtime provenance. `evallog.json` records the benchmark, agent,
aggregate results and episode samples. Episode traces reference array data in `blobs/`.
See [Runs and traces](../user-guide/experiments.md#runs-and-traces) for inspection and replay.
