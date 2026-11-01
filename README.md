# hylo-harness

Part of [Hylo](https://hyloai.github.io/), the stack for physical AI. This repository holds the **kernel** and the
shared layers built on it: **cognition**, the **evaluation** engine and **evolution**.

The base of Hylo: contracts and execution kernel, evaluation engine, agents, memory, learning, gated
improvement campaigns, the run catalog and the adapters to policy stacks and LLMs. Every other Hylo
repository (bench, serve, studio, plugins) builds on this one and talks to the others only through formats defined
in `hylo-core`.

| Package | Namespace | Purpose |
|---|---|---|
| `packages/hylo-core` | `hylo.core`, `hylo.testing` | contracts, kernel, SafetyShield, events and traces, plugin registry, experiment config, remote protocols and wire codec, artifacts and the split guard, cross-repository formats (active version, scorecard, workcell API), conformance kit, the `hylo` dispatcher |
| `packages/hylo-eval` | `hylo.eval` | benchmark contract and splits, evaluation runner, scorers, paired statistics and gate tests, the test benchmark (testing/reach-v0) |
| `packages/hylo-agents` | `hylo.agents` | agents that run any policy (chunked, ACT temporal ensemble, steered), memory / critic / noise steering, skills, planner, success judges, world-model environment, shadow agent |
| `packages/hylo-memory` | `hylo.memory` | action and noise banks, skill library, lesson book, SQLite memory |
| `packages/hylo-policies` | `hylo.policies.reference` | reference policies, one per family (kNN, BC, flow matching, tokens, MPC), numpy only, trained from your runs |
| `packages/hylo-learn` | `hylo.learn` | learner jobs and sealed artifacts, judge calibration and labelling, external-data ingestion, online learning |
| `packages/hylo-evolve` | `hylo.evolve` | improvement protocol: diagnosis, candidates, cascade and group-sequential gate, registry with rollback, campaigns |
| `packages/hylo-catalog` | `hylo.catalog` | read-only queries over runs, episodes, comparisons, campaigns and artifacts |
| `packages/hylo-cli` | `hylo.cli` | `hylo run`, `hylo evolve`, `hylo learn`, `hylo calibrate`, `hylo plugin`, ... |
| `adapters/hylo-policy-lerobot` | `hylo.policies.lerobot` | LeRobot policies (SmolVLA, ...) with steering, their learners and builders |
| `adapters/hylo-policy-openpi` | `hylo.policies.openpi` | client for openpi policy servers (pi0, pi0-FAST, pi0.5) |
| `adapters/hylo-reasoner-llm` | `hylo.reasoners.llm` | runtime LLM / VLM components (planner, VLM judge) |
| `adapters/hylo-proposer-llm` | `hylo.proposers.llm` | LLM candidate proposer and reflector for campaigns |
| `adapters/hylo-data-video` | `hylo.datasources.video` | videos as external data sources |

The harness has no environments of its own: runnable examples live where the simulators are, in hylo-bench
(`examples/`: MuJoCo reach and table, LIBERO; campaigns, the reference-policy leaderboard, the RSI demo). Unit tests
use a test double (`hylo.testing`, plugins named `testing.*`, hidden from `hylo plugins` unless `--all`).

```bash
uv run hylo plugins                                          # plugins: contract version, maturity, licence
uv run hylo store ls                                         # runs and artifacts in this machine's store, by id
uv run hylo plugin new policy my-policy                      # scaffold a plugin package (policy, agent, guard, ...)
uv run pytest                                              # unit tests on the test double
```

Documentation: [extending Hylo](docs/extending.md), [package layering](docs/dependency-graph.md),
[interfaces](docs/interfaces/), [plugin system](docs/plugin-system.md), [experiment schema](docs/experiment-schema.md).
Architecture, RFCs and research live in hylo-docs.

## The Hylo repositories

| Repository | What it holds | Depends on |
|---|---|---|
| hylo-harness (this one) | the base: contracts and kernel, evaluation engine, agents, memory, learning, improvement campaigns, run catalog and store, adapters, `hylo` | - |
| hylo-bench | benchmarking: protocols, scorecards, leaderboards; LIBERO and MuJoCo suites and their setup | harness |
| hylo-serve | deployment: policy server, workcell, real-time runner, operator channel, robot adapters | harness (hylo-core only) |
| hylo-studio | the playground: control plane, run explorer, live sandbox, campaigns, React front end | harness |
| hylo-plugins | example and community plugins | harness |
| hylo-docs | architecture, RFCs, research notes, experiment records | - |
| deployments-template | template for a lab's private `<lab>-deployments`: workcell configs, calibration, safety limits | - |

Leaves never import each other; they cooperate through formats defined in `hylo-core`.

## Development workspace

The repositories sit side by side in one directory (named `harness/`, `bench/`, `serve/`, `studio/`, `plugins/`,
`docs/`, `deployments-template/`); `dev/bootstrap.sh` clones the missing ones (`HYLO_GIT_BASE`), writes a uv
workspace over all of them (`dev/workspace.py`) and installs every package editable into one venv:

```bash
mkdir hylo && cd hylo                    # Python 3.10+, uv
git clone https://github.com/HyloAI/hylo-harness.git harness
HYLO_GIT_BASE=https://github.com/HyloAI harness/dev/bootstrap.sh   # clones the other repositories, one venv
uv run pytest                            # every present repository, plus the boundary lint (harness/dev)
uv run hylo --help                         # subcommands contributed by the installed packages
uv run hylo run bench/examples/configs/mujoco_table_memory.yaml    # language + spatial memory, MuJoCo table
uv run hylo evolve run bench/examples/evolve/mujoco_reach_gain.yaml # a gated improvement campaign
```

A subset works too (the harness alone, or the harness and one leaf): the workspace covers whatever is present.
`dev/repos.toml` lists the repositories and their allowed dependencies; `dev/test_boundaries.py` checks package
layering and repository rules. Rules: core functionality before infrastructure; third-party code stays out of the
packages (see hylo-docs); every package keeps its own tests and dependencies.

Documentation (hylo-docs, `uv run mkdocs serve -f docs/mkdocs.yml` from the workspace): [home](../docs/docs/index.md), [installation](../docs/docs/getting-started/installation.md), [concepts](../docs/docs/getting-started/concepts.md), [design overview](../docs/docs/design/overview.md).

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Copyright 2026 Hylo AI.
