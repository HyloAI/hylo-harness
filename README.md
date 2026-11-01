# worldcurrent-harness

The base of worldcurrent: contracts and execution kernel, evaluation engine, agents, memory, learning, gated
improvement campaigns, the run catalog and the adapters to policy stacks and LLMs. Every other worldcurrent
repository (bench, serve, studio, plugins) builds on this one and talks to the others only through formats defined
in `wc-core`.

| Package | Namespace | Purpose |
|---|---|---|
| `packages/wc-core` | `worldcurrent.core`, `worldcurrent.testing` | contracts, kernel, SafetyShield, events and traces, plugin registry, experiment config, remote protocols and wire codec, artifacts and the split guard, cross-repository formats (active version, scorecard, workcell API), conformance kit, the `wc` dispatcher |
| `packages/wc-eval` | `worldcurrent.eval` | benchmark contract and splits, evaluation runner, scorers, paired statistics and gate tests, the test benchmark (testing/reach-v0) |
| `packages/wc-agents` | `worldcurrent.agents` | agents that run any policy (chunked, ACT temporal ensemble, steered), memory / critic / noise steering, skills, planner, success judges, world-model environment, shadow agent |
| `packages/wc-memory` | `worldcurrent.memory` | action and noise banks, skill library, lesson book, SQLite memory |
| `packages/wc-policies` | `worldcurrent.policies.reference` | reference policies, one per family (kNN, BC, flow matching, tokens, MPC), numpy only, trained from your runs |
| `packages/wc-learn` | `worldcurrent.learn` | learner jobs and sealed artifacts, judge calibration and labelling, external-data ingestion, online learning |
| `packages/wc-evolve` | `worldcurrent.evolve` | improvement protocol: diagnosis, candidates, cascade and group-sequential gate, registry with rollback, campaigns |
| `packages/wc-catalog` | `worldcurrent.catalog` | read-only queries over runs, episodes, comparisons, campaigns and artifacts |
| `packages/worldcurrent-cli` | `worldcurrent.cli` | `wc run`, `wc evolve`, `wc learn`, `wc calibrate`, `wc plugin`, ... |
| `adapters/wc-policy-lerobot` | `worldcurrent.policies.lerobot` | LeRobot policies (SmolVLA, ...) with steering, their learners and builders |
| `adapters/wc-policy-openpi` | `worldcurrent.policies.openpi` | client for openpi policy servers (pi0, pi0-FAST, pi0.5) |
| `adapters/wc-reasoner-llm` | `worldcurrent.reasoners.llm` | runtime LLM / VLM components (planner, VLM judge) |
| `adapters/wc-proposer-llm` | `worldcurrent.proposers.llm` | LLM candidate proposer and reflector for campaigns |
| `adapters/wc-data-video` | `worldcurrent.datasources.video` | videos as external data sources |

The harness has no environments of its own: runnable examples live where the simulators are, in worldcurrent-bench
(`examples/`: MuJoCo reach and table, LIBERO; campaigns, the reference-policy leaderboard, the RSI demo). Unit tests
use a test double (`worldcurrent.testing`, plugins named `testing.*`, hidden from `wc plugins` unless `--all`).

```bash
uv run wc plugins                                          # plugins: contract version, maturity, licence
uv run wc store ls                                         # runs and artifacts in this machine's store, by id
uv run wc plugin new policy my-policy                      # scaffold a plugin package (policy, agent, guard, ...)
uv run pytest                                              # unit tests on the test double
```

Documentation: [extending worldcurrent](docs/extending.md), [package layering](docs/dependency-graph.md),
[interfaces](docs/interfaces/), [plugin system](docs/plugin-system.md), [experiment schema](docs/experiment-schema.md).
Architecture, RFCs and research live in worldcurrent-docs.

## The worldcurrent repositories

| Repository | What it holds | Depends on |
|---|---|---|
| worldcurrent-harness (this one) | the base: contracts and kernel, evaluation engine, agents, memory, learning, improvement campaigns, run catalog and store, adapters, `wc` | - |
| worldcurrent-bench | benchmarking: protocols, scorecards, leaderboards; LIBERO and MuJoCo suites and their setup | harness |
| worldcurrent-serve | deployment: policy server, workcell, real-time runner, operator channel, robot adapters | harness (wc-core only) |
| worldcurrent-studio | the playground: control plane, run explorer, live sandbox, campaigns, React front end | harness |
| worldcurrent-plugins | example and community plugins | harness |
| worldcurrent-docs | architecture, RFCs, research notes, experiment records | - |
| deployments-template | template for a lab's private `<lab>-deployments`: workcell configs, calibration, safety limits | - |

Leaves never import each other; they cooperate through formats defined in `wc-core`.

## Development workspace

The repositories sit side by side in one directory (named `harness/`, `bench/`, `serve/`, `studio/`, `plugins/`,
`docs/`, `deployments-template/`); `dev/bootstrap.sh` clones the missing ones (`WC_GIT_BASE`), writes a uv
workspace over all of them (`dev/workspace.py`) and installs every package editable into one venv:

```bash
harness/dev/bootstrap.sh                 # in the directory holding the checkouts; Python 3.10+
uv run pytest                            # every present repository, plus the boundary lint (harness/dev)
uv run wc --help                         # subcommands contributed by the installed packages
uv run wc run bench/examples/configs/mujoco_table_memory.yaml    # language + spatial memory, MuJoCo table
uv run wc evolve run bench/examples/evolve/mujoco_reach_gain.yaml # a gated improvement campaign
```

A subset works too (the harness alone, or the harness and one leaf): the workspace covers whatever is present.
`dev/repos.toml` lists the repositories and their allowed dependencies; `dev/test_boundaries.py` checks package
layering and repository rules. Rules: core functionality before infrastructure; third-party code stays out of the
packages (see worldcurrent-docs); every package keeps its own tests and dependencies.
