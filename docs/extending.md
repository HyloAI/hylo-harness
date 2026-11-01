# Extending worldcurrent

Everything that varies between methods is a plugin: a class that satisfies a small contract, registered under a
name through a Python entry point (or referenced as `module:attr`). A plugin package never modifies worldcurrent.
The examples below live in the plugins repository (worldcurrent-plugins: `packages/wc-plugin-examples`, module
`wc_examples`), written the way a third party would; run them from that repository (they use MuJoCo planar reach
from worldcurrent-bench).

## 1. Start a plugin

```bash
wc plugin new learner my-recap          # also: policy, agent, guard, gate_test, online_learner, datasource, judge, generator
uv pip install -e plugins/wc-plugin-my-recap
uv run pytest plugins/wc-plugin-my-recap/tests          # the template passes its conformance check as generated
wc plugin check learner my-recap                        # same check from the CLI, on any registered plugin
wc plugins                                               # every plugin: contract version, maturity, licence
```

Environments and agents are checked with `wc conformance CONFIG [--agent]`.

Each plugin group is a versioned contract (`wc.policy/v0.1`, ...; `worldcurrent.core.plugins.CONTRACTS`). A plugin
package declares what it implements in one dictionary, registered under the `worldcurrent.meta` entry-point group
(the template generates it):

```python
META = {"contracts": {"learners": "wc.learner/v0.1"}, "maturity": "unverified",
        "plugins": {"learners/my-recap": {"weights": "...", "notes": "..."}}}     # per-plugin overrides, optional
```

Maturity is `implemented` (run end to end on its target, with recorded runs), `unverified`, or `needs-hardware`;
`weights` names the licence of model weights the plugin needs, the code licence is `license` in pyproject.toml.
The registry refuses a plugin that declares another major version, or a newer minor version, than the core
implements; `wc plugin check` warns when a declaration is missing.

## 2. Extension points

| Group | Contract (where) | Built-in examples | What it extends | Conformance |
|---|---|---|---|---|
| `envs` | `Environment`: `spec / reset / step / close` (core/env.py) | libero, mujoco.*, lerobot.robot, world_model, remote.ws | simulators, robots | `wc conformance` |
| `policies` | `ChunkPolicy`: `spec / reset / default_execution_steps / sample(obs, seeds, noise_scale) -> (N, T, A)` (core/policy.py) | policy.knn, policy.bc, policy.flow, policy.tokens, policy.mpc (wc-policies), lerobot, examples.ik_reacher, examples.mixture | the model inside an agent: any policy family | `wc plugin check policy` |
| `agents` | `Agent`: `spec / reset / act / on_step / on_episode_end` (core/agent.py) | policy, temporal_ensemble, steered (each runs any `policies` plugin), lerobot.*, openpi, memory / noise / critic steering, shadow, planner.subgoal | decision logic, test-time methods | `wc plugin check agent`, `wc conformance --agent` |
| `learners` (offline) | `Learner.fit(job) -> sealed artifact` (core/learning.py) | critic.chunk_value, judge.trace, world_model.state, lerobot.* | learning from recorded data | `wc plugin check learner` |
| `learners` (online) | `OnlineLearner`: `start / policies / update / finished / seal` (core/learning.py) | examples.cem | learning by acting (policy search, on-policy RL, iterative self-training) | `wc plugin check online_learner` |
| `instruments` | `SuccessJudge.judge(trace)`, `ValueModel`, `WorldModel` (core/instrument.py) | judge.trace, judge.vlm | measuring and labelling | `wc plugin check judge`, then `wc calibrate` |
| `datasources` | `DataSource.episodes() -> ExternalEpisode` with `origin`, `licence` (core/learning.py, core/datasource.py) | lerobot.dataset, video.frames, video.files | demonstrations, videos, other robots | `wc plugin check datasource` |
| `generators` | `propose(diagnosis, cfg, parent_id, archive) -> [(Candidate, hypothesis)]` (evolve/protocol.py) | rule.*, llm.steering, experience, learn | how candidates are proposed | `wc plugin check generator` |
| `gate_tests` | `GateTest.compare(candidate, parent) -> PairedComparison`, `metric` (eval/stats.py) | mcnemar, sign, paired_permutation, examples.min_effect | the statistic the gate decides with | `wc plugin check gate_test` |
| `guards` | `review(action, ctx) -> GuardVerdict`, `name` (core/safety.py) | delta_limit, workspace, norm_limit, keep_out, examples.slow_zone | safety rules | `wc plugin check guard` |
| `memory` | `MemoryStore`, `ActionMemory` (core/memory.py, core/experience.py) | action_bank, skill_library, lesson_book, sqlite | experience stores | conformance kit `check_memory_store` |
| `scorers`, `benchmarks` | `Scorer` (core/scoring.py), a factory returning a `Benchmark` (core/benchmark.py) | success, steps, mujoco/reach-v0, libero | metrics, task sets | – |
| `reasoners`, `choosers`, `planners`, `reflectors` | `Reasoner`, `Chooser`, `Planner`, `Reflector` (core/reasoning.py) | testing.rule_landmark, chooser.llm, planner.llm, llm.reflector | deliberation, object choice, subgoal planning, lessons from campaigns | – |

## 3. What a plugin gets for free, and what it cannot change

For free:
- store ids: configs reference artifacts as `artifact:<name>@<digest>`, resolved to this machine's path when your
  plugin is built (`wc store`, worldcurrent.core.store); your plugin receives a path as before;
- the split guard: learners read development data only. Gate and held-out runs are refused at build time and
  again at the gate.
- sealing and provenance;
- running in any venv (`launch`);
- resumable jobs;
- the gate (cascade, sequential looks, Holm, regression and safety vetoes, held-out report);
- traces, replay, A/B comparison;
- the playground.

By design, a plugin cannot change:
- the protected configuration scopes (safety, benchmark, trace, scoring);
- the mandatory safety guards (finite check first, bounds clamp last);
- the rule that only the gate activates a behaviour artifact, and that instruments are trusted only after
  calibration.

Candidates are configuration patches. Model-generated code is not executed until an isolated runtime exists.

## 4. Worked examples (all runnable)

| Example | Files | Run |
|---|---|---|
| A guard plugin that slows the robot inside a zone | `wc_examples/guards.py`, `examples/configs/mujoco_guarded.yaml` | `wc run examples/configs/mujoco_guarded.yaml` |
| A gate on steps instead of success; a practical-significance gate | `wc_examples/stats.py`, `examples/evolve/mujoco_steps_gate.yaml` | `wc evolve run examples/evolve/mujoco_steps_gate.yaml` |
| An online learner (cross-entropy method) as a candidate generator | `wc_examples/online.py`, `examples/evolve/mujoco_online_cem.yaml` | `wc evolve run examples/evolve/mujoco_online_cem.yaml` |
| A policy written from scratch, and a mixture of registered policies, steered | `wc_examples/policies.py`, `examples/configs/mujoco_mixture.yaml` | `wc run examples/configs/mujoco_mixture.yaml` |
| One reference policy per family, trained and scored on MuJoCo reach (bench) | `examples/policies/` in worldcurrent-bench | `uv run python examples/policies/compare.py` |
| An offline learner, a judge, a data source, a generator, a policy, an agent | `wc plugin new <group> <name>` templates | `uv run pytest <package>/tests` |

## 5. Policies: one contract, many families

An agent is what the kernel runs; a policy (`policies` group) is the model inside it. Any policy plugin runs with
`{type: policy, params: {policy: SPEC}}`, ACT-style with `temporal_ensemble`, or steered (seeded noise,
sample-and-select, recovery rules) with `{type: steered, params: {sampler: SPEC}}`, and the reference policies of
`wc-policies` cover one example of each family (examples/policies/):

| Family | Published methods | Here |
|---|---|---|
| Retrieval | VINN | `policy.knn` |
| Regression BC with action chunks | BC, BC-RNN, ACT's chunking | `policy.bc`; ACT's temporal ensembling is the `temporal_ensemble` agent |
| Diffusion / flow matching | Diffusion Policy, pi0, SmolVLA, GR00T's action head | `policy.flow` (flow matching); the LeRobot ones via `lerobot` |
| Tokenized actions | RT-1, OpenVLA, VQ-BeT, pi0-FAST | `policy.tokens` (binned, independent heads) |
| Model-based planning | MPC, PETS, TD-MPC | `policy.mpc` over a `world_model.state` artifact |
| Large VLAs | pi0 / pi0.5, SmolVLA, GR00T N1.5, OpenVLA | `lerobot` (any LeRobot policy by repo id), `openpi` (an openpi server), or a new policy plugin |

## 6. Mapping published methods onto the extension points

| Method | Plugins |
|---|---|
| V-GPS, RoboMonkey (re-rank sampled actions) | a learner for the value model, plus an agent that samples and selects |
| Retrieve-then-Steer, DSRL (offline) | a learner that builds a bank, plus a prior / noise steering agent |
| Expert iteration, BC, RECAP (offline) | a data source and labeler if needed, plus a learner producing weights |
| SimpleVLA-RL / GRPO, DSRL (online), RECAP (iterative) | an online learner: `policies()` returns the current policy, `update()` takes a step from the rollouts' traces |
| GEPA, Eureka (prompt / reward search) | a generator: prompts and reward weights are configuration values |
| RoboArena-style pairwise evaluation | a gate test on a preference metric |
| FAIL-Detect / SAFE | an instrument for the detector; a guard to act on it |
| A new robot or simulator | an environment (or a LeRobot Robot, through `lerobot.robot`) |
| A new policy family (GR00T, OpenVLA, ...) | an agent adapter, or serving it behind the openpi protocol |
