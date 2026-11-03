# Plugin System

> Implementation: `hylo.core.registry` and `hylo.core.plugins` in hylo-core.

## 1. Design sources

| Source | Borrowed | Discarded |
|---|---|---|
| inspect-robots `registry.py` | `importlib.metadata` entry points; auto-loading can be disabled via an environment variable | — |
| vla-eval | `module:Class` references written directly in YAML, lazy import | No entry points, no capability declarations |
| RPent / XPolicyLab | — | Directory name as plugin (no version, no metadata) |

## 2. Plugin groups

Every contract is defined in hylo-core (module in brackets), so a plugin of any group depends on hylo-core alone.

| Group (entry point group) | Contract (hylo-core module) | Examples |
|---|---|---|
| `hylo.envs` | `Environment` (`core.env`) | `testing.point_reach`, `mujoco.planar_reach`, `libero` |
| `hylo.agents` | `Agent` or a factory returning one (`core.agent`) | `testing.p_controller`, `policy`, `steered`, `lerobot.steered` |
| `hylo.policies` | `ChunkPolicy` (`core.policy`) | `policy.flow`, `policy.bc`, `lerobot` |
| `hylo.reasoners` | `Reasoner` (`core.reasoning`) | `testing.rule_landmark` |
| `hylo.choosers` | `Chooser` (`core.reasoning`) | `chooser.rule_color`, `chooser.llm`, `chooser.vlm` |
| `hylo.planners` | `Planner` (`core.reasoning`) | `planner.identity`, `planner.llm` |
| `hylo.reflectors` | `Reflector` (`core.reasoning`) | `llm.reflector` |
| `hylo.memory` | `MemoryStore` / `ActionMemory` (`core.memory`, `core.experience`) | `sqlite`, `action_bank`, `lesson_book` |
| `hylo.scorers` | `Scorer` (`core.scoring`) | `success`, `steps`, `safety_interventions` |
| `hylo.gate_tests` | `GateTest` (`core.scoring`) | `mcnemar`, `sign`, `paired_permutation` |
| `hylo.benchmarks` | a factory returning a `Benchmark` (`core.benchmark`) | `testing/reach-v0`, `mujoco/reach-v0`, `libero` |
| `hylo.generators` | `Proposer` (`core.improvement`) | `rule.gain`, `rule.steering`, `llm.steering`, `learn` |
| `hylo.learners` | `Learner` / `OnlineLearner` (`core.learning`) | `critic.chunk_value`, `policy.flow` |
| `hylo.instruments` | `SuccessJudge` / `ValueModel` / `WorldModel` (`core.instrument`) | `judge.trace`, `judge.vlm` |
| `hylo.datasources` | `DataSource` (`core.learning`) | `video.frames`, `lerobot.dataset` |
| `hylo.guards` | `Guard` (`core.safety`) | `delta_limit`, `keep_out` |
| `hylo.surfaces` | a factory returning a tool `Surface` (`core.tooling`) | `mujoco.world` |

## 3. Resolution rules

```python
registry.resolve(group, "testing.point_reach")              # 1. explicitly or lazily registered name
registry.resolve(group, "my_lab.envs.kitchen:KitchenEnv") # 2. if it contains ":", import as module:attr
# 3. on a name miss, load Hylo.<group> entry points (only once; HYLO_NO_PLUGIN_AUTOLOAD=1 disables)
registry.build(group, {"type": "...", "params": {...}})   # factory call
```

- **Lazy**: `builtin.py` registers string references with `register_lazy`, so `import Hylo` does not import MuJoCo **[Validated in prototype]**.
- **Diagnosable errors**: an error for an unknown name lists the known plugins **[Validated in prototype]**.

## 4. Plugin metadata

A distribution declares metadata through a `hylo.meta` entry point pointing to a lightweight `META` dictionary.
The registry exposes each plugin as `PluginMeta` (`hylo.core.plugins`), including its contract version, maturity,
licence and weights information. For example, the declaration can contain:

```python
META = {
    "contracts": {"policies": "hylo.policy/v0.1"},
    "maturity": "unverified",
    "plugins": {"policies/my_policy": {"weights": "per checkpoint"}},
}
```

The registry refuses an incompatible declared contract version. Undeclared plugins can still load;
`hylo plugin check` warns about the missing declaration. `hylo plugins` lists installed metadata without loading
heavy plugin implementations. The code licence comes from the distribution's package metadata.

## 5. Isolation levels

| Level | When to use | Mechanism |
|---|---|---|
| `in_process` | Pure Python, dependencies compatible with the main environment | Direct import |
| `subprocess` | Dependency conflicts, but a separate venv can be created on the same machine | `RemoteEnvironment(ref, python=<other venv>)` **[Validated in prototype]** |
| `container` | System-level dependencies (EGL/Vulkan/CUDA/Isaac) or untrusted code | Same protocol, via docker exec / network endpoint |

The protocol layer **forbids pickle**; array encoding/decoding only allows bool/int/uint/float dtypes **[Validated in prototype]**.

## 6. Conformance Kit

Each plugin group provides a set of conformance tests (`conformance/`) that third-party adapters must pass:

- **Environment**: `spec()` does not trigger simulation construction; two resets with the same seed yield the same obs; `success` vs `terminated` semantics; simulation error → `truncated + info.sim_error`; if `snapshot` is declared, stepping after restore is consistent step by step.
- **Agent/Policy**: `spec()` matches the actual output dimensions; no cross-episode residual state after `reset` (except declared memory); `ActionChunk` shape.
- **MemoryStore**: deduplication, key invalidation, `as_of`, radius queries, `forget` soft delete.
- **Scorer**: depends only on the Trace (can run without env/agent).

The corresponding test cases in the prototype's `tests/` are the embryo of the conformance kit.

## 7. Tool surfaces

A surface (`hylo.surfaces`, contract `hylo.tools/v0.1` in `core.tooling`) is a set of tools a model or a person can
call: each `ToolDef` has a name, a kind (`read`, `act` or `end`), a structured description and a closed JSON Schema
for its arguments. `Surface.call` checks the arguments, runs the tool and returns either a result or a structured
`ToolError` (code, message, field, whether a retry can help). `Surface.digest()` identifies the declarations a run
used. Surfaces are served unchanged by `hylo world chat` (a chat model), `hylo mcp` (coding agents over MCP), the
Studio's World page and `hylo roles`.

