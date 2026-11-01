# Plugin System

> Implementation: wc-core `worldcurrent/core/registry.py` **[Validated in prototype]**

## 1. Design sources

| Source | Borrowed | Discarded |
|---|---|---|
| inspect-robots `registry.py` | `importlib.metadata` entry points; auto-loading can be disabled via an environment variable | — |
| vla-eval | `module:Class` references written directly in YAML, lazy import | No entry points, no capability declarations |
| RPent / XPolicyLab | — | Directory name as plugin (no version, no metadata) |

## 2. Plugin groups

| Group (entry point group) | Protocol | Examples |
|---|---|---|
| `worldcurrent.envs` | `Environment` | `testing.point_reach`, `mujoco.planar_reach`, `libero.remote` |
| `worldcurrent.agents` | `Agent` factory | `testing.p_controller`, `testing.landmark` |
| `worldcurrent.policies` | `Policy` | `openpi.remote`, `lerobot.local` |
| `worldcurrent.reasoners` | `Reasoner` | `testing.rule_landmark`, `llm.openai_compatible`, `langgraph.graph` |
| `worldcurrent.memory` | `MemoryStore` | `sqlite`, `sparkdsg`, `mem0` |
| `worldcurrent.scorers` | `Scorer` | `success`, `safety_interventions`, `vlm_judge` |
| `worldcurrent.benchmarks` | `Benchmark` factory | `testing/reach-v0`, `libero/libero_10` |
| `worldcurrent.generators` | `CandidateGenerator` | `rule.gain`, `dspy.gepa`, `llm.structured` |

## 3. Resolution rules

```python
registry.resolve(group, "testing.point_reach")              # 1. explicitly or lazily registered name
registry.resolve(group, "my_lab.envs.kitchen:KitchenEnv") # 2. if it contains ":", import as module:attr
# 3. on a name miss, load worldcurrent.<group> entry points (only once; WORLDCURRENT_NO_PLUGIN_AUTOLOAD=1 disables)
registry.build(group, {"type": "...", "params": {...}})   # factory call
```

- **Lazy**: `builtin.py` registers string references with `register_lazy`, so `import worldcurrent` does not import MuJoCo **[Validated in prototype]**.
- **Diagnosable errors**: an error for an unknown name lists the known plugins **[Validated in prototype]**.

## 4. Plugin metadata (introduced in v0.2)

```python
class PluginInfo(Contract):
    name: str; version: str; group: str
    isolation: Literal["in_process", "subprocess", "container"]
    requires: dict[str, str]          # {"python": ">=3.8,<3.9", "gpu": "optional", "vulkan": "required"}
    capabilities: dict[str, bool]     # corresponds to EnvCapabilities / PolicyCapabilities
    license: str                      # code license
    weights_license: str | None       # required for policies
    assets_license: str | None        # required for envs (ManiSkill: CC BY-NC 4.0)
```

The Runner checks `requires` and `capabilities` before building (for example, a benchmark requires `privileged_state`, or an experiment requires `snapshot` to do shadow replay); if they are not satisfied it refuses to run, rather than failing halfway through.

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
