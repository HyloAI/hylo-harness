# Module Dependency Graph

> Current package table and enforced rules: docs/design/architecture.md §2.1 and `dev/test_boundaries.py`
> (which also checks the repository rules of docs/design/repository-structure.md). The graph below is the
> original design (2026-10-30); package names follow the current ones.

## 1. Package dependencies (compile-time import relationships)

```mermaid
flowchart BT
  core[hylo-core<br/>numpy, pyyaml]
  eval[hylo-eval] --> core
  agents[hylo-agents] --> core
  memory[hylo-memory] --> core
  evolve[hylo-evolve] --> core
  evolve --> eval
  cli[Hylo CLI<br/>meta-package] --> core & eval & agents & memory & evolve

  subgraph adapters[Adapter packages - each declares its own heavy dependencies]
    envmj[hylo-env-mujoco<br/>+mujoco] --> core
    envremote[hylo-env-libero / robocasa / maniskill<br/>server side, own venv] --> core
    bridge[hylo-bridge-vlaeval<br/>+websockets, msgpack] --> core
    polopenpi[hylo-policy-openpi<br/>+websockets, msgpack] --> core
    pollerobot[hylo-policy-lerobot<br/>+lerobot, torch] --> agents
    rsllm[hylo-reasoner-llm<br/>+vendor SDK] --> agents
    rslg[hylo-reasoner-langgraph<br/>+langgraph] --> agents
    evdspy[hylo-evolve-dspy<br/>+dspy] --> evolve
    memdsg[hylo-memory-sparkdsg<br/>+spark-dsg] --> memory
    memmem0[hylo-memory-mem0<br/>+mem0] --> memory
  end
```

**Dependency rules** (enforced in CI with an AST import-lint, modeled on Zetta's `test_layering.py`):

1. `hylo-core` does not import any other Hylo package.
2. `hylo-eval`, `hylo-agents` and `hylo-memory` depend only on `hylo-core` and **do not depend on each other**.
   - Memory does not know about Agent; Agent holds a memory instance through the `MemoryStore` protocol (dependency injection).
   - Bench does not know Agent internals; Scorers only read the Trace.
3. `hylo-evolve` depends on `hylo-core` + `hylo-eval` (it needs to run evaluations); it interacts with memory/agents only through config patches and protocols.
4. No non-adapter package **may** import an adapter package; adapter packages do not depend on each other.
5. Heavy dependencies (torch, jax, mujoco, langchain-core, dspy, robosuite) appear only in adapter packages.

> Resolved in the workspace packages: memory contracts (`MemoryItem`, `MemoryQuery`, `MemoryStore`, `WorkingMemory`) live in `hylo.core.memory`; agents obtain a backend through the `memory` plugin group, so hylo-agents and hylo-memory never import each other. `dev/test_boundaries.py` enforces rules 1–5 on every `src/` tree.

## 2. Runtime process topology

```mermaid
flowchart LR
  subgraph main[Main process - Hylo venv]
    RUN[EvaluationRunner] --> K[Kernel]
    K --> AG[Agent<br/>Reasoner/Policy/Memory]
    K --> SH[SafetyShield]
    K --> TR[(Trace)]
  end
  subgraph envp[Env process - benchmark-specific venv/container]
    ES[EnvServer] --> SIM[LIBERO / RoboCasa / ManiSkill]
  end
  subgraph polp[Policy process - model-specific venv/container, GPU]
    PS[openpi / LeRobot / vla-eval model server]
  end
  subgraph llm[External API]
    LLM[LLM/VLM endpoint]
  end
  SH -- JSON lines / WS --> ES
  AG -- openpi WS msgpack --> PS
  AG -- HTTPS --> LLM
```

In **Served mode** the topology is inverted: the vla-eval benchmark container owns the loop, and our process acts as the model server:

```mermaid
flowchart LR
  BC[vla-eval benchmark container<br/>Orchestrator owns the loop] -- OBSERVATION --> ADP[hylo-bridge-vlaeval<br/>ModelServer adapter]
  ADP --> AG[Agent] --> SH[SafetyShield] --> ADP
  ADP -- ACTION --> BC
  ADP --> TR[(Trace)]
```

## 3. Summary of how external projects are depended on

| External project | Method | Target package |
|---|---|---|
| MuJoCo | PyPI dependency | hylo-env-mujoco |
| LIBERO / RoboCasa / ManiSkill | PyPI/source dependency in a separate venv | hylo-env-* (server side) |
| vla-evaluation-harness | Protocol-compatible + run its containers | hylo-bridge-vlaeval |
| openpi | Protocol-compatible (no import) | hylo-policy-openpi |
| LeRobot | Optional dependency + data format export | hylo-policy-lerobot, hylo-eval (exporter) |
| inspect-robots | Port code (MIT, with attribution) | hylo-core (safety/compat parts) |
| RPent | Port the tool layer (Apache-2.0, with attribution) | hylo-agents (tools) |
| Zetta | Borrow schema design, rewrite | hylo-evolve |
| LangGraph / DSPy | Optional dependency | hylo-reasoner-langgraph / hylo-evolve-dspy |
| Spark-DSG / mem0 / Graphiti | Optional dependency | hylo-memory-* |
