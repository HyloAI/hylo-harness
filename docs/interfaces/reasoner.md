# Interface: Reasoner / Planner / Tools

> Package: `hylo-reasoners` (rule-based, `hylo.reasoners.rules`); contract in `hylo-core` **[Validated in prototype: synchronous, rule-based]**

## 1. Role split

| Role | Frequency | Input | Output | Examples |
|---|---|---|---|---|
| Reasoner | event-triggered (0.1–2 Hz) | observation, working memory, long-term memory, trigger reason | `Decision` | VLM/LLM, rules, LangGraph graph, DSPy program |
| Planner | called by the Reasoner, or is the Reasoner | goal, world state, skill library, World Model | subgoal/skill sequence | symbolic planning, MPC, LLM planning |
| Policy | every control step | observation + current subgoal/instruction | `ActionChunk` | VLA, diffusion policy, controller |

v0.1 treats the Planner as one implementation of a Reasoner (no separate protocol); once a planner appears that needs to be swapped independently (e.g. World Model-based MPC), v0.2 will split out a `Planner` protocol.

## 2. Protocol

```python
class Decision(Contract):
    kind: str                         # "subgoal" | "instruction" | "skill" | "abort" | "ask_human"
    output: Any                       # pose, language instruction, SkillCall(name, args) ...
    rationale: str                    # human-readable rationale (LLM thinking summary; rule name)
    trigger: str
    component: str                    # "name@version", written to the Trace
    inputs: Mapping[str, Any] = {}    # references (observation seq, retrieval hit ids), for diagnostics

@dataclass
class ReasonerContext:
    task_id: str; episode_id: str
    working: WorkingMemory
    memory: MemoryStore | None
    emit: Callable[[str, Mapping[str, Any]], None]
    tools: Toolkit | None = None      # v0.2

class Reasoner(Protocol):
    name: str
    def begin(self, rctx: ReasonerContext) -> None: ...                       # episode start: retrieve long-term memory
    def deliberate(self, obs: Observation, trigger: str, rctx: ReasonerContext) -> Decision: ...
    def end(self, success: bool | None, rctx: ReasonerContext) -> None: ...   # episode end: write memory
```

v0.2 async variant: `async def deliberate(...)`, run by `ComposableAgent` in a background task; the Policy uses the most recently committed Decision (see `execution-lifecycle.md` §4).

## 3. Tools (v0.2, port of RPent)

RPent's tool layer is currently the implementation best suited to "LLM invoking physical capabilities"; we plan to port it (Apache-2.0, with attribution):

```python
@tool(readonly=True)
def locate(object_name: str) -> Pose: ...          # perception/memory query: can run in parallel
@tool(readonly=False)
def run_policy(instruction: str, max_steps: int) -> SkillResult: ...   # physical action: serial FIFO
```

- Single source: the same decorator generates both the model-visible JSON schema and runtime validation.
- Scheduling: read-only tools run in parallel, write tools run serially; cancellation waits for a safe point.
- **Key difference from RPent**: every action produced by a tool still goes through the Kernel's SafetyShield; the LLM has no Bash/file-write permission; the tool allowlist is set by the experiment config and cannot be expanded by RSI candidates.
- Success is decided by the Task, not by the LLM's own `finish` claim.

## 4. Adapters

| Adapter | Description | Version |
|---|---|---|
| `testing.rule_landmark` | prototype reference implementation: visible → go to goal; otherwise recall → go to recalled position (if reached but not visible, judge the memory stale and forget it); otherwise explore | ✅ prototype |
| `llm.openai_compatible` | any OpenAI-compatible endpoint (including local vLLM); structured output → Decision; images via base64 | v0.1 P1 |
| `llm.anthropic` / other vendors | official SDKs | v0.2 |
| `langgraph.graph` | wraps a compiled StateGraph as a Reasoner; `thread_id = episode_id`; only references in state (no images); checkpointer used for planner-level replay | v0.2 |
| `dspy.program` | loads DSPy-optimized **JSON** program state (loading `.pkl` is forbidden) | v0.2 |

## 5. Observability requirements

Each `deliberate` produces at least one `agent.decision` event (when the output changes), containing: `kind, output, rationale, trigger, component`; LLM adapters additionally record `model, prompt_hash, tokens_in/out, latency_s`, and (if enabled) store the full prompt/response in a blob. This is the basis for RSI to diagnose "reasoning error vs execution error".
