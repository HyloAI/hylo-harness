# Interface: Agent / Policy / World Model

> Packages: `wc-core` (Agent protocol) + `wc-agents` **[Validated in prototype]**

## 1. Agent (the only intelligence abstraction the Kernel knows about)

```python
class AgentSpec(Contract):
    name: str; version: str
    required_state: tuple[str, ...]; required_cameras: tuple[str, ...]
    action_semantics: ActionSemantics | None
    action_dim: int | None
    components: Mapping[str, str]          # {"policy": "openpi.pi05@<ckpt>", "reasoner": "...", "memory": "..."} → written to the Trace
    weights_license: str | None            # required for learned policies

@dataclass
class EpisodeContext:
    episode_id: str; task: TaskInstance; env_spec: EnvSpec; seed: int
    emit: Callable[[str, Mapping[str, Any]], None]   # writes to the Trace (source="agent")

class Agent(Protocol):
    def spec(self) -> AgentSpec: ...
    def reset(self, ctx: EpisodeContext) -> None: ...
    def act(self, observation: Observation) -> ActionChunk: ...
    def on_step(self, result: StepResult) -> None: ...
    def on_episode_end(self, outcome: EpisodeOutcome) -> None: ...
```

**Constraints**
- The Agent never gets the SafetyShield, an Environment handle, or privileged state.
- After `reset`, no state from the previous episode may remain, unless it is an explicitly declared Memory (conformance test item).
- Bind-time compatibility check: a mismatch in control_mode / rotation_repr / dimension / required keys = error (not a warning) **[Validated in prototype]**.

## 2. ActionChunk and ChunkExecutor

```python
class ActionChunk(Contract):
    actions: np.ndarray            # (T, *action_shape)
    execution_steps: int | None    # replan after executing this many steps (LeRobot ChunkPolicySpec: prediction ≠ execution)
    provenance: Mapping[str, Any]  # model id, checkpoint, latency, reference to raw/unnormalized output
```

`ChunkExecutor` (inside the Kernel) handles queuing and replanning; swappable strategies in v0.2: `OpenLoop` (default), `TemporalEnsemble`, `RealTimeChunking`, `AsyncPrefetch`.
This unifies openpi's `ActionChunkBroker`, LeRobot's `select_action` queue, and vla-eval's ensemble in one place.

## 3. Policy (fast system)

```python
class PolicySpec(Contract):
    name: str; version: str
    input_features: Mapping[str, FeatureSpec]     # LeRobot FeatureSpec semantics (type, shape, names, semantics)
    action_semantics: ActionSemantics
    prediction_steps: int; execution_steps: int
    control_hz: float | None
    weights_license: str

class Policy(Protocol):
    def spec(self) -> PolicySpec: ...
    def reset(self, ctx: EpisodeContext) -> None: ...          # must also reset the pre/post processors (LeRobot lesson)
    def act(self, obs: Observation, instruction: str | None) -> ActionChunk: ...
    def invalidate(self) -> None: ...                           # instruction changed mid-episode: drop the action queue, keep memory (LeRobot drop_queued_actions)
```

| Implementation | Approach | v0.1 |
|---|---|---|
| `PControllerAgent`, `PlanarIKAgent`, `PGoalPolicy` | in-process baselines | ✅ prototype |
| `OpenPIRemotePolicy` | openpi WS wire protocol (msgpack-numpy, metadata in first frame); obs key mapping and image preprocessing configured on the client | P1 |
| `LeRobotPolicyAdapter` | `make_policy` + processors; `chunk_inference_spec()` → PolicySpec; policies without chunk support degrade to `StepwisePolicyAdapter` | P2 |
| `VlaEvalModelServerPolicy` | connects to vla-eval's 17 model servers | P2 |

## 4. ComposableAgent (reference composition)

```python
class ComposableAgent(Agent):
    def __init__(self, policy: GoalPolicy | Policy, reasoner: Reasoner, memory: MemoryStore | None,
                 triggers: ReasoningTriggers, world_model: WorldModel | None = None): ...
```

- The Reasoner is trigger-driven (`episode_start`, `perception_change`, `subgoal_reached`, `periodic`; v0.2 adds `failure_signal`, `low_confidence`) **[Validated in prototype]**.
- Emits `agent.decision` when the Decision changes; Memory retrieval emits `memory.retrieval` **[Validated in prototype]**.
- Prototype result (testing/reach-v0 validation, 100 episodes): composite Agent with memory success 1.00; best pure baseline 0.62 (cannot complete the briefly-visible and landmark tasks).

## 5. World Model (interface in v0.2, implementation in v0.3)

```python
class WorldModel(Protocol):
    def spec(self) -> WorldModelSpec: ...           # observation/action spaces, prediction horizon, whether latent
    def encode(self, obs: Observation) -> Latent: ...
    def predict(self, latent: Latent, actions: np.ndarray) -> Prediction: ...   # predicted observation/latent + uncertainty
    def reward(self, latent: Latent, task: TaskSpec) -> float | None: ...        # optional

class WorldModelEnvironment(Environment):           # wraps a World Model as an Environment
    """Runs episodes "in imagination" with the same Kernel, same Agent and same Scorer: used for planner candidate evaluation,
    cheap pre-screening of RSI candidates (cannot replace the validation gate), and sim-vs-model consistency studies."""
```

This is one of the differentiators between worldcurrent and existing projects: the world model is treated as a first-class evaluable object, not as a private implementation detail of the planner.
