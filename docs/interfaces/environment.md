# Interface: Environment

> Packages: `hylo-core` (protocol) + `hylo-env-*` (implementations) **[Validated in prototype]**

## 1. Protocol

```python
class Environment(Protocol):
    def spec(self) -> EnvSpec: ...                       # must be cheap and side-effect free
    def reset(self, task: TaskInstance) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def close(self) -> None: ...

# Optional capabilities: declared in EnvSpec.capabilities, with the corresponding protocol implemented
class Snapshottable(Protocol):
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...

class PrivilegedStateSource(Protocol):
    def privileged_state(self) -> dict[str, Any]: ...   # goes into the Trace only; the Kernel never passes it to the Agent

class Renderable(Protocol):                              # v0.2
    def render(self, camera: str, *, width: int, height: int) -> np.ndarray: ...

class BatchedEnvironment(Protocol):                      # v0.3: ManiSkill GPU / Isaac Lab / mujoco.batch
    def reset_batch(self, tasks: Sequence[TaskInstance], env_ids: np.ndarray | None = None) -> BatchedObservation: ...
    def step_batch(self, actions: np.ndarray) -> BatchedStepResult: ...
```

## 2. Data types

```python
class EnvSpec(Contract):
    name: str; version: str
    observation: ObservationSpec        # state: {key: shape}, cameras: {name: (H,W,C)}, units, has_instruction
    action: BoxSpace                    # shape, low, high, semantics: ActionSemantics
    control_hz: float
    capabilities: EnvCapabilities       # seedable, snapshot, render, privileged_state, batch_size, self_paced, real_robot

class ActionSemantics(Contract):
    control_mode: Literal["joint_pos","joint_delta","joint_vel","eef_delta_pos","eef_delta_pose","eef_abs_pose","planar_vel"]
    frame: str = "base"
    rotation_repr: str | None = None    # axis_angle | quat_xyzw | rot6d
    gripper: str | None = None          # binary_close_positive | continuous_open01 ... (vocabulary follows vla-eval DimSpec.format)
    dim_labels: tuple[str, ...] = ()
    max_step: tuple[float | None, ...] | None = None

class Observation(Contract):
    state: Mapping[str, np.ndarray]     # canonical keys and units (aligned with LeRobot / inspect-robots CANONICAL_STATE_UNITS)
    images: Mapping[str, np.ndarray]    # HWC uint8, orientation already corrected (the LIBERO/RoboCasa [::-1] is done inside the adapter)
    instruction: str | None             # the instruction actually produced by the Environment (RoboCasa generates it dynamically per episode)
    time: float
    extra: Mapping[str, Any]

class StepResult(Contract):
    observation: Observation
    reward: float | None
    terminated: bool                    # episode ended naturally (success or unrecoverable failure)
    truncated: bool                     # external truncation (time limit, simulation error)
    success: bool | None                # Task predicate result, independent of terminated
    info: Mapping[str, Any]             # conventional keys: distance, sim_error, fail_reason
```

## 3. Semantic constraints (conformance test items)

1. `spec()` does not build the simulation (ManiSkill doing a real reset in its constructor to derive the obs space is a counter-example).
2. Two `reset` calls on the same `TaskInstance` (same seed) produce the same initial observation (when `seedable=True`).
3. `success` is decided by the Task; `terminated`/`truncated` only describe the lifecycle. The adapter is responsible for mapping upstream semantics:

| Upstream | Upstream semantics | Mapping |
|---|---|---|
| LIBERO | `done = success` | `success=done, terminated=done, truncated=steps>=max` |
| RoboCasa | `_check_success()`, never terminates | `success=_check_success(), terminated=success (configurable), truncated=steps>=horizon` |
| ManiSkill | `terminated = success \| fail` | `success=info.success, terminated=success or fail, info.fail_reason` |
| MuJoCo (our own Task) | none | we implement it; divergence → `truncated=True, info.sim_error` |

4. Numerical simulation errors must be surfaced as `truncated + info.sim_error`, never silently (MuJoCo silently resets `MjData` on divergence). The prototype handles BADQPOS/BADQVEL/BADQACC **[Validated in prototype]**.
5. When `snapshot=True` is declared, the subsequent trajectory after `restore(snapshot())` is bit-identical **[Validated in prototype: MuJoCo]**.
6. Action arrays are interpreted according to `BoxSpace.semantics.dim_labels`; composite actions (RoboCasa 12-D) must be mapped by named components; relying on flat ordering is forbidden.

## 4. Remote environments (EnvServer protocol `hylo.envserver/v0.1`)

- Transport: JSON lines over subprocess stdin/stdout (v0.1); v0.2 adds a WebSocket transport to support containers.
- Handshake: the server first sends `{"hello": "hylo.envserver/v0.1"}`; a version mismatch raises `EnvironmentFault`.
- Request: `{"id", "method": spec|reset|step|privileged_state|close, "args"}`; response: `{"id", "ok", "result"|"error", "traceback"}`.
- Encoding: contracts are pydantic models, so the wire format is `model_dump(mode="json")` / `model_validate`; ndarray → `{"__nd__", dtype, shape, b64}`, only b/i/u/f dtypes allowed; **no pickle**.
- Remote and in-process execution of the same episode produce step-by-step identical results **[Validated in prototype]**.

## 5. Adapter inventory

| Adapter | Runs in | v0.1 | Notes |
|---|---|---|---|
| `testing.point_reach` | in-process | ✅ prototype | 5 task types: reach/push perturbation/briefly visible/landmark |
| `mujoco.planar_reach` | in-process | ✅ prototype | own MJCF + Task; snapshot; divergence detection |
| `libero.*` | EnvServer (py3.8 venv) | P1 | BDDL → `Predicate(kind="expr")`; init states → `InitStateRef` |
| `vlaeval.*` | Served mode | P1 | see execution-lifecycle §3 |
| `robocasa.*`, `maniskill.*` | EnvServer | P2 | |
| `isaaclab.*` | container + Batched | P3 | |
