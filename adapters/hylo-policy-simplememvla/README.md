# SimpleMemVLA

An optional Hylo memory and agent implementation of the released SimpleMemVLA method.
Select this agent when using its compatible frozen video-conditioned checkpoint. Other
Hylo memory and agent choices remain independent. No training is performed.

See the [implementation README](../../packages/hylo-memory/docs/simplememvla/README.md)
for the source map, data-flow and history-window diagrams, sampling algorithm, memory API,
episode lifecycle, batching behavior and verification boundaries.

## Components

- `hylo.memory.simplememvla.SimpleMemVLAHistory`: Hylo-owned episode-local RGB buffers,
  frame sampling, padding, temporal patch identities and feature-cache storage.
- `hylo.policies.simplememvla.bridge`: connects those buffers to upstream preprocessing
  and batched inference. Neural model, subtask decoding and action-head code remain upstream.
- `SimpleMemVLARuntime`: one frozen model, with independent memory sessions and a shared
  `infer_batch` interface. Explicit seeds isolate request RNG state; `seed=None` retains
  native batch RNG semantics.
- `SimpleMemVLAAgent`: the native Hylo agent lifecycle, selectable as `simplememvla`.

The default RoboMME configuration stores a 60-second front-camera history sampled at
2 fps, uses 120 sampled frames and two-frame temporal patches, and supplies the current
wrist image. At 20 native fps this retains at most 1,191 front frames and one wrist frame.
Short fixed windows repeat the first observed frame; variable windows use the upstream
even-frame rounding rule when enabled by a checkpoint. Sampling uses native frame indices,
not wall time. The inference processor supplies the upstream video timestamps.

All raw frames are owned copies; invalid camera ticks are rejected before any camera is
appended. Episode reset clears raw frames, prompt state and visual feature caches. The
upstream inference path evicts cached patch features outside the current sampled window.
No semantic facts, handwritten task counters or cross-episode experience rules are added.

## Selection

Install this optional workspace adapter in the runtime that hosts the policy. The base
package imports neither Torch nor the upstream model until a runtime is created. Supply
the upstream source on `PYTHONPATH` in that inference environment and a local checkpoint;
the adapter does not download weights or install training dependencies.

```yaml
agent:
  type: simplememvla
  params:
    checkpoint: /models/simplememvla_robomme
    camera_map: {front: front, wrist: wrist}
    joint_key: joint_pos
    gripper_key: gripper_pos
    execution_steps: 16
```

The released RoboMME agent expects seven arm joint positions and one or two finger
positions, using the first finger as the eighth state value. Output is the native
eight-dimensional Panda joint-position/gripper action layout. The environment must match
that controller's units and gripper convention; the adapter does not translate embodiments.
Action dimension, control mode and 20 Hz control rate are checked at reset.

`on_step` records every executed control-step observation, including steps inside an action
chunk. Calling `act` on that same step does not append its frame again. An environment with
a demonstration prefix should call `agent.observe_history(prefix_observations)` after reset,
excluding the current observation subsequently passed to `act`. Observation state, images
and instruction are explicitly selected; task IDs, reward, success, hidden state and `extra`
are not forwarded to the model. The native agent seeds each decision deterministically from
the episode seed and decision index, so its RNG protocol differs from upstream unseeded
single-process batching.

The memory implementation can also be used directly without a model:

```python
from hylo.memory.simplememvla import SimpleMemVLAHistory

memory = SimpleMemVLAHistory()
memory.reset("episode-1")
memory.observe({"front": front_rgb, "wrist": wrist_rgb})
video = memory.video("front")
patch_ids = memory.patch_keys()
```

It is a streaming visual-memory API, not an implementation of the semantic `MemoryStore`
query/write protocol. It is therefore not registered as a semantic-store plugin.

## Reproduction

Upstream pin: `OpenBMB/SimpleMemVLA@453d6ec293e63318bbc2bce545d0b9256ab1bc51`.
Checkpoint pin: `yinchenghust/simplememvla_robomme@6bf7204311496b93f12889cad0eaf3fcbddc35e4`.
Tested reference inference uses Torch 2.9.1, Transformers 5.11.0, bf16, SDPA, FLA 0.2.2
and causal-conv1d 1.6.2.post1. Upstream source is a separate runtime dependency; original
reference checkouts are not modified. MIT attribution is included in the package notices.

CPU tests cover window/padding semantics, copy ownership, atomic input validation, reset,
camera/state boundaries and the agent lifecycle. The opt-in `tests/verify_upstream.py`
checks both fixed and variable histories against upstream, then compares prepared tensors,
subtasks and seeded action chunks at padding, long-history, rollover and prompt-change
boundaries. It requires the pinned inference environment and GPU; it is not run by default
unit tests.

The earlier 40/40 reference pilot measured upstream SimpleMemVLA. Hylo implementation
parity and regression results are recorded separately under
`runs/memory-20261104/hylo-simplememvla-v1/`. These checks establish reproduction of this
option, not a gain over other memory schemes or a full benchmark reproduction.

On 2026-11-04, 18 upstream sampling comparisons passed. GPU comparisons at five history
boundaries (batch 2) produced identical prepared inputs, subtasks and bitwise-identical
actions. The Hylo memory bridge then passed 8/8 closed-loop regression episodes across
PickXtimes, VideoUnmask, PickHighlight and PatternLock (previously observed test IDs 0 and
1, initial batch 8). The native Hylo agent lifecycle has CPU coverage; the GPU closed-loop
test uses the native RoboMME evaluator with Hylo memory sessions, so a full native Hylo
kernel run remains unverified. See the experiment record
`docs/docs/research/experiments/2026-11-04-simplememvla-integration.md` in the workspace.
