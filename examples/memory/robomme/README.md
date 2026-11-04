# RoboMME Memory Evaluation

The experimental model-managed Working Memory implementation was removed on 2026-11-04.
Its runner, phase prompt, isolated panel launcher and dedicated prompt-validation API are
no longer available. Historical results are retained; the measured 64-attempt panel did
not demonstrate a memory benefit.

This directory retains the earlier heuristic diagnostic baseline and shared evaluation
utilities. SimpleMemVLA is an independent option with its own
[adapter README](../../../adapters/hylo-policy-simplememvla/README.md).

## Runtime

`run_paired.py` requires a separately installed SERA-compatible RoboMME environment,
a frozen grounded-subgoal pi0.5 websocket server and the local vision service.
`serve_vision.py` supports an existing Qwen model through Transformers or vLLM. It uses
loopback HTTP and accepts planning, grounding and region-selection requests. Unsupported
modes are rejected; the removed Working Memory protocol is not served.

From the harness repository root, with those services already available:

```bash
python examples/memory/robomme/run_paired.py \
  --sera-root /path/to/SERA --output /path/to/new-run \
  --split val --episode-indices 1 7 --conditions memory_off memory_on
python examples/memory/robomme/summarize.py /path/to/new-run \
  --baseline memory_off --candidate memory_on
```

Previously inspected episode IDs are development data for this workspace, not a fresh
validation panel. This is a diagnostic runner, not the canonical full benchmark protocol.

## Request Batching

The vLLM service retains `--batch-size 2 --batch-wait-ms 200` for concurrent HTTP requests.
`microbatch.py` keeps each request's inputs and response future separate, uses a bounded
queue and reports actual batch size and timing. It batches requests, not simulator
execution. A sequential runner alone does not fill concurrent batches.

## Conditions

`run_paired.py` compares `memory_off` and `memory_on` using the earlier event-memory
adapter. The removed `history`, `working` and `recent` runner conditions are no longer
exposed here. `summarize.py` still accepts arbitrary condition labels so old results can
be inspected without restoring the removed runtime.

## Boundaries

The retained runner is the earlier heuristic pilot, with optional RGB-region grounding
and execution checks. Its results must be interpreted according to that protocol; it is
not the retired model-managed memory design or a reproduction of SimpleMemVLA.
The shared vision service's source digest has changed. A new run requires a new source
manifest and output directory; old measurements do not validate the changed service.

## Artifacts

Historical experiment records remain in the docs workspace at
`docs/research/experiments/2026-11-03-working-memory.md` and its results JSON.
The 12 removed implementation and test files were archived with content hashes outside
the runtime packages at `hyloai/runs/retired-working-memory/2026-11-04/source.tar.gz` and
`manifest.json`. Existing experiment outputs and reference repositories were not removed.
