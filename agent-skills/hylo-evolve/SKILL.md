---
name: hylo-evolve
description: Run a Hylo improvement campaign with `hylo evolve`, where a proposer suggests candidates, the gate tests them against the active version on fresh cases, and winners are promoted (or rolled back later). Use when asked to improve an agent with evidence, compare a candidate to the current version, or promote, roll back or shadow a version.
---

# Improvement campaigns

A campaign config (`schema_version: hylo.evolve/v0.1`) names an experiment, a proposer, the gate protocol and the
number of cycles. Each cycle:

1. The proposer suggests candidates: a parameter change (`rule.gain`), an LLM-written change (`llm.*`), or a
   program written in a research session (`research.llm`).
2. Each candidate is screened on `search_val`.
3. The best ones run paired against the active version on `gate`, judged by a sequential test (McNemar by default).
4. A winner is reported on `heldout` and promoted, automatically or after approval.

## Commands

```bash
uv run hylo evolve run bench/examples/evolve/mujoco_reach_gain.yaml   # run the cycles
uv run hylo evolve status runs/evolve/mujoco-reach-gain               # candidates, verdicts, the active version
uv run hylo evolve approve runs/evolve/mujoco-reach-gain                # approve a pending promotion (approval: manual)
uv run hylo evolve rollback runs/evolve/mujoco-reach-gain --reason "field regression"
uv run hylo evolve shadow runs/evolve/mujoco-reach-gain --candidate ID   # run beside the active version, no control
```

## Research programs through the gate

```yaml
proposer:
  type: research.llm
  params:
    model: {backend: ollama, model: "qwen3.5:35b-a3b", vision: true}
    dev: 3
    budget: {turns: 30, resets: 6, submits: 3}
protocol: {alpha: 0.05, stage1_pairs: 10, gate_top_k: 1, gate_looks: 1, gate_test: {type: mcnemar}}
```

## Reading the result

`status` lists each candidate with its screening score, its verdict (`rejected` at screening, `gated_fail` at
the gate, `gate_passed`, `pending` approval, `promoted`), the paired counts and the p-value. A `gated_fail` often
means the gate lacked the evidence to tell the candidate apart, not that it is worse: say which, and do not call it
a win either way. Promotion writes a sealed artifact; `hylo artifact verify` checks it.

## Rules

- Never change the gate or held-out cases to get a candidate through.
- Quote the paired counts with any claim of improvement.
- Campaigns can take long; run them in the background and check `status`.
