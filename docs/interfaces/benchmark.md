# Interface: Benchmark / Task / Scorer

> Packages: `wc-core` (TaskSpec) + `wc-eval` **[Validated in prototype]**

## 1. Task

```python
InitSpec = SeedInit | InitStateRef(uri, count) | SnapshotInit(uri)

class Predicate(Contract):
    kind: Literal["builtin", "expr", "callable"]   # builtin name | declarative expression (BDDL goal) | "pkg.mod:Class.method"
    name: str = ""; expr: str | None = None; ref: str | None = None

class TaskSpec(Contract):
    id: str; version: str
    instruction: str                     # static text or template; the actual text is produced by env.reset
    init: InitSpec
    success: Predicate
    max_steps: int
    params: Mapping[str, Any]            # task parameters passed to the Environment
    tags: tuple[str, ...]
    def digest(self) -> str

class TaskInstance(Contract):
    spec: TaskSpec; seed: int; episode_index: int; split: str
    @property
    def key(self) -> str                 # "{id}#{seed}": primary key for paired comparison across versions
```

LIBERO mapping: `InitStateRef("libero://libero_10/<task>.pruned_init", 50)`, `Predicate("expr", expr=<BDDL :goal>)`.
RoboCasa mapping: `SeedInit` or `SnapshotInit`, `Predicate("callable", ref="robocasa...:<Task>._check_success")`, `instruction` is a template.

## 2. Benchmark

```python
class Benchmark(Contract):
    id: str; version: str
    tasks: tuple[TaskSpec, ...]
    splits: Mapping[str, tuple[int, ...]]     # train / validation / heldout → seeds; reusing seeds across splits is forbidden at construction
    requires_privileged: bool = False          # whether scoring depends on privileged information (written to EvalLog)
    metadata: Mapping[str, str]
    def instances(self, split: str) -> list[TaskInstance]
    def digest(self) -> str
```

v0.2 adds: the `TaskProvider` protocol (dynamic/procedurally generated tasks, e.g. RoboCasa layout sampling), `episodes_per_task`, per-task `max_steps` overrides, and upstream commit and asset hashes for external benchmarks.

## 3. Scorer / Grader

```python
class Score(Contract):
    value: float | None              # None = abstain (infra failure, agent error); excluded from the mean
    explanation: str = ""
    metadata: Mapping[str, Any] = {}

class Scorer(Protocol):
    name: str
    def score(self, trace: EpisodeTrace, task: TaskInstance) -> Score   # reads the Trace only

class Grader(Protocol):              # v0.2: captures external judgments (human, VLM, simulation oracle), written as grade events in the Trace
    name: str
    def grade(self, trace: EpisodeTrace, task: TaskInstance) -> GradeEvent
```

Built-in Scorers: `success`, `safety_interventions`, `steps` **[Validated in prototype]**. v0.2: `success_once`, `success_at_end` (ManiSkill semantics), `latency_p95`, `vlm_judge` (Grader).

**Scorer and Grader are separate** (from inspect-robots): a Grader may have side effects (calling a VLM, asking an operator) and writes its judgment as an event; a Scorer is always a pure function, so historical Traces can be re-scored at any time.

## 4. EvaluationRunner and EvalLog

```python
def evaluate(benchmark, split, *, env_factory, agent, run_root, safety, scorers, manifest) -> EvalResult
```

EvalLog (`wc.evallog/v0.1`):

```json
{
  "version": "wc.evallog/v0.1", "run_id": "...",
  "benchmark": {"id", "version", "digest", "split", "requires_privileged"},
  "agent": {"name", "version", "components"},
  "results": {"success_mean": 1.0, "safety_interventions_mean": 0.03, "steps_mean": 38.3, "n_episodes": 100, "n_scored": 100},
  "samples": [{"key": "testing/reach_fixed#10000", "task_id", "seed", "episode_id", "status", "scores": {...}, "trace": "episodes/<id>.jsonl"}]
}
```

v0.2: the top-level structure aligns with the inspect-robots EvalLog (`eval/results/stats/samples`) so that its `view` tool can be reused; sharding/resume adopts vla-eval's SQLite work queue pattern.

## 5. Statistical comparison

```python
def paired_compare(candidate: Mapping[key, float|None], parent: Mapping[key, float|None]) -> PairedComparison
#   drops pairs where either side abstained; returns n_pairs, candidate_wins, parent_wins, one-sided exact McNemar p
```

One-sided exact McNemar (the same test as Zetta `gating.py`) **[Validated in prototype: 5:0 → p=1/32]**.
