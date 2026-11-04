# Interface: Memory

> Package: `hylo-memory` (data types pushed down into `hylo-core`) **[Validated in prototype]**
> Based on: `../../research/comparisons/memory-comparison.md`, `../../research/projects/memory-landscape.md` §10

## 1. Data model

```python
class MemoryKind(str, Enum): WORKING, EPISODIC, SEMANTIC, SPATIAL, SKILL

class Pose(Contract):      frame_id: str; position: tuple[float, float, float]; orientation: tuple[float, ...] | None
class TimeSpan(Contract):  start: float; end: float | None; clock: Literal["sim", "wall"]

class MemoryItem(Contract):
    kind: MemoryKind
    content: str                         # human/LLM readable
    key: str | None                      # identity of the thing described; a new write with the same (kind, key) invalidates the old item
    payload: Mapping[str, Any]           # kind-specific structure (see §3)
    entities: tuple[str, ...]
    scope: Mapping[str, str]             # task_id / robot_id / episode_id
    pose: Pose | None                    # first-class field, not embedded
    event_time: TimeSpan | None          # when it happened in the world
    valid_from: float | None             # when the fact became true (the invalidation time valid_to is maintained by the store)
    confidence: float
    source: str                          # producer "component@version"
    id: str
    # v0.2: embeddings: {space@version: vector}, extent, links[(relation, target_id)], provenance(evidence uris)
```

## 2. Protocol

```python
class MemoryQuery(Contract):
    text: str | None; kinds: frozenset[MemoryKind] | None; key: str | None
    entities: tuple[str, ...]; scope: Mapping[str, str] | None
    time_range: tuple[float, float] | None      # intersected with the event_time interval
    as_of: float | None                         # items "believed true at that time"
    near: tuple[Pose, float] | None             # radius query; frame_id must match
    include_invalidated: bool = False; k: int = 10
    # v0.2: embedding=(space, vector), object_ids, relation=("on", "O12"), rerank

class Hit(Contract): item: MemoryItem; score: float; details: Mapping[str, float]; valid_to: float | None

class MemoryStore(Protocol):
    def write(self, items: Sequence[MemoryItem]) -> WriteResult: ...     # events: ADD | INVALIDATE | SKIP_DUP (v0.2: MERGE)
    def query(self, q: MemoryQuery) -> list[Hit]: ...
    def forget(self, q: MemoryQuery) -> int: ...                         # soft delete (expired_at), auditable
    def snapshot(self) -> str: ...                                       # content hash → Run Manifest
    # v0.2: consolidate(scope, budget) -> ConsolidationReport; history(item_id)
```

**Semantics validated in prototype**: radius queries use true distance (bbox pre-filter + exact check); mismatched frames never match; a new fact with the same key invalidates the old fact, but history remains queryable with `as_of`; duplicate writes are skipped; `forget` is a soft delete; WorkingMemory has a capacity limit.

## 3. Four kinds of memory

`hylo.memory.simplememvla.SimpleMemVLAHistory` is an additional, independent streaming
visual-memory option. It implements the SimpleMemVLA history window and temporal patch
identities; it does not implement the semantic-store protocol above. The optional
`hylo-policy-simplememvla` adapter connects it to a compatible frozen checkpoint and exposes
the `simplememvla` agent plugin. See `adapters/hylo-policy-simplememvla/README.md`
for its observation interface, upstream pin and reproduction checks.

| Kind | v0.1 backend | When written | Retrieval | Production backend |
|---|---|---|---|---|
| Working | in-process bounded OrderedDict | every deliberate | key | same |
| Episodic | SQLite | end of episode (outcome, key events) | scope + time window + text (v0.2 vectors) | mem0 vector layer / Graphiti |
| Spatial | SQLite (bbox + radius) | when an object/landmark is observed | near(pose, r) + key + as_of | Spark-DSG (hierarchical scene graph, observation intervals, change-feed) + ConceptGraphs association rules |
| Skill | v0.2: directory + JSON + content hash | **only after promotion through the validation gate** | trigger/precondition matching; embedding used only for recall | same + sandboxed execution |

Skill item payload (v0.2): `{"code" | "params", "api_surface": "hylo.tools@v1", "trigger", "preconditions", "postconditions", "status": proposed|verified|refuted|deprecated, "verdicts": {useful, misleading, wrong, not_applicable}, "evidence": [trace refs]}`.
Skill writes go through the same gate as the Evolution layer (`improvement.md`); this is where Memory and RSI meet.

The experimental model-managed Working Memory runtime, agent wrapper and dedicated
RoboMME runner were removed on 2026-11-04. Historical experiment records remain available.
This removal does not affect the bounded in-process store above, event-memory primitives,
or the independent SimpleMemVLA option.

## 4. Memory and observability

- Every retrieval → `memory.retrieval {query, n_hits, hits[id, pos, score]}`; every write/forget → `memory.write {op, key, events}` **[Validated in prototype]**.
- v0.2: Scorer `memory_helpfulness`: associates retrieval hits with episode outcomes, produces ASPIRE-style `useful/misleading` ratings, and writes them back as item statistics.
- **Stale memory handling** (validated in prototype): when the Reasoner reaches a recalled position but does not perceive the target → `forget` and switch to exploration; in a test that injects a wrong landmark, the episode still succeeds and the memory is corrected to the true position.

## 5. Prototype data

`testing/revisit_landmark` (target visible only within 0.25 m): with memory, the first episode explores and later episodes go straight there; `tests/test_memory.py::test_spatial_memory_shortens_later_episodes` asserts that the total steps of later episodes are significantly lower than in the no-memory version.
