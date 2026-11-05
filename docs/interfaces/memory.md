# Interface: Memory

Memory backends and their consumers meet through contracts in `hylo-core`. Choose the contract that matches the
information and lifetime you need; spatial records, ordered events, video history and reusable experience have
different APIs. [Use memory](../../user-guide/using-memory.md) provides the user-facing entry points.

## Spatial and semantic records { #1-data-model }

`MemoryItem` holds typed content with optional identity, entities, scope, pose and time. `MemoryKind` includes
`working`, `episodic`, `semantic`, `spatial` and `skill`; these are record labels, not interchangeable backend
implementations. A `skill`-labelled memory item does not replace a skill-library record or its verification rules.

| Type | Important fields |
|---|---|
| `MemoryItem` | `kind`, `content`, `key`, `payload`, `entities`, `scope`, `pose`, `event_time`, `valid_from`, `confidence`, `source`, `id` |
| `Pose` | `frame_id`, three-dimensional `position`, optional quaternion `orientation` in xyzw order |
| `TimeSpan` | `start`, optional `end`, and `clock` (`sim` or `wall`) |
| `MemoryQuery` | kind/key/text/entity/scope filters, `time_range`, `as_of`, `clock`, `near`, `include_invalidated`, `k` |
| `Hit` | the item, score, scoring details and optional `valid_to` |

Queries must use compatible frames and clock domains. See [Spatial memory](../../user-guide/memory/spatial.md)
for invalidation, duplicate evidence and historical queries, and the
[core memory API](../../api/modules/hylo-core/hylo.core.memory.md) for exact fields and defaults.

## Store protocol { #2-protocol }

```python
class MemoryStore(Protocol):
    def write(self, items: Sequence[MemoryItem]) -> WriteResult: ...
    def query(self, q: MemoryQuery) -> list[Hit]: ...
    def forget(self, q: MemoryQuery) -> int: ...
    def snapshot(self) -> str: ...
```

`write` reports `ADD`, `INVALIDATE` or `SKIP_DUP`. `forget` expires matching records; it does not erase audit
history. `snapshot` identifies the stored content. The SQLite backend implements this protocol. Callers scope
observations to their episode and emit memory events so retrieval and writes can be inspected in the trace.

## Other memory interfaces { #3-four-kinds-of-memory }

| Component | Interface and consumer | Lifetime |
|---|---|---|
| Bounded working store | `WorkingMemory.put/get/clear/items`; a reasoner's scratchpad | clear at episode boundaries |
| Event memory | `EpisodeMemory.reset/append/recall/close`; an observation extractor and an event-memory agent | exact recall within a task/episode scope |
| Video history | `SimpleMemVLAHistory`; the compatible video-conditioned model runtime | session-owned frames and caches, cleared at reset/end |
| Action bank | `ActionMemory`; a compatible policy's sampling/steering path | sealed development artifact, frozen during evaluation |
| Skill library | `SkillLibrary`; the consumer for each skill kind | versioned experience, frozen during evaluation |
| Lesson book | `LessonBook`; an experience-aware proposer | curated development experience |

[Event memory](../../user-guide/memory/events.md) requires an extractor: the journal does not infer facts from
images. [Video memory](../../user-guide/memory/video.md) uses streaming frame history rather than `MemoryStore`.
Its native model integration has separate checkpoint, embodiment and runtime requirements.

Skills have `proposed`, `verified`, `refuted` and `deprecated` states. Campaign-derived skills carry gate evidence;
recipe skills carry model-free replay evidence. These checks establish different claims. Use the
[skill-library rules](../../user-guide/memory/skills.md#status) for the evidence and permitted consumers.

## Lifecycle and observability { #4-memory-and-observability }

- Reset or select an episode scope before recording observations. Do not expose another evaluation episode's history.
- Emit `memory.write` and `memory.retrieval` events from the consuming integration where applicable.
- Keep persistent action banks, skill libraries and lesson books frozen while evaluating an agent.
- Episode-local spatial, event and video history can update from permitted observations during evaluation.
- A content digest identifies stored evidence; it does not establish perception accuracy or a performance gain.

## Verification { #5-prototype-data }

Use conformance and lifecycle checks for the interface, then compare complete agents under matching task and
observation conditions. [Memory results](../../showcase/memory.md) and
[video-memory integration checks](../../showcase/video-memory.md) describe their measured scope separately.
