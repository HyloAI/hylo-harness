from hylo.agents.composable import make_landmark_agent
from hylo.core.kernel import EpisodeRunner
from hylo.core.task import TaskInstance
from hylo.core.trace import TraceStore
from hylo.core.memory import MemoryItem, MemoryKind, MemoryQuery, Pose, TimeSpan, WorkingMemory
from hylo.memory.sqlite import SQLiteMemoryStore


def item(x, y, key="cup", t=0.0, **kw):
    return MemoryItem(kind=MemoryKind.SPATIAL, key=key, content=f"{key} at {x},{y}",
                      pose=Pose(frame_id="map", position=(x, y, 0.0)), event_time=TimeSpan(start=t), **kw)


def test_spatial_radius_query_uses_real_distance():
    m = SQLiteMemoryStore()
    m.write([item(0, 0, key="a"), item(0.3, 0.3, key="b"), item(2, 2, key="c")])
    hits = m.query(MemoryQuery(near=(Pose(frame_id="map", position=(0, 0, 0)), 0.35)))
    assert {h.item.key for h in hits} == {"a"}          # (0.3, 0.3) is 0.42 away: inside the bbox, outside the radius


def test_frame_mismatch_never_matches():
    m = SQLiteMemoryStore()
    m.write([item(0, 0)])
    assert m.query(MemoryQuery(near=(Pose(frame_id="odom", position=(0, 0, 0)), 1.0))) == []


def test_new_fact_invalidates_old_one_but_history_is_queryable():
    m = SQLiteMemoryStore()
    m.write([item(0, 0, valid_from=10.0)])
    res = m.write([item(1, 1, valid_from=20.0)])
    assert [e[0] for e in res.events] == ["INVALIDATE", "ADD"]
    now = m.query(MemoryQuery(key="cup"))
    assert len(now) == 1 and now[0].item.pose.position[0] == 1
    then = m.query(MemoryQuery(key="cup", as_of=15.0))
    assert len(then) == 1 and then[0].item.pose.position[0] == 0


def test_duplicate_write_is_skipped_and_forget_expires():
    m = SQLiteMemoryStore()
    m.write([item(0, 0)])
    assert m.write([item(0, 0)]).events[0][0] == "SKIP_DUP"
    assert m.forget(MemoryQuery(key="cup")) == 1
    assert m.query(MemoryQuery(key="cup")) == []


def test_working_memory_is_bounded():
    wm = WorkingMemory(capacity=2)
    for i in range(5):
        wm.put(str(i), i)
    assert [k for k, _ in wm.items()] == ["3", "4"]


def test_spatial_memory_shortens_later_episodes(env, tasks, tmp_path):
    """Episode 1 explores; later episodes recall the landmark. Without memory every episode explores."""
    task = tasks["revisit_landmark"]

    def steps(memory_path):
        memory = {"type": "sqlite", "params": {"path": memory_path}} if memory_path else None
        agent = make_landmark_agent(memory=memory)
        runner = EpisodeRunner(env, agent, trace=TraceStore(tmp_path / str(memory_path)))
        return [runner.run(TaskInstance(spec=task, seed=s)).outcome for s in range(4)]

    with_mem = steps(":memory:")
    without = steps(None)
    assert all(o.success for o in with_mem)
    assert sum(o.steps for o in with_mem[1:]) < sum(o.steps for o in without[1:])


def test_stale_memory_is_forgotten(env, tasks):
    agent = make_landmark_agent()
    agent.memory.write([MemoryItem(kind=MemoryKind.SPATIAL, key="goal:testing/revisit_landmark", content="wrong",
                                   pose=Pose(frame_id="testing:world", position=(0.6, -0.6, 0.0)))])
    rec = EpisodeRunner(env, agent).run(TaskInstance(spec=tasks["revisit_landmark"], seed=0))
    assert rec.outcome.success
    hits = agent.memory.query(MemoryQuery(key="goal:testing/revisit_landmark"))
    assert hits and abs(hits[0].item.pose.position[0] - (-0.6)) < 0.06
