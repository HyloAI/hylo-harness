from hylo.memory.sqlite import SQLiteMemoryStore
from hylo.testing.conformance import check_memory_store


def test_sqlite_store_conforms():
    rep = check_memory_store(lambda: SQLiteMemoryStore())
    assert rep.ok, rep.summary()
