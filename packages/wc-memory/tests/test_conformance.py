from worldcurrent.memory.sqlite import SQLiteMemoryStore
from worldcurrent.testing.conformance import check_memory_store


def test_sqlite_store_conforms():
    rep = check_memory_store(lambda: SQLiteMemoryStore())
    assert rep.ok, rep.summary()
