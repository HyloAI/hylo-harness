import pytest

from worldcurrent.testing.point_reach import PointReachEnv, reach_tasks
from worldcurrent.core.task import TaskInstance


@pytest.fixture
def env():
    return PointReachEnv()


@pytest.fixture
def tasks():
    return {t.id.split("/")[1]: t for t in reach_tasks()}


@pytest.fixture
def instance(tasks):
    return TaskInstance(spec=tasks["reach_fixed"], seed=3)
