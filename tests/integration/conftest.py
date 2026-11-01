import pytest

from hylo.testing.point_reach import PointReachEnv, reach_tasks
from hylo.core.task import TaskInstance


@pytest.fixture
def env():
    return PointReachEnv()


@pytest.fixture
def tasks():
    return {t.id.split("/")[1]: t for t in reach_tasks()}


@pytest.fixture
def instance(tasks):
    return TaskInstance(spec=tasks["reach_fixed"], seed=3)
