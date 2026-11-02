import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))     # tests import manip_testkit (pytest runs with importlib mode)

import manip_testkit as tk  # noqa: E402


@pytest.fixture(scope="session")
def arm_model(tmp_path_factory):
    return tk.write_arm(tmp_path_factory.mktemp("arm"))


@pytest.fixture(scope="session")
def arm_spec(arm_model):
    return tk.arm_spec(arm_model)


@pytest.fixture(scope="session")
def panda_spec():
    path = tk.find_panda()
    if path is None:
        pytest.skip("Franka Panda MJCF not found (set HYLO_PANDA_MJCF)")
    return tk.panda_spec(path)
