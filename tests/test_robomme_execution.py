import importlib.util
from pathlib import Path

import numpy as np
import pytest


DIRECTORY = Path(__file__).parents[1] / "examples/memory/robomme"
spec = importlib.util.spec_from_file_location("execution_checks", DIRECTORY / "execution.py")
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def test_urdf_chain_applies_origin_before_joint_rotation(tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text('''<robot name="test"><link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="extension" type="fixed"><parent link="arm"/><child link="tip"/>
        <origin xyz="1 0 0"/></joint></robot>''')
    fk = execution.UrdfChain(path, ["hinge"], "tip")
    assert np.allclose(fk([np.pi / 2]), [1, 1, 0])
    with pytest.raises(ValueError):
        fk([float("nan")])


def state(z, aperture):
    return np.array([z, 0, 0, 0, 0, 0, 0, aperture])


def monitor():
    m = execution.PickPlaceMonitor(lambda q: np.array([0, 0, q[0]]))
    m.accept("pick up the red cube at <point_1>", "pick up the red cube at <10, 10>")
    return m


def test_press_continuity_is_bounded_and_does_not_emit_completion():
    m = monitor()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    m.accept("press the button at <point_1>", "press the button at <10, 10>")
    for step in range(1, 65):
        assert m.gate("pick up the red cube")[1] == "continue_press_primitive"
        assert not m.observe(state(0.1, 0.04), 1, frame, step)
    assert m.gate("pick up the red cube")[1] is None


def test_pick_place_events_need_lift_low_release_and_rgb_evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[8:14, 8:14] = [255, 0, 0]
    m = monitor()
    for step in range(1, 4):
        assert not m.observe(state(0.01, 0.018), -1, frame, step)
    assert m.gate("place the red cube onto target")[1] is not None
    assert not m.observe(state(0.25, 0.018), -1, frame, 4)
    assert m.gate("place the red cube onto target")[1] is None
    m.accept("place the red cube onto target", "place the red cube onto target", (10.5, 10.5))
    assert m.gate("pick up the red cube")[1] is not None
    events = m.observe(state(0.02, 0.04), 1, frame, 5)
    assert len(events) == 1 and events[0].evidence == ("frame:3", "frame:4", "frame:5")
    assert events[0].value["status"] == "sensor_supported_hypothesis"
    assert not m.observe(state(0.02, 0.04), 1, frame, 6)


@pytest.mark.parametrize("lift,release_z,visible", [(False, 0.02, True), (True, 0.25, True), (True, 0.02, False)])
def test_incomplete_or_ambiguous_cycles_are_not_counted(monkeypatch, lift, release_z, visible):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    if visible:
        frame[8:14, 8:14] = [255, 0, 0]
    m = monitor()
    for step in range(1, 4):
        m.observe(state(0.01, 0.018), -1, frame, step)
    m.observe(state(0.25 if lift else 0.04, 0.018), -1, frame, 4)
    m.accept("place the red cube onto target", "place the red cube onto target", (10.5, 10.5))
    assert not m.observe(state(release_z, 0.04), 1, frame, 5)
