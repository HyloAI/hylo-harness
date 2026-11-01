import numpy as np
import pytest
from pydantic import ValidationError

from hylo.core.config import ExperimentConfig
from hylo.core.task import InitStateRef, TaskInstance, TaskSpec
from hylo.core.types import Action, ActionChunk, Observation, StepResult


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError, match="extra"):
        TaskSpec(id="t", version="1", instruction="x", max_step=5)  # typo of max_steps


def test_contracts_are_frozen():
    spec = TaskSpec(id="t", version="1", instruction="x")
    with pytest.raises(ValidationError):
        spec.max_steps = 3


def test_arrays_round_trip_through_json():
    obs = Observation(state={"q": np.arange(3, dtype=np.float32)}, extra={"depth": np.ones((2, 2), np.uint16)})
    back = Observation.model_validate_json(obs.model_dump_json())
    assert back.state["q"].dtype == np.float32 and np.array_equal(back.state["q"], [0, 1, 2])
    assert back.extra["depth"].dtype == np.uint16


def test_object_arrays_are_rejected():
    with pytest.raises(ValidationError):
        Action(data=np.array([object()]))


def test_init_spec_is_a_discriminated_union():
    t = TaskSpec(id="libero/x", version="1", instruction="x",
                 init={"kind": "state_ref", "uri": "libero://libero_10/x.pruned_init", "count": 50})
    assert isinstance(t.init, InitStateRef)
    back = TaskInstance.model_validate_json(TaskInstance(spec=t, seed=3).model_dump_json())
    assert back.spec.digest() == t.digest() and back.key == "libero/x#3"


def test_step_result_info_may_carry_arrays():
    r = StepResult(observation=Observation(state={}), reward=None, terminated=False, truncated=True,
                   info={"contact": np.zeros(3)})
    assert isinstance(StepResult.model_validate_json(r.model_dump_json()).info["contact"], np.ndarray)


def test_action_chunk_shape_is_validated():
    with pytest.raises(ValidationError):
        ActionChunk(actions=np.zeros(2))


def test_experiment_schema_is_exported():
    schema = ExperimentConfig.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "hylo.experiment/v0.1"
    assert set(schema["required"]) >= {"name", "benchmark", "env", "agent"}
