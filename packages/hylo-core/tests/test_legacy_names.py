"""Data written before the rename to Hylo (2026-11-01) stays readable: wc.* schema ids, worldcurrent.* modules."""

import json

from hylo.core.artifact import verify
from hylo.core.config import ExperimentConfig
from hylo.core.contract import current_module, current_schema
from hylo.core.plugins import compatible, parse
from hylo.core.registry import Registry
from hylo.core.remote.workcell import PolicyEndpoint


def test_schema_ids_and_modules_from_before_the_rename_are_read_as_hylo():
    assert current_schema("wc.critic/v0.1") == "hylo.critic/v0.1"
    assert current_schema("hylo.critic/v0.1") == "hylo.critic/v0.1" and current_schema("wc.other") == "wc.other"
    assert current_module("worldcurrent.agents.baseline:PControllerAgent") == "hylo.agents.baseline:PControllerAgent"
    assert current_module("wc_examples.policies:IKReacherPolicy") == "hylo_examples.policies:IKReacherPolicy"


def test_an_old_experiment_config_loads_and_its_components_resolve():
    cfg = ExperimentConfig.model_validate({
        "schema_version": "wc.experiment/v0.1", "name": "old",
        "benchmark": {"type": "testing/reach-v0"}, "env": {"type": "testing.point_reach"},
        "agent": {"type": "worldcurrent.agents.baseline:PControllerAgent", "params": {"kp": 2.0}}})
    assert cfg.schema_version == "hylo.experiment/v0.1"
    assert Registry().resolve("agents", cfg.agent.type).__name__ == "PControllerAgent"
    assert Registry().reference("agents", "worldcurrent.agents.baseline:PControllerAgent").startswith("hylo.")


def test_an_artifact_sealed_before_the_rename_still_verifies(tmp_path):
    art = tmp_path / "old-judge"
    art.mkdir()
    (art / "critic.json").write_text(json.dumps({"schema": "wc.critic/v0.1"}))
    from hylo.core.artifact import dir_digest

    (art / "artifact.json").write_text(json.dumps({
        "schema_version": "wc.artifact/v0.1", "kind": "success_judge", "role": "instrument", "name": "old-judge",
        "digest": dir_digest(art), "provenance": {"builder": "worldcurrent.learn.judges@0.1"}}))
    assert verify(art).schema_version == "hylo.artifact/v0.1"


def test_plugins_and_endpoints_that_still_say_wc():
    assert parse("wc.policy/v0.1") == ("hylo.policy", 0, 1) and compatible("wc.policy/v0.1", "hylo.policy/v0.1")
    assert PolicyEndpoint(url="ws://gpu:8000/wc", protocol="wc").protocol == "hylo"
