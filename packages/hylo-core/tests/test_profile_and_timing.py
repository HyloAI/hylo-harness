"""RFC-0005: what an experiment measures vs where it runs (profiles), and one episode loop for sync and realtime."""

import sys

import pytest
import yaml

from hylo.core.config import ExperimentConfig
from hylo.core.profile import resolve
from hylo.eval.experiment import run_experiment

BASE = ExperimentConfig(name="t", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                        env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}})


@pytest.fixture
def profile(tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "hylo.profile/v0.1", "runtimes": {
        "here": {"python": sys.executable, "env": {"HYLO_TEST_FLAG": "~/x", "KEEP": "1"}}}}))
    monkeypatch.setenv("HYLO_PROFILE", str(path))
    return path


def test_where_components_run_does_not_change_what_is_measured():
    a = BASE.with_overrides({"env.remote": {"runtime": "libero-sim"}})
    b = BASE.with_overrides({"env.remote": {"python": "/opt/venv/bin/python", "env": {"LD_LIBRARY_PATH": "/x"}}})
    assert a.digest() == b.digest() == BASE.digest()
    remote = {"type": "remote", "params": {"ref": "testing.p_controller", "params": {"kp": 2.0}, "runtime": "lerobot"}}
    moved = {"type": "remote", "params": {"ref": "testing.p_controller", "params": {"kp": 2.0}, "python": "/elsewhere/python",
                                          "env": {"HF_HOME": "/cache"}}}
    assert BASE.with_overrides({"agent": remote}).digest() == BASE.with_overrides({"agent": moved}).digest()
    assert BASE.with_overrides({"agent.params.kp": 3.0}).digest() != BASE.digest()           # the policy is measured


def test_timing_is_measured_but_the_default_keeps_older_digests():
    assert "timing" not in BASE.measured()
    rt = BASE.with_overrides({"timing": {"mode": "realtime", "latency_s": 0.2}})
    assert rt.measured()["timing"]["latency_s"] == 0.2 and rt.digest() != BASE.digest()


def test_runtimes_come_from_the_machine_profile_and_explicit_values_win(profile, monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    py, env = resolve("here", env={"KEEP": "2"})
    assert py == sys.executable and env == {"HYLO_TEST_FLAG": "/home/someone/x", "KEEP": "2"}
    assert resolve(None, python="/p")[0] == "/p"
    with pytest.raises(KeyError, match="hylo profile set-runtime nowhere"):
        resolve("nowhere")


def test_a_remote_environment_starts_from_a_named_runtime(profile, tmp_path):
    local = run_experiment(BASE, run_root=tmp_path, label="local")
    remote = run_experiment(BASE.with_overrides({"env.remote": {"runtime": "here"}}), run_root=tmp_path, label="remote")
    assert remote.per_key() == local.per_key()


def test_benchmarks_can_measure_realtime_execution(tmp_path):
    sync = run_experiment(BASE, run_root=tmp_path, label="sync")
    zero = run_experiment(BASE.with_overrides({"timing": {"mode": "realtime", "latency_s": 0.0, "refill_steps": 0,
                                                          "adaptive_refill": False}}), run_root=tmp_path, label="zero")
    assert {s["key"]: s["steps"] for s in zero.log["samples"]} == {s["key"]: s["steps"] for s in sync.log["samples"]}
    slow = run_experiment(BASE.with_overrides({"agent.params": {"kp": 2.0, "horizon": 8, "execution_steps": 8},
                                               "timing": {"mode": "realtime", "latency_s": 0.25}}), run_root=tmp_path,
                          label="slow")
    from hylo.core.trace import TraceStore

    end = TraceStore(slow.run_dir).read(slow.log["samples"][0]["episode_id"]).last("episode.end")["payload"]
    assert end["holds"] > 0 and end["chunks"] > 0 and slow.metrics["success_mean"] > 0.5
