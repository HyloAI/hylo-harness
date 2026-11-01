import pytest
from pydantic import ValidationError

from hylo.eval.benchmark import Benchmark
from hylo.eval.scoring import one_sided_exact_mcnemar, paired_compare
from hylo.core.config import ExperimentConfig
from hylo.core.registry import registry


def test_mcnemar_matches_binomial_tail():
    assert one_sided_exact_mcnemar(5, 0) == pytest.approx(1 / 32)
    assert one_sided_exact_mcnemar(0, 0) == 1.0
    assert one_sided_exact_mcnemar(3, 3) > 0.5


def test_paired_compare_drops_abstentions():
    cmp = paired_compare({"a": 1.0, "b": 1.0, "c": None}, {"a": 0.0, "b": 1.0, "c": 0.0})
    assert (cmp.n_pairs, cmp.candidate_wins, cmp.parent_wins) == (2, 1, 0)


def test_benchmark_rejects_seed_shared_between_splits(tasks):
    with pytest.raises(ValidationError, match="both"):
        Benchmark(id="x", version="0", tasks=(tasks["reach_fixed"],), splits={"train": (1, 2), "heldout": (2,)})


def _cfg(**kw):
    base = dict(name="t", benchmark={"type": "testing/reach-v0"}, env={"type": "testing.point_reach"},
                agent={"type": "testing.p_controller", "params": {"kp": 1.0}})
    return ExperimentConfig(**{**base, **kw})


def test_config_digest_is_stable_and_ignores_output_dir():
    assert _cfg().digest() == _cfg(output_dir="elsewhere").digest()
    assert _cfg().digest() != _cfg(seed=1).digest()


def test_config_overrides_and_schema_version():
    assert _cfg().with_overrides({"agent.params.kp": 4.0}).agent.params["kp"] == 4.0
    with pytest.raises(ValidationError, match="schema_version"):
        ExperimentConfig.model_validate({**_cfg().model_dump(), "schema_version": "hylo.experiment/v9"})


def test_registry_resolves_names_and_refs():
    assert registry.resolve("envs", "testing.point_reach").__name__ == "PointReachEnv"
    assert registry.resolve("envs", "hylo.testing.point_reach:PointReachEnv").__name__ == "PointReachEnv"
    with pytest.raises(KeyError, match="known"):
        registry.resolve("envs", "nope")


def test_parallel_workers_match_serial_results(tmp_path):
    from hylo.eval.experiment import run_experiment

    cfg = _cfg(benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "validation": 6}},
               agent={"type": "testing.p_controller", "params": {"kp": 1.0}})
    serial = run_experiment(cfg, split="validation", run_root=tmp_path)
    parallel = run_experiment(cfg.with_overrides({"workers": 4}), split="validation", run_root=tmp_path)
    assert [s["key"] for s in serial.log["samples"]] == [s["key"] for s in parallel.log["samples"]]
    assert serial.per_key() == parallel.per_key()
