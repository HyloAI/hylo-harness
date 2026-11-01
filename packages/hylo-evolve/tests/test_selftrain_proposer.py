import json
import stat

from hylo.core.config import ExperimentConfig
from hylo.evolve.selftrain import ExpertIterationProposer

FAKE_TRAINER = """#!/usr/bin/env python3
import json, sys, pathlib
a = sys.argv
out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
assert a[a.index("--splits") + 1:a.index("--splits") + 3] == ["search", "search_val"]
(out / "selftrain.json").write_text(json.dumps({"weights": str(out / "w"), "weights_sha256_16": "ab" * 8,
    "dataset": {"episodes": 3, "frames": 600}, "pretrained": a[a.index("--pretrained") + 1]}))
"""


def _cfg(tmp_path, python):
    return ExperimentConfig.model_validate({
        "schema_version": "hylo.experiment/v0.1", "name": "x",
        "benchmark": {"type": "testing/reach-v0", "split": "search"},
        "env": {"type": "testing.point_reach"},
        "agent": {"type": "remote", "params": {"ref": "m:f", "python": str(python), "params": {"repo_id": "org/policy"}}},
        "trace": {"image_every": 1}})


def _run_dir(tmp_path, n_success):
    rd = tmp_path / "run"
    rd.mkdir()
    samples = [{"status": "success" if i < n_success else "failure"} for i in range(10)]
    (rd / "evallog.json").write_text(json.dumps({"samples": samples}))
    return rd


def test_proposes_new_weights_as_a_gated_config_patch(tmp_path):
    fake = tmp_path / "python"
    fake.write_text(FAKE_TRAINER)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    prop = ExpertIterationProposer(out_root=str(tmp_path / "st"), min_successes=3)
    [(cand, hyp)] = prop.propose({"run_dir": str(_run_dir(tmp_path, 5))}, _cfg(tmp_path, fake), "v1", [])
    assert cand.kind == "config_patch" and cand.parent_id == "v1"
    assert list(cand.patch) == ["agent.params.params.repo_id"] and cand.patch["agent.params.params.repo_id"].endswith("/w")
    assert cand.evidence["unit"] == "L4" and cand.evidence["pretrained"] == "org/policy"
    assert "successful search episodes" in hyp


def test_too_few_successes_proposes_nothing(tmp_path):
    prop = ExpertIterationProposer(out_root=str(tmp_path / "st"), min_successes=8)
    assert prop.propose({"run_dir": str(_run_dir(tmp_path, 5))}, _cfg(tmp_path, "/nonexistent"), None, []) == []
