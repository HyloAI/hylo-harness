import json
import stat

from hylo.core.config import ExperimentConfig
from hylo.evolve.candidate import check_protected_scope
from hylo.evolve.critic import CRITIC_FACTORY, CriticProposer

FAKE = """#!/usr/bin/env python3
import json, sys, pathlib
a = sys.argv
assert a[a.index("--splits") + 1:a.index("--splits") + 3] == ["search", "search_val"]
out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
(out / "critic.json").write_text(json.dumps({"weights_sha256_16": "cd" * 8,
    "trained_on": {"cv": {"same_state_pair_accuracy": ACC, "same_state_pairs": 120}}}))
"""

CFG = ExperimentConfig.model_validate({
    "schema_version": "hylo.experiment/v0.1", "name": "x", "benchmark": {"type": "testing/reach-v0", "split": "search"},
    "env": {"type": "testing.point_reach"},
    "agent": {"type": "remote", "params": {"ref": "m:make_steered", "params": {"repo_id": "r", "steering": {"execution_steps": 25}}}}})


def _fake(tmp_path, acc):
    f = tmp_path / f"py{acc}"
    f.write_text(FAKE.replace("ACC", str(acc)))
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(f)


def test_proposes_critic_steering_when_the_critic_ranks_chunks(tmp_path):
    prop = CriticProposer(out_root=str(tmp_path / "c"), betas=(0.0, 0.5), python=_fake(tmp_path, 0.63))
    out = prop.propose({"run_dir": str(tmp_path / "cycle/runs/parent")}, CFG, "v2", [])
    assert len(out) == 2
    cand = out[0][0]
    check_protected_scope(cand)
    assert cand.patch["agent.params.ref"] == CRITIC_FACTORY
    assert cand.patch["agent.params.params.steering"] == {"execution_steps": 25, "n_samples": 4,
                                                         "selector": "gripper_vote_medoid", "seed_noise": True}
    assert cand.patch["agent.params.params.critic"]["beta"] == 0.0 and cand.evidence["unit"] == "L3"
    patched = CFG.with_overrides(cand.patch)
    assert patched.agent.params["params"]["repo_id"] == "r"      # the policy itself is untouched


def test_a_critic_at_chance_is_not_proposed(tmp_path):
    prop = CriticProposer(out_root=str(tmp_path / "c"), python=_fake(tmp_path, 0.51))
    assert prop.propose({}, CFG, None, []) == []
