import json

import pytest

from worldcurrent.core.artifact import (ArtifactError, Provenance, SourceRef, SplitLeak, check_splits, development_runs,
                                        dir_digest, read_manifest, referenced_artifacts, verify, write_manifest)

LEARNER = "worldcurrent.testing.learners:SummaryLearner"


def _run(root, name, split, statuses):
    rd = root / name
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({"benchmark": {"split": split}}))
    (rd / "evallog.json").write_text(json.dumps({"run_id": name, "samples": [
        {"episode_id": f"e{i}", "status": s} for i, s in enumerate(statuses)]}))
    return rd


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    _run(root, "a-search", "search", ["success", "failure"])
    _run(root, "a-val", "search_val", ["success", "success"])
    _run(root, "a-gate", "gate", ["success"] * 5)
    _run(root, "a-ho", "heldout", ["failure"] * 5)
    return root


def test_split_guard_skips_or_refuses_gate_and_heldout(runs):
    found = development_runs([runs])
    assert sorted(r.split for _, r in found) == ["search", "search_val"]
    with pytest.raises(SplitLeak):
        development_runs([runs], strict=True)
    with pytest.raises(SplitLeak):
        check_splits(["search", "gate_r1"])


def test_sealed_artifact_detects_tampering_and_refuses_leaky_sources(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "w.bin").write_bytes(b"abc")
    m = write_manifest(art, kind="value_model", role="behaviour",
                       provenance=Provenance(builder="t", sources=[SourceRef(id="r1", split="search")]))
    assert m.digest == dir_digest(art) and verify(art / "w.bin").digest == m.digest   # a file inside resolves too
    (art / "w.bin").write_bytes(b"abd")
    with pytest.raises(ArtifactError):
        verify(art)
    with pytest.raises(SplitLeak):
        write_manifest(art, kind="value_model", role="behaviour",
                       provenance=Provenance(builder="t", sources=[SourceRef(id="r2", split="gate")]))


def test_referenced_artifacts_finds_paths_in_a_patch_and_checks_provenance(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    for d in (good, bad):
        d.mkdir()
        (d / "x").write_text("1")
    write_manifest(good, kind="action_memory", role="behaviour", provenance=Provenance(builder="t"))
    m = write_manifest(bad, kind="action_memory", role="behaviour", provenance=Provenance(builder="t"))
    raw = json.loads((bad / "artifact.json").read_text())        # a manifest edited by hand to hide nothing
    raw["provenance"]["sources"] = [{"kind": "run", "id": "g", "split": "heldout"}]
    (bad / "artifact.json").write_text(json.dumps(raw))
    patch = {"agent.params.memory": {"path": str(good)}, "agent.params.ref": "pkg.mod:factory", "agent.params.n": 3}
    assert list(referenced_artifacts(patch)) == [str(good)]
    with pytest.raises(SplitLeak):
        referenced_artifacts({**patch, "agent.params.other": str(bad / "x")})
    assert m.digest
