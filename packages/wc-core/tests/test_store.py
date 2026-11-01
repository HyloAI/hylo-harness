"""RFC-0005 part 2: store ids instead of paths, deletion protection, sync between stores."""

import json
import shutil

import pytest

from worldcurrent.agents.baseline import PControllerAgent
from worldcurrent.core.artifact import (ArtifactError, Provenance, SourceRef, find_runs, referenced_artifacts,
                                        write_manifest)
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.registry import registry
from worldcurrent.core.store import Store, StoreError, adopt, sync
from worldcurrent.eval.experiment import run_experiment


@registry.register("agents", "testing.p_with_artifact")
class PWithArtifact(PControllerAgent):
    """The P controller, plus a path parameter: what it was given is recorded for the test."""

    seen: list[str] = []

    def __init__(self, path: str, kp: float = 2.0, **kw) -> None:
        super().__init__(kp=kp, **kw)
        PWithArtifact.seen.append(path)


def _run(root, name, split="search"):
    rd = root / name
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({"benchmark": {"split": split}, "config_digest": "c" * 16}))
    (rd / "evallog.json").write_text(json.dumps({"run_id": rd.name, "samples": [{"episode_id": "e0", "status": "success"}]}))
    return rd


def _artifact(root, rel, sources=(), payload="w"):
    d = root / rel
    d.mkdir(parents=True)
    (d / "weights.txt").write_text(payload)
    m = write_manifest(d, kind="other", role="behaviour", provenance=Provenance(builder="test@0.1", sources=[
        SourceRef(kind="run", id=s.name, split="search", path=str(s)) for s in sources]))
    return f"artifact:{m.name}@{m.digest}"


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    run = _run(root, "demos/r-search-1")
    ref = _artifact(root, "learned/gain", [run])
    monkeypatch.setenv("WC_STORE", str(root))
    return Store(root), ref, run


def test_runs_and_artifacts_get_ids_and_provenance_links(store):
    s, ref, run = store
    ids = {e.id for e in s.entries(refresh=True)}
    assert ids == {ref, "run:r-search-1"}
    assert s.get(ref.split("@")[0]).id == ref and s.get(ref[: ref.index("@") + 9]).id == ref    # name, digest prefix
    assert s.referrers("run:r-search-1") == [ref]
    assert s.id_for(run) == "run:r-search-1" and s.ref_for(s.root / "learned/gain") == ref
    assert find_runs(["run:r-search-1"]) == [run]
    with pytest.raises(StoreError, match="with that name"):
        s.get("artifact:gain@ffffffffffff")


def test_a_config_names_content_not_a_place(store, tmp_path, monkeypatch):
    s, ref, _ = store
    cfg = ExperimentConfig(name="t", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                           env={"type": "testing.point_reach"},
                           agent={"type": "testing.p_with_artifact", "params": {"path": ref}})
    PWithArtifact.seen.clear()
    here = run_experiment(cfg, run_root=tmp_path / "out", label="here")
    assert PWithArtifact.seen[0] == str(s.root / "learned/gain")                 # resolved when the agent is built
    manifest = json.loads((here.run_dir / "manifest.json").read_text())
    assert manifest["artifacts"][ref]["id"] == ref and manifest["config"]["agent"]["params"]["path"] == ref

    other = tmp_path / "elsewhere"                                               # the same content on another machine
    shutil.copytree(s.root / "learned", other / "deep/learned")
    monkeypatch.setenv("WC_STORE", str(other))
    PWithArtifact.seen.clear()
    there = run_experiment(cfg, run_root=tmp_path / "out", label="there")
    assert PWithArtifact.seen[0] == str(other / "deep/learned/gain")
    assert json.loads((there.run_dir / "manifest.json").read_text())["config_digest"] == manifest["config_digest"]
    assert there.per_key() == here.per_key()


def test_deletion_is_refused_while_referred_to_and_leaves_a_tombstone(store):
    s, ref, _ = store
    camp = s.root / "evolve/c1"
    camp.mkdir(parents=True)                                     # a campaign ledger naming the artifact
    (camp / "archive.jsonl").write_text(json.dumps({"candidate": {"patch": {"agent.params.path": ref}}}) + "\n")
    s.rebuild()
    with pytest.raises(StoreError, match="referred to by"):
        s.remove("run:r-search-1")                               # the artifact was built from it
    with pytest.raises(StoreError, match="campaign:evolve/c1"):
        s.remove(ref)
    stone = s.remove(ref, force=True, reason="superseded")
    assert stone["forced"] and stone["referrers"] == ["campaign:evolve/c1"]
    with pytest.raises(StoreError, match="removed at .*superseded"):
        s.get(ref)
    assert s.referrers("run:r-search-1") == []                   # nothing left that was built from the run
    s.remove("run:r-search-1")
    assert not (s.root / "demos/r-search-1").exists()


def test_the_gate_checks_ids_and_refuses_unknown_provenance(store, tmp_path):
    s, ref, _ = store
    assert list(referenced_artifacts({"x": ref}).values())[0].digest == ref.split("@")[1]
    (s.root / "learned/gain/weights.txt").write_text("changed")
    with pytest.raises(ArtifactError, match="modified after sealing"):
        referenced_artifacts({"x": ref})
    legacy = s.root / "legacy/critic"
    legacy.mkdir(parents=True)
    (legacy / "w.npz").write_bytes(b"0")
    rep = adopt(legacy, kind="value_model", role="behaviour", store=s)
    assert rep["unknown_provenance"] and rep["id"].startswith("artifact:critic@")
    m = list(referenced_artifacts({"x": rep["id"]}).values())[0]
    assert m.provenance.params["unknown_provenance"] is True                   # protocol.py refuses this in a candidate


def test_sync_copies_what_is_missing_and_refuses_conflicts(store, tmp_path):
    s, ref, _ = store
    robot = Store(tmp_path / "robot")
    _run(robot.root, "workcell/w-1", split="deployment")
    rep = sync(s, str(robot.root), direction="pull")
    assert rep["copied"] == ["run:w-1"] and (s.root / "workcell/w-1/evallog.json").exists()
    rep = sync(s, str(robot.root), direction="push", ids=[ref], with_sources=True)
    assert sorted(rep["copied"]) == sorted([ref, "run:r-search-1"])
    assert Store(robot.root).get(ref).path == "learned/gain"
    assert sync(s, str(robot.root), direction="push", ids=[ref])["skipped"] == [ref]       # sealed: never re-copied
    shutil.rmtree(robot.root / "learned/gain")
    _artifact(robot.root, "learned/gain", payload="different")
    with pytest.raises(StoreError, match="conflict"):
        sync(s, str(robot.root), direction="push", ids=[ref])
