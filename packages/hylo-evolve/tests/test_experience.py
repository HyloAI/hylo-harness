from hylo.evolve.candidate import Candidate
from hylo.evolve.experience import classify, skills_from_archive
from hylo.evolve.protocol import ArchiveEntry, StageResult
from hylo.memory.skills import SkillLibrary

PATH = "agent.params.params.steering"


def entry(cid, verdict, effect=None, stage=3):
    stages = [StageResult(stage=1)]
    if effect is not None:
        stages.append(StageResult(stage=stage, success=0.5 + effect, parent_success=0.5, n_pairs=100, p_value=0.07))
    return ArchiveEntry(candidate=Candidate(id=cid, patch={PATH: {"execution_steps": 25}}, rationale="r",
                                            parent_id=None, generator="g"),
                        hypothesis=f"hyp {cid}", stage_reached=3, stages=stages, verdict=verdict)


def test_status_follows_evidence():
    assert classify(entry("a", "promoted", 0.1))[0] == "verified"
    assert classify(entry("b", "gated_fail", 0.09))[0] == "proposed"      # positive but underpowered
    assert classify(entry("c", "gated_fail", -0.02))[0] == "refuted"
    assert classify(entry("d", "rejected"))[0] == "refuted"
    assert classify(entry("e", "archived", 0.01, stage=2))[0] == "proposed"
    assert classify(entry("b", "gated_fail", 0.09))[1]["effect"] == 0.09


def test_archive_to_library_is_idempotent(tmp_path):
    arch = tmp_path / "archive.jsonl"
    arch.write_text("\n".join(e.model_dump_json() for e in
                              [entry("cand-a", "promoted", 0.1), entry("cand-c", "gated_fail", -0.02)]) + "\n")
    lib = SkillLibrary(str(tmp_path / "skills"))
    s1 = skills_from_archive(tmp_path, lib, task_tags=("libero_10",), env="libero", prefix="l10")
    assert s1["by_status"] == {"verified": 1, "refuted": 1}
    assert skills_from_archive(tmp_path, lib, task_tags=("libero_10",), prefix="l10")["added"] == []
    [(s, _)] = lib.match("anything", task_tags=("libero", "libero_10"), env="libero")
    assert s.name == "l10-a" and s.body == {"steering": {"execution_steps": 25}} and s.evidence["effect"] == 0.1
