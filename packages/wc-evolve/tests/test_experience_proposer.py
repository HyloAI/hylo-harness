from worldcurrent.evolve.candidate import Candidate
from worldcurrent.evolve.experience import ExperienceProposer, credit_lessons
from worldcurrent.evolve.protocol import ArchiveEntry, StageResult
from worldcurrent.memory.lessons import LessonBook
from worldcurrent.memory.skills import SkillLibrary

PATH = "agent.params.params.steering"


def entry(cid, verdict, effect, lessons=()):
    return ArchiveEntry(candidate=Candidate(id=cid, patch={PATH: {"execution_steps": 25}}, rationale="r", parent_id=None,
                                            generator="g", evidence={"lessons": list(lessons)}),
                        hypothesis=f"h {cid}", stage_reached=3, verdict=verdict,
                        stages=[StageResult(stage=3, success=0.5 + effect, parent_success=0.5, n_pairs=100, p_value=0.01)])


class Inner:
    name = "inner@0"

    def __init__(self):
        self.seen = None

    def propose(self, diagnosis, cfg, parent_id, archive):
        self.seen = diagnosis
        return []


class Reflector:
    def __init__(self):
        self.calls = 0

    def reflect(self, diagnosis, book):
        self.calls += 1
        return [{"op": "ADD", "scope": "task:libero/libero_10/T1", "kind": "strategy", "title": "approach slowly",
                 "content": "c", "evidence": [{"run_id": "r", "episode_id": "e", "split": "search"}]}]


def test_credit_is_once_per_decided_candidate(tmp_path):
    book = LessonBook(str(tmp_path))
    lid = book.apply([{"op": "ADD", "scope": "global", "kind": "strategy", "title": "t", "content": "c"}])[0]["id"]
    archive = [entry("cand-a", "promoted", 0.1, [lid]), entry("cand-b", "gated_fail", -0.05, [lid]),
               entry("cand-c", "gated_fail", 0.05, [lid])]                    # underpowered: no credit
    credit_lessons(book, archive)
    credit_lessons(book, archive)
    les = book.all()[0]
    assert (les.helpful, les.harmful) == (1, 1)


def test_experience_proposer_feeds_lessons_and_numberless_skills(tmp_path):
    inner, refl = Inner(), Reflector()
    prop = ExperienceProposer(inner, lessons=LessonBook(str(tmp_path / "l")), skills=SkillLibrary(str(tmp_path / "s")),
                              reflector=refl, skill_tags=["libero_10"], env="libero")
    diag = {"run_dir": str(tmp_path / "run-1"), "per_task": {"T1": {"stall": 2}}}
    prop.propose(diag, None, None, [entry("cand-a", "promoted", 0.1)])
    prop.propose(diag, None, None, [entry("cand-a", "promoted", 0.1)])     # resumed: nothing is redone
    assert refl.calls == 1
    assert [l["title"] for l in inner.seen["lessons"]] == ["approach slowly"]
    [skill] = inner.seen["skills"]
    assert skill["status"] == "verified" and "evidence" not in skill and "0.1" not in str(skill)
