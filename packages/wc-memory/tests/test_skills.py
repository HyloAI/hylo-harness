import pytest

from worldcurrent.core.experience import Skill, SkillTrigger
from worldcurrent.memory.action_bank import FrozenMemoryError
from worldcurrent.memory.skills import SkillLibrary

BOTH = r"put both the (?P<a>.+) and the (?P<b>.+) in the basket"


def skill(name, kind="plan", pattern=None, tags=(), body=None, version="1"):
    return Skill(name=name, version=version, description=f"{name} skill", kind=kind,
                 trigger=SkillTrigger(instruction_pattern=pattern, task_tags=tags),
                 body=body or {"subgoals": ["pick up the {a} and place it in the basket",
                                            "pick up the {b} and place it in the basket"]})


def test_lifecycle_match_bindings_and_specificity(tmp_path):
    lib = SkillLibrary(str(tmp_path))
    lib.add(skill("both-basket", pattern=BOTH, tags=("libero_10",)))
    lib.add(skill("suite-steer", kind="steering", tags=("libero_10",), body={"steering": {"execution_steps": 25}}))
    assert lib.match("Put both the alphabet soup and the tomato sauce in the basket.", task_tags=("libero", "libero_10")) == []
    with pytest.raises(ValueError):
        lib.add(skill("x").model_copy(update={"status": "verified"}))     # verification only through the gate
    lib.transition("both-basket@1", "verified", {"effect": 0.08})
    lib.transition("suite-steer@1", "verified")
    with pytest.raises(ValueError):
        lib.transition("suite-steer@1", "refuted")                       # verified -> refuted is not a transition
    hits = lib.match("Put both the alphabet soup and the tomato sauce in the basket.", task_tags=("libero", "libero_10"))
    assert [s.name for s, _ in hits] == ["both-basket", "suite-steer"]   # pattern match is more specific
    assert hits[0][1] == {"a": "alphabet soup", "b": "tomato sauce"}
    assert [s.name for s, _ in lib.match("open the drawer", task_tags=("libero_10",))] == ["suite-steer"]
    assert lib.match("open the drawer", task_tags=("libero_goal",)) == []
    lib.record_verdict("suite-steer@1", "useful")
    assert lib.get("suite-steer@1").verdicts == {"useful": 1}


def test_persistence_freeze_and_skill_md(tmp_path):
    lib = SkillLibrary(str(tmp_path / "lib"))
    lib.add(skill("both-basket", pattern=BOTH))
    lib.transition("both-basket@1", "verified")
    digest = lib.snapshot()
    frozen = SkillLibrary(str(tmp_path / "lib"), frozen=True)
    assert frozen.snapshot() == digest and len(frozen) == 1
    with pytest.raises(FrozenMemoryError):
        frozen.record_verdict("both-basket@1", "useful")
    assert lib.export_skill_md(str(tmp_path / "md")) == 1
    text = (tmp_path / "md" / "both-basket" / "SKILL.md").read_text()
    assert text.startswith("---\nname: \"both-basket\"") and "pick up the {a}" in text
