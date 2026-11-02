"""hylo.scene/v0.1: validation of the scene contract and the problems found from the file alone."""

import json

import pytest
from pydantic import ValidationError

from hylo.core.scene import (AllOf, JointAtMost, On, Scene, Sequence, referenced, scene_problems,
                             validation_problems, walk)


def block(name, label=None, **kw):
    return {"name": name, "label": label or name.replace("_", " "), "shape": "box", "size": [0.04, 0.04, 0.04],
            "pose": {"pos": [0.0, 0.0]}, **kw}


def scene(objects, success, **kw):
    return {"name": "t", "instruction": "do the thing", "objects": objects, "success": success, **kw}


DRAWER = [
    {"name": "cabinet", "label": "cabinet", "fixed": True, "pose": {"pos": [0.0, 0.2]},
     "parts": [{"shape": "box", "size": [0.2, 0.2, 0.01], "pos": [0, 0, 0.1]},
               {"shape": "box", "size": [0.2, 0.01, 0.1], "pos": [0, 0.1, 0.05]}]},
    {"name": "drawer", "label": "drawer", "parent": "cabinet", "pose": {"pos": [0, 0, 0.04]}, "shape": "tray",
     "size": [0.17, 0.17, 0.05], "joint": {"type": "slide", "axis": [1, 0, 0], "range": [0, 0.15]}},
    block("red_block", region={"x": [-0.1, 0.0], "y": [-0.2, -0.1], "yaw_deg": [-30, 30]}, pose=None),
]


def test_a_full_scene_validates_round_trips_and_has_a_schema():
    s = Scene.model_validate(scene(DRAWER, {"op": "sequence", "terms": [
        {"op": "joint_at_least", "object": "drawer", "fraction": 0.6},
        {"op": "all_of", "terms": [{"op": "inside", "a": "red_block", "b": "drawer"},
                                   {"op": "not", "term": {"op": "lifted", "a": "red_block"}}]},
        {"op": "joint_at_most", "object": "drawer", "value": 0.02}]}))
    assert s.schema_version == "hylo.scene/v0.1" and isinstance(s.success, Sequence)
    assert isinstance(s.success.terms[1], AllOf) and isinstance(s.success.terms[2], JointAtMost)
    assert Scene.model_validate_json(s.model_dump_json()) == s
    assert [n.op for n in walk(s.success)] == ["sequence", "joint_at_least", "all_of", "inside", "not", "lifted",
                                                 "joint_at_most"]
    assert ("red_block", "a") in referenced(s.success) and ("drawer", "object") in referenced(s.success)
    assert scene_problems(s) == []
    assert s.object("drawer").joint.range == (0.0, 0.15) and s.object("cabinet").movable is False
    schema = json.dumps(Scene.model_json_schema())
    assert "joint_at_least" in schema and "sequence" in schema and "region" in schema


@pytest.mark.parametrize("obj, words", [
    (block("b", size=[0.04, 0.04]), "box needs size"),
    (block("b", shape="sphere", size=[0.02, 0.02]), "sphere needs size"),
    (block("b", shape="capsule", size=[0.03, 0.04]), "at least 2 x radius"),
    (block("b", shape="mesh", size=[1.0]), "needs `mesh`"),
    (block("b", shape="tray", size=[0.1, 0.1, 0.04], wall=0.05), "too thick"),
    (block("b", size=[0.04, 0.04, 3.0]), "(0, 2]"),
    (block("b", parts=[{"shape": "box", "size": [0.1, 0.1, 0.1]}]), "either `shape`"),
    (block("b", region={"x": [0, 0.1], "y": [0, 0.1]}), "`pose` or `region`"),
    (block("b", joint={"type": "slide", "range": [0, 0.1]}), "a joint needs a `parent`"),
    (block("b", parent="a", pose={"pos": [0, 0]}), "[x, y, z] in the parent's frame"),
    (block("b", parent="a", pose={"pos": [0, 0, 0]}, joint={"type": "slide", "range": [0.1, 0.0]}), "low < high"),
    (block("b", parent="a", pose={"pos": [0, 0, 0]}, joint={"type": "slide", "range": [0, 2.0]}), "metres"),
    (block("b", parent="a", pose={"pos": [0, 0, 0]}, joint={"type": "hinge", "range": [0, 90], "initial": 120}),
     "outside its range"),
    ({"name": "b", "label": "b", "shape": "box", "size": [0.04, 0.04, 0.04]}, "needs a `pose`"),
    (block("Bad-Name"), "pattern"),
    (block("b", colour="red", mass=-1), "greater than 0"),
    (block("b", extra_field=1), "Extra inputs"),
    (block("b", colour="light_brown"), "unknown colour"),
    (block("b", colour=[1.0, 2.0, 0.0]), "in 0..1"),
])
def test_invalid_objects_are_rejected_with_a_reason(obj, words):
    with pytest.raises(ValidationError) as err:
        Scene.model_validate(scene([obj], {"op": "on", "a": "b", "b": "table"}))
    assert words in str(err.value)
    probs = validation_problems(err.value)
    assert probs and all(p.code == "schema" for p in probs) and probs[0].message.startswith("objects.0")


@pytest.mark.parametrize("success, words", [
    ({"op": "inside", "a": "b"}, "exactly one of `b`"),
    ({"op": "near", "a": "b", "b": "c", "point": [0, 0]}, "exactly one of `b` or `point`"),
    ({"op": "joint_at_least", "object": "b"}, "exactly one of `value` or `fraction`"),
    ({"op": "joint_at_least", "object": "b", "fraction": 1.5}, "less than or equal to 1"),
    ({"op": "row", "objects": ["b"]}, "at least 2"),
    ({"op": "all_of", "terms": []}, "at least 1"),
    ({"op": "beside", "a": "b", "b": "c"}, "does not match any of the expected tags"),
])
def test_invalid_predicates_are_rejected(success, words):
    with pytest.raises(ValidationError) as err:
        Scene.model_validate(scene([block("b")], success))
    assert words in str(err.value)


def problems(objects, success):
    return scene_problems(Scene.model_validate(scene(objects, success)))


def codes(objects, success):
    return [p.code for p in problems(objects, success)]


def test_static_problems_names_labels_references_cycles():
    on = {"op": "on", "a": "a", "b": "table"}
    assert codes([block("a"), block("a", "other")], on) == ["duplicate"]
    assert codes([block("a", "cube"), block("b", "Cube")], on) == ["duplicate_label"]
    assert codes([block("a")], {"op": "on", "a": "a", "b": "plate"}) == ["reference"]
    assert codes([block("a", on="ghost")], on) == ["reference"]
    assert "cycle" in problems([block("a", on="b"), block("b", on="a")], on)[0].message
    p = problems([block("a")], {"op": "on", "a": "ghost", "b": "table"})[0]
    assert p.code == "reference" and "ghost" in p.message and p.objects == ("ghost",)


def test_static_problems_predicates_that_can_never_hold():
    fixed = [block("a", fixed=True), block("b", "plate", fixed=True)]
    assert codes(fixed, {"op": "on", "a": "a", "b": "b"}) == ["unsatisfiable"]                 # nothing can move
    assert codes([block("a")], {"op": "on", "a": "a", "b": "a"}) == ["unsatisfiable"]
    big = block("big", size=[0.3, 0.3, 0.05])
    cup = {"name": "cup", "label": "cup", "shape": "cup", "size": [0.045, 0.08], "fixed": True, "pose": {"pos": [0.1, 0]}}
    assert codes([big, cup], {"op": "inside", "a": "big", "b": "cup"}) == ["unsatisfiable"]
    assert codes([block("a"), cup], {"op": "inside", "a": "a", "b": "cup"}) == []
    assert codes([block("a")], {"op": "joint_at_least", "object": "a", "value": 0.1}) == ["unsatisfiable"]
    assert codes(DRAWER, {"op": "joint_at_least", "object": "drawer", "value": 0.3}) == ["unsatisfiable"]
    two = [block("a"), block("b"), block("c")]
    assert codes(two, {"op": "all_of", "terms": [{"op": "on", "a": "a", "b": "b"}, {"op": "on", "a": "a", "b": "c"}]}) \
        == ["unsatisfiable"]
    assert codes(two, {"op": "all_of", "terms": [{"op": "on", "a": "a", "b": "b"}, {"op": "on", "a": "b", "b": "a"}]}) \
        == ["unsatisfiable"]
    assert codes(two, {"op": "row", "objects": ["a", "b", "a"]}) == ["unsatisfiable"]
    # a drawer (jointed child of a fixture) moves; the fixture itself does not
    assert codes(DRAWER, {"op": "joint_at_least", "object": "drawer", "fraction": 0.5}) == []
    assert codes(DRAWER, {"op": "near", "a": "cabinet", "point": [0.1, 0.1]}) == ["unsatisfiable"]
    assert isinstance(Scene.model_validate(scene([block("a")], {"op": "on", "a": "a", "b": "table"})).success, On)
