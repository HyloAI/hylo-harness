"""The scene builder workflow: the checker's problem classes, the build loop with a scripted model, and a privileged
solve through the kernel (code.llm with a scripted model on mujoco.panda_scene)."""

import copy
import json

import pytest

pytest.importorskip("mujoco")

from hylo.codeagents.scene_check import SceneChecker, check_scene, parse_scene  # noqa: E402
from hylo.codeagents.scenes import (EXAMPLE_1, EXAMPLE_2, PRIVILEGED_REFERENCE, SceneBuilder, extract_json,  # noqa: E402
                                    scene_command, solve_scene)
from hylo.core.registry import registry  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402

try:
    registry.resolve("envs", "mujoco.panda_scene")
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"mujoco.panda_scene is not installed ({exc})", allow_module_level=True)

BASE = {
    "name": "drawer_block", "instruction": "open the drawer and put the red block in it", "max_steps": 900,
    "objects": [
        {"name": "cabinet", "label": "cabinet", "fixed": True, "colour": "wood", "pose": {"pos": [-0.05, 0.22]},
         "parts": [{"shape": "box", "size": [0.2, 0.2, 0.01], "pos": [0, 0, 0.095]},
                   {"shape": "box", "size": [0.2, 0.01, 0.1], "pos": [0, 0.095, 0.045]},
                   {"shape": "box", "size": [0.2, 0.01, 0.1], "pos": [0, -0.095, 0.045]},
                   {"shape": "box", "size": [0.01, 0.2, 0.1], "pos": [-0.095, 0, 0.045]},
                   {"shape": "box", "size": [0.2, 0.2, 0.01], "pos": [0, 0, -0.005]}]},
        {"name": "drawer", "label": "drawer", "parent": "cabinet", "colour": "white", "mass": 0.3, "shape": "tray",
         "size": [0.17, 0.17, 0.05], "pose": {"pos": [0.005, 0, 0.032]},
         "joint": {"type": "slide", "axis": [1, 0, 0], "range": [0, 0.15], "damping": 2}},
        {"name": "red_block", "label": "red block", "shape": "box", "size": [0.04, 0.04, 0.04], "colour": "red",
         "mass": 0.06, "region": {"x": [-0.05, 0.1], "y": [-0.25, -0.1], "yaw_deg": [-30, 30]}},
        {"name": "cup", "label": "blue cup", "shape": "cup", "size": [0.045, 0.08], "colour": "blue", "fixed": True,
         "pose": {"pos": [0.12, 0.0]}},
        {"name": "ball", "label": "yellow ball", "shape": "sphere", "size": [0.02], "colour": "yellow", "mass": 0.03,
         "pose": {"pos": [-0.15, -0.05]}}],
    "success": {"op": "all_of", "terms": [{"op": "joint_at_least", "object": "drawer", "fraction": 0.5},
                                          {"op": "inside", "a": "red_block", "b": "drawer"}]},
}


def edit(fn):
    s = copy.deepcopy(BASE)
    fn(s, {o["name"]: o for o in s["objects"]})
    return s


@pytest.fixture(scope="module")
def checker():
    from hylo.envs.mujoco.assets import panda_assets  # noqa: F401 - the env's assets must be available

    try:
        panda_assets()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    c = SceneChecker(seeds=(0, 1))
    yield c
    c.close()


def check(checker, tmp_path, scene):
    p = tmp_path / "scene.json"
    p.write_text(scene if isinstance(scene, str) else json.dumps(scene))
    return checker.check(p)


def test_a_good_scene_passes_every_stage(checker, tmp_path):
    rep = check(checker, tmp_path, BASE)
    assert rep.ok and rep.stages == ("schema", "static", "physics", "reach", "satisfiable"), rep.text()
    assert set(rep.settled) == {"cabinet", "drawer", "red_block", "cup", "ball"}
    for ex in (EXAMPLE_1, EXAMPLE_2):                                # the prompt's examples are valid scenes
        assert check(checker, tmp_path, ex).ok


CASES = [
    ("json", lambda: '{"name": "x", "objects": [', "not valid JSON"),
    ("schema", lambda: edit(lambda s, o: o["ball"].update(size=[0.02, 0.03])), "sphere needs size"),
    ("duplicate_label", lambda: edit(lambda s, o: o["ball"].update(label="red block")), "share the label"),
    ("reference", lambda: edit(lambda s, o: s.update(success={"op": "on", "a": "green_block", "b": "table"})),
     "not an object"),
    ("unsatisfiable", lambda: edit(lambda s, o: s.update(success={"op": "near", "a": "cup", "point": [0, 0]})),
     "cannot move"),
    ("load", lambda: edit(lambda s, o: o["ball"].update(shape="mesh", mesh="missing.obj", size=[1.0])), "does not load"),
    ("placement", lambda: edit(lambda s, o: o["ball"].update(pose=None, region={"x": [-0.05, -0.04], "y": [-0.2, -0.19]})
                               or o["red_block"].update(region={"x": [-0.05, -0.04], "y": [-0.2, -0.19]})),
     "no free start position"),
    ("penetration", lambda: edit(lambda s, o: o["ball"].update(pose={"pos": [0.12, 0.0]})), "overlap"),
    ("unstable", lambda: edit(lambda s, o: o["ball"].update(pose={"pos": [-0.15, -0.05, 0.3]})), "moved 2"),
    ("unstable", lambda: edit(lambda s, o: o["red_block"].update(region=None, pose={"pos": [0.0, -0.2], "roll_deg": 30})),
     "tilted 30"),
    ("unstable", lambda: edit(lambda s, o: o["drawer"]["joint"].update(axis=[0, 0, 1], range=[-0.03, 0.03])), "drifts"),
    ("fell_off", lambda: edit(lambda s, o: o["ball"].update(pose={"pos": [0.47, 0.0]})), "fell off"),
    ("predicate_true_at_start", lambda: edit(lambda s, o: s.update(success={"op": "on", "a": "ball", "b": "table"})),
     "already holds"),
    ("too_wide", lambda: edit(lambda s, o: o["red_block"].update(size=[0.12, 0.12, 0.04], region=None,
                                                                 pose={"pos": [0.0, -0.2]})), "narrowest"),
    ("schema", lambda: edit(lambda s, o: o["ball"].update(colour="light_brown")), "unknown colour"),
    ("unreachable", lambda: edit(lambda s, o: o["ball"].update(pose={"pos": [0.40, -0.05]})), "cannot be grasped"),
    ("unreachable", lambda: edit(lambda s, o: o["cup"].update(pose={"pos": [0.4, 0.3]})
                                 or s.update(success={"op": "inside", "a": "ball", "b": "cup"})), "cannot bring"),
]


SOLID = {"name": "solid", "instruction": "put the block in the chest", "objects": [
    {"name": "chest", "label": "chest", "shape": "box", "size": [0.15, 0.2, 0.1], "fixed": True, "pose": {"pos": [0.1, 0.2]}},
    {"name": "block", "label": "block", "shape": "box", "size": [0.04, 0.04, 0.04], "pose": {"pos": [-0.1, -0.2]}}],
    "success": {"op": "inside", "a": "block", "b": "chest"}}
CASES.append(("unsatisfiable", lambda: SOLID, "chest is solid where block has to go"))
SPRUNG = copy.deepcopy(SOLID)
SPRUNG["objects"].append({"name": "door", "label": "door", "parent": "chest", "shape": "box", "size": [0.01, 0.2, 0.1],
                          "pose": {"pos": [0.08, 0, 0]}, "joint": {"type": "slide", "axis": [1, 0, 0], "range": [0, 0.1],
                                                                    "spring": 50}})
SPRUNG["success"] = {"op": "all_of", "terms": [{"op": "joint_at_least", "object": "door", "value": 0.05},
                                               {"op": "inside", "a": "block", "b": "chest"}]}
CASES.append(("unsatisfiable", lambda: SPRUNG, "chest is solid"))          # a spring does not hide the rest
CASES.append(("predicate_true_at_start", lambda: edit(lambda s, o: s.update(success={"op": "sequence", "terms": [
    {"op": "joint_at_least", "object": "drawer", "value": 0.0}, {"op": "inside", "a": "red_block", "b": "drawer"}]})),
    "first step of the sequence"))


@pytest.mark.parametrize("code, make, words", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_the_checker_catches_each_problem_class(checker, tmp_path, code, make, words):
    rep = check(checker, tmp_path, make())
    assert not rep.ok
    assert code in [p.code for p in rep.problems], rep.text()
    assert any(words in p.message for p in rep.problems if p.code == code), rep.text()
    assert all(" (seed 1)" not in p.message for p in rep.problems if p.code == code)   # seeds deduplicated


def test_a_sequence_is_arranged_stage_by_stage(checker, tmp_path):
    tower = {"name": "tower", "instruction": "stack the blocks by size", "objects": [
        {"name": n, "label": f"{n} block", "shape": "box", "size": [w, w, 0.04], "pose": {"pos": [-0.15, y]}}
        for n, w, y in (("small", 0.04, -0.2), ("medium", 0.05, 0.0), ("large", 0.06, 0.2))],
        "success": {"op": "sequence", "terms": [{"op": "on", "a": "medium", "b": "large"},
                                                {"op": "on", "a": "small", "b": "medium"}]}}
    rep = check(checker, tmp_path, tower)
    assert rep.ok and "satisfiable" in rep.stages, rep.text()


def test_parse_scene_accepts_text_paths_and_mappings(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(BASE))
    for src in (p, str(p), json.dumps(BASE), BASE):
        scene, probs = parse_scene(src)
        assert scene is not None and probs == [] and scene.name == "drawer_block"
    assert extract_json("plan\n```json\n{\"a\": 1}\n```\nmore") == '{"a": 1}'
    assert extract_json("<think>{no}</think> here {\"a\": {\"b\": 2}} done") == '{"a": {"b": 2}}'
    assert extract_json("nothing") is None


def reply(scene) -> str:
    return f"Plan: a drawer.\n```json\n{json.dumps(scene)}\n```"


def test_the_builder_feeds_problems_back_until_the_scene_passes(checker, tmp_path):
    overlap = edit(lambda s, o: o["ball"].update(pose={"pos": [0.12, 0.0]}))
    model = ScriptedChat(["I will think about it first.", reply(overlap), reply(BASE)])
    res = SceneBuilder(model, attempts=5).build("open the drawer and put the red block in it", tmp_path / "out")
    assert res["built"] and res["attempts"] == 3 and res["usage"]["calls"] == 3
    recs = res["records"]
    assert [r["ok"] for r in recs] == [False, False, True]
    assert recs[0]["problems"][0]["code"] == "format" and recs[1]["problems"][0]["code"] == "penetration"
    assert recs[2]["stages"] == ["schema", "static", "physics", "reach", "satisfiable"]
    system, first = model.calls[0][0]["content"], model.calls[0][1]["content"]
    assert "hylo.scene/v0.1" in system and "joint_at_least" in system and "x [-0.27, 0.2]" in system
    assert first.startswith("Task: open the drawer")
    fb = model.calls[2][-1]["content"]                               # the third call sees the second's problems
    assert "rejected attempt 2" in fb and "[penetration]" in fb and "ball and cup overlap" in fb
    assert "earlier attempts" in fb and "attempt 1: [format]" in fb
    assert len(model.calls[2]) == 4                                  # system, task, last attempt, feedback
    saved = json.loads((tmp_path / "out" / "build.json").read_text())
    assert saved["built"] and len(saved["records"]) == 3
    assert parse_scene(tmp_path / "out" / "scene.json")[0].name == "drawer_block"
    assert sorted(p.name for p in (tmp_path / "out" / "attempts").iterdir()) == ["02.json", "03.json"]


def test_the_builder_stops_at_its_attempt_budget(checker, tmp_path):
    bad = edit(lambda s, o: s.update(success={"op": "on", "a": "ball", "b": "table"}))
    model = ScriptedChat([], then=reply(bad))
    res = SceneBuilder(model, attempts=2).build("whatever", tmp_path / "out")
    assert not res["built"] and res["attempts"] == 2
    assert all(r["problems"][0]["code"] == "predicate_true_at_start" for r in res["records"])


BALL_CUP = {"name": "ball_in_cup", "instruction": "put the ball in the cup",
            "objects": [{"name": "cup", "label": "cup", "shape": "cup", "size": [0.045, 0.08], "colour": "blue",
                         "fixed": True, "pose": {"pos": [0.1, 0.2]}},
                        {"name": "ball", "label": "ball", "shape": "sphere", "size": [0.02], "colour": "red",
                         "mass": 0.05, "region": {"x": [-0.15, -0.05], "y": [-0.25, -0.15]}}],
            "success": {"op": "inside", "a": "ball", "b": "cup"}}

PROGRAM = '''
s = robot.scene(); table = robot.scene_table()
pts = robot.object_points("ball")
box = robot.box(pts, table)
plan = robot.plan(robot.grasps("arm", pts, table), table)
out = robot.run(plan)
up = out["tcp"]["arm"].copy(); up[2, 3] += 0.15
robot.run(robot.plan(robot.Goal("lift", "arm", tcp=up), table))
obj_in_tcp = np.linalg.inv(plan.goal.tcp) @ box.pose
local = (pts - box.pose[:3, 3]) @ box.pose[:3, :3]
goal = robot.place_at("arm", obj_in_tcp, local, xy=s["cup"]["pos"][:2], table=robot.scene_table(s["cup"]["top"]),
                      hover=0.02)
robot.run(robot.plan(goal, table))
print(s["cup"]["interior"]["half"], s["ball"]["movable"], len(pts))
'''


def test_privileged_solve_through_the_kernel(checker, tmp_path):
    d = tmp_path / "ball"
    d.mkdir()
    (d / "scene.json").write_text(json.dumps(BALL_CUP))
    assert check_scene(d / "scene.json").ok
    model = ScriptedChat([f"Pick and place.\n```python\n{PROGRAM}\n```"], then="```python\nfinish('done')\n```")
    res = solve_scene(d, model, turns=4, cameras=[], video=False)
    assert res["success"] and res["status"] == "success" and res["privileged"]
    assert res["env"]["params"]["privileged_obs"] is True and res["tools"].endswith("PrivilegedSceneTools")
    assert res["turns"] >= 1 and res["errors"] == 0
    system = model.calls[0][0]["content"]
    assert "robot.scene()" in system and "privileged" in system and PRIVILEGED_REFERENCE.splitlines()[0] in system
    turns = [json.loads(line) for line in (d / "solve" / "turns.jsonl").read_text().splitlines()]
    assert "True" in turns[0]["output"]
    assert json.loads((d / "solve.json").read_text())["success"]


def test_cli_check(checker, tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(edit(lambda s, o: o["ball"].update(label="red block"))))
    assert scene_command(["check", str(p), "--seeds", "0"]) == 1
    assert "duplicate_label" in capsys.readouterr().out
