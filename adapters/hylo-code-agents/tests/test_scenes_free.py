"""Free-mode scene building: the scene-program allow-list, the free checker's problem classes, the build loop with a
scripted model, and a privileged two-arm solve through the kernel (code.llm with a scripted model on
mujoco.free_scene)."""

import json
import textwrap

import pytest

pytest.importorskip("mujoco")

from hylo.codeagents.sandbox import CodeRejected  # noqa: E402
from hylo.codeagents.scene_free import FreeSceneBuilder, is_free_scene, solve_free_scene  # noqa: E402
from hylo.codeagents.scene_free_check import (FreeSceneChecker, movable_points, program_declarations,  # noqa: E402
                                              validate_scene_program)
from hylo.codeagents.scene_free_prompt import EXAMPLE_1, EXAMPLE_2, free_builder_system  # noqa: E402
from hylo.codeagents.scene_free_tools import FREE_REFERENCE  # noqa: E402
from hylo.codeagents.scenes import scene_command  # noqa: E402
from hylo.core.registry import registry  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402

try:
    registry.resolve("envs", "mujoco.free_scene")
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"mujoco.free_scene is not installed ({exc})", allow_module_level=True)


@pytest.fixture(scope="module")
def checker():
    from hylo.envs.mujoco.assets import ALOHA, PANDA, assets

    try:
        assets(PANDA)
        assets(ALOHA)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    c = FreeSceneChecker(seeds=(0,))
    yield c
    c.close()


def write(tmp_path, source, name="s"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "scene.py").write_text(textwrap.dedent(source))
    return d


def panda(body, success="return False", extra="", robot="panda"):
    """A program on the Panda table whose world is the MJCF worldbody `body` (table top z = 0)."""
    return (f'ROBOT = "{robot}"\nINSTRUCTION = "do it"\nMAX_STEPS = 300\n\n'
            f'def build(scene):\n    scene.add_mjcf("""<mujoco>{body}</mujoco>""")\n\n'
            f"def success(sim):\n    {success}\n" + textwrap.dedent(extra))


BOX = '<worldbody><body name="cube" pos="{x} {y} {z}"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>{more}</worldbody>'


def cube(x=0.0, y=0.1, z=0.0201, more=""):
    return BOX.format(x=x, y=y, z=z, more=more)


LIFT_DEMO = """
def demo(sim):
    sim.grab("cube", hand=0)
    sim.move(0, sim.pos("cube") + [0, 0, 0.1], seconds=1.0)
"""
LIFTED = "return bool(sim.pos('cube')[2] > 0.06)"


def test_the_scene_allow_list_is_wider_than_robot_programs_but_still_guarded():
    validate_scene_program("import mujoco\nimport numpy as np\nROBOT = 'panda'")
    for bad, words in [("import os", "import os"), ("import mujoco\nmujoco.MjSpec.from_file('x')", "from_file"),
                       ("x = ().__class__", "__class__"), ("open('f')", "open"), ("class A: pass", "class"),
                       ("spec.to_xml(); spec.to_file('x')", "to_file")]:
        with pytest.raises(CodeRejected, match=words):
            validate_scene_program(bad)
    decl, probs = program_declarations("ROBOT = 'ur5'\nINSTRUCTION = 'x'\ndef build(scene): pass\n")
    assert decl["ROBOT"] == "ur5" and {p.message.split()[0] for p in probs} == {"ROBOT", "the"}


def test_good_programs_pass_every_stage(checker, tmp_path):
    for i, src in enumerate((EXAMPLE_1, EXAMPLE_2, panda(cube(), LIFTED, LIFT_DEMO))):
        rep = checker.check(write(tmp_path, src, f"g{i}"))
        assert rep.ok and rep.stages == ("program", "load", "physics", "reach", "satisfiable"), rep.text()
        assert not [w for w in rep.warnings if w.code == "unverified"]
    assert set(rep.settled) == {"cube"} and rep.settled["cube"]["free"]


CASES = [
    ("program", lambda: "import os\n" + panda(cube()), "import os is not allowed"),
    ("program", lambda: panda(cube()).replace("def success", "def succeed"), "success(...)"),
    ("load", lambda: panda("<worldbody><body name='x'><geom type='nope'/></body></worldbody>"), "does not parse"),
    ("load", lambda: panda(cube(more="<body name='w' pos='0.1 0 0.05'><joint name='h'/><geom size='0.01'/></body>")
                           + "<actuator><motor joint='h'/></actuator>"), "actuator"),
    ("penetration", lambda: panda(cube(more='<body name="slab" pos="0 0.1 0.02"><geom type="box" size="0.05 0.05 0.02"/>'
                                           '</body>')), "overlap"),
    ("robot_overlap", lambda: panda(cube(x=-0.5, y=0.0, z=0.06)), "overlaps the robot"),
    ("unstable", lambda: panda(cube(z=0.3)), "does not stay where it starts"),
    ("fell_off", lambda: panda(cube(x=0.5, y=0.0, z=0.2)), "off the table"),
    ("success_true_at_start", lambda: panda(cube(), "return True"), "already True at the start"),
    ("success_error", lambda: panda(cube(), "return sim.pos('ghost')[2] > 1"), "no body 'ghost'"),
    ("success_type", lambda: panda(cube(), "return 0"), "not a bool"),
    ("unreachable", lambda: panda(cube(x=0.42, y=0.6)), "no arm reaches any point of cube"),
    ("unsatisfiable", lambda: panda(cube(), LIFTED, "def demo(sim):\n    sim.wait(0.5)\n"), "never became True"),
    ("demo_error", lambda: panda(cube(), LIFTED, "def demo(sim):\n    sim.grab('ghost')\n"), "no body 'ghost'"),
    ("unreachable", lambda: panda(cube(), "return bool(sim.pos('cube')[2] > 0.06)",
                                  "def demo(sim):\n    sim.set_body('cube', pos=[0.42, 0.6, 0.0201])\n"
                                  "    sim.grab('cube')\n    sim.move(0, [0.42, 0.6, 0.2])\n"), "demo grabs cube"),
]


@pytest.mark.parametrize("code,make,words", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_the_checker_catches_each_problem_class(checker, tmp_path, code, make, words):
    rep = checker.check(write(tmp_path, make()))
    assert not rep.ok and code in [p.code for p in rep.problems], rep.text()
    assert words in rep.text(), rep.text()


def test_warnings_do_not_fail_a_program(checker, tmp_path):
    rope = ('<extension><plugin plugin="mujoco.elasticity.cable"/></extension><worldbody>'
            '<composite prefix="rope" type="cable" curve="s" count="12 1 1" size="0.24" offset="-0.1 0.1 0.012" '
            'initial="none"><plugin plugin="mujoco.elasticity.cable"><config key="bend" value="1e5"/></plugin>'
            '<joint kind="main" damping="0.02"/><geom type="capsule" size="0.012" density="300"/></composite></worldbody>')
    rep = checker.check(write(tmp_path, panda(rope, "return bool(sim.nodes('rope')[-1][2] > 0.2)")))
    assert rep.ok, rep.text()
    assert {w.code for w in rep.warnings} == {"anchored", "unverified"}
    assert rep.settled["rope"]["nodes"] == 11


def test_movable_points_cover_bodies_cables_and_cloth():
    state = {"bodies": {"a": {"free": True, "bounds": [[0, 0, 0], [0.04, 0.04, 0.04]]},
                        "wall": {"free": False, "bounds": [[0, 0, 0], [1, 1, 1]]}},
             "joints": {}, "cables": {"r": {"nodes": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}},
             "flexes": {"c": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]}}}
    pts = movable_points(state)
    assert set(pts) == {"a", "r", "c"} and len(pts["c"]) == 5 and pts["r"][1] == [1, 0, 0]


def reply(src: str) -> str:
    return f"Plan: a scene.\n```python\n{src}\n```"


def test_the_builder_feeds_checker_problems_back_until_the_program_passes(checker, tmp_path):
    model = ScriptedChat(["I will think first.", reply("import os\n" + EXAMPLE_1),
                          reply(panda(cube(z=0.3), LIFTED, LIFT_DEMO)), reply(panda(cube(), LIFTED, LIFT_DEMO))])
    res = FreeSceneBuilder(model, attempts=5, seeds=(0,)).build("lift the cube", tmp_path / "out")
    assert res["built"] and res["attempts"] == 4 and res["usage"]["calls"] == 4 and res["mode"] == "free"
    recs = res["records"]
    assert [r["ok"] for r in recs] == [False, False, False, True]
    assert [r["problems"][0]["code"] if r["problems"] else None for r in recs] == ["format", "program", "unstable", None]
    assert recs[3]["stages"] == ["program", "load", "physics", "reach", "satisfiable"]
    system, first = model.calls[0][0]["content"], model.calls[0][1]["content"]
    assert "hylo.freescene/v0.1" in system and "- aloha: arms left" in system and "- g1:" in system
    assert "flexcomp" in system and "composite" in system and first.startswith("Task: lift the cube")
    fb = model.calls[3][-1]["content"]                        # the fourth call sees the third's problems
    assert "rejected the program" in fb and "[unstable]" in fb and "cube" in fb and "Remaining: attempts" in fb
    saved = json.loads((tmp_path / "out" / "build.json").read_text())
    assert saved["built"] and len(saved["records"]) == 4
    assert sorted(p.name for p in (tmp_path / "out" / "attempts").iterdir()) == ["02.py", "03.py"]
    assert "sim.grab" in (tmp_path / "out" / "scene.py").read_text()
    assert is_free_scene(tmp_path / "out") and not is_free_scene(tmp_path)


def test_the_builder_stops_at_its_attempt_budget(checker, tmp_path):
    model = ScriptedChat([], then=reply(panda(cube(), "return True")))
    res = FreeSceneBuilder(model, attempts=2, seeds=(0,)).build("whatever", tmp_path / "out")
    assert not res["built"] and res["attempts"] == 2
    assert all(r["problems"][0]["code"] == "success_true_at_start" for r in res["records"])


CABLE_TASK = '''
ROBOT = "aloha"
INSTRUCTION = "lift the cable by both ends"
MAX_STEPS = 400

def build(scene):
    scene.add_mjcf(f"""<mujoco><extension><plugin plugin="mujoco.elasticity.cable"/></extension><worldbody>
      <composite prefix="cable" type="cable" curve="s" count="17 1 1" size="0.44" offset="-0.22 0.05 {scene.table_top + 0.012}"
                 initial="free">
        <plugin plugin="mujoco.elasticity.cable"><config key="twist" value="1e6"/><config key="bend" value="1e6"/>
          <config key="vmax" value="0"/></plugin>
        <joint kind="main" damping="0.02"/><geom type="capsule" size="0.012" density="300" condim="4"/>
      </composite></worldbody></mujoco>""")

def success(sim):
    n = sim.nodes("cable")
    return bool(min(n[0][2], n[-1][2]) > sim.table_top + 0.1)
'''

SOLUTION = '''
s = robot.scene(); table = robot.scene_table()
n = s["cables"]["cable"]["nodes"]
plans = {}
for arm, i, j in (("left", 0, 1), ("right", -1, -2)):
    d = n[i] - n[j]; yaw = math.atan2(d[1], d[0]) + math.pi / 2
    p = n[i].copy(); p[2] = max(p[2], table.height + 0.017)
    plans[arm] = robot.plan(robot.grasp_at(robot.top_down(p, yaw), arm=arm), table)
out = robot.run(plans)
st = robot.status(); ups = {}
for arm in ("left", "right"):
    t = st["tcp"][arm].copy(); t[2, 3] += 0.15
    ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t), table)
robot.run(ups, sync=True)
print(robot.holding(arm="left"), robot.holding(arm="right"), len(n))
'''


def test_privileged_two_arm_solve_through_the_kernel(checker, tmp_path):
    d = write(tmp_path, CABLE_TASK, "cable")
    assert scene_command(["check", str(d), "--seeds", "0"]) == 0
    model = ScriptedChat([f"Both ends at once.\n```python\n{SOLUTION}\n```"], then="```python\nfinish('done')\n```")
    res = solve_free_scene(d, model, turns=3, cameras=[], video=False)
    assert res["success"] and res["status"] == "success" and res["privileged"] and res["robot"] == "aloha"
    assert res["env"]["type"] == "mujoco.free_scene" and res["env"]["params"]["privileged_obs"] is True
    assert res["tools"].endswith("PrivilegedFreeTools") and res["turns"] >= 1 and res["errors"] == 0
    system = model.calls[0][0]["content"]
    assert FREE_REFERENCE.splitlines()[0] in system and "two arms" in system and "robot.top_down" in system
    turns = [json.loads(line) for line in (d / "solve" / "turns.jsonl").read_text().splitlines()]
    assert turns[0]["output"].startswith("True True 16")
    assert json.loads((d / "solve.json").read_text())["success"]


def test_cli_check_on_a_free_program(checker, tmp_path, capsys):
    d = write(tmp_path, panda(cube(), "return True"))
    assert scene_command(["check", str(d / "scene.py"), "--seeds", "0"]) == 1
    assert "success_true_at_start" in capsys.readouterr().out
    assert "def demo(sim)" in free_builder_system()


def test_the_showcase_script_runs_end_to_end_with_scripted_models(checker, tmp_path):
    """examples/scenes_free/run_all.py --smoke: build, check, solve, scripted solve and collect through the real code
    path with scripted chat models (no Ollama, no cameras), then the Blender stage's re-simulation (not the render)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "scenes_free" / "run_all.py"
    spec = importlib.util.spec_from_file_location("scenes_free_run_all", path)
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)
    assert run_all.smoke(tmp_path) == 0
    rows = json.loads((tmp_path / "results.json").read_text())
    assert rows[0]["build"]["built"] and rows[0]["solve"]["success"] and rows[0]["robot"] == "aloha"
    assert rows[0]["scripted"]["label"].startswith("scripted solution") and rows[0]["newton"] == run_all.NOT_RUN
    d = tmp_path / "smoke_cable"
    run = json.loads((d / "solve.json").read_text())
    qpos = run_all.replay_qpos(d, run)
    assert len(qpos) == run["steps"] + 1 and (d / "world.xml").is_file()
    shipped = json.loads((path.parent / "results.json").read_text())
    tasks = json.loads((path.parent / "tasks.json").read_text())["tasks"]
    assert [r["task"] for r in shipped] == [t["name"] for t in tasks]
