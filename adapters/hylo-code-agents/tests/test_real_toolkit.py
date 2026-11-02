"""code.program and code.llm with the real hylo.manip toolkit, through the kernel and the SafetyShield.

The environment wraps hylo-manip's test kit (a 7-DoF test arm in a MuJoCo scene, kinematic, ray-cast RGB-D and
segmentation): the toolkit sees real observations (cameras, depth, segments) and plans real joint trajectories."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.codeagents.agents import CodeLLMAgent, CodeProgramAgent  # noqa: E402
from hylo.core.kernel import EpisodeRunner  # noqa: E402
from hylo.core.task import TaskInstance, TaskSpec  # noqa: E402
from hylo.core.trace import TraceStore  # noqa: E402
from hylo.core.types import (ActionSemantics, BoxSpace, EnvSpec, ObservationSpec, StepResult)  # noqa: E402
from hylo.reasoners.llm.chat import ScriptedChat  # noqa: E402

KIT = Path(__file__).resolve().parents[3] / "packages" / "hylo-manip" / "tests" / "manip_testkit.py"
if not KIT.exists():
    pytest.skip("hylo-manip test kit not found", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("hylo_manip_testkit_for_codeagents", KIT)
tk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tk
_spec.loader.exec_module(tk)

RED, BLUE = (0.9, 0.1, 0.1, 1), (0.1, 0.2, 0.9, 1)
TARGET = (0.55, 0.2)


class KitEnv:
    """An Environment over the test kit's StubExecutor: one action per step, success = red block at TARGET."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.robot = tk.arm_spec(tk.write_arm(directory), cameras=("front", "side", "back"))

    def spec(self) -> EnvSpec:
        arm = self.robot.arms[0]
        cams = {c: (96, 128, 3) for c in ("front", "side", "back")}
        return EnvSpec(name="testing.manip_kit", version="0.1",
                       observation=ObservationSpec(state={"q.arm": (7,), "hand.arm": (1,)}, cameras=cams),
                       action=BoxSpace(shape=(8,), low=(*arm.lower, 0.0), high=(*arm.upper, 1.0),
                                       semantics=ActionSemantics(control_mode="joint_pos", frame="base")),
                       control_hz=20.0, robot=self.robot)

    def reset(self, task: TaskInstance):
        rng = np.random.default_rng(task.seed)
        objects = {"red_block": ((0.45, 0.05 + rng.uniform(0, 0.05), 0.025), (0.025, 0.02, 0.025), RED, 0.3),
                   "blue_block": ((0.42, -0.12, 0.04), (0.03, 0.03, 0.04), BLUE, 0.0)}
        self.ex = tk.StubExecutor(self.dir, objects, self.robot, hw=(96, 128))
        return self._obs(task.spec.instruction)

    def _obs(self, instruction=None):
        obs = self.ex.observation()
        self.instruction = instruction or self.instruction
        return obs.model_copy(update={"instruction": self.instruction})

    def step(self, action):
        self.ex.run(np.asarray(action.data, dtype=float)[None])
        T = self.ex.pose("red_block")
        ok = self.ex.attached is None and bool(np.allclose(T[:2, 3], TARGET, atol=0.02))
        return StepResult(observation=self._obs(), reward=float(ok), terminated=False, truncated=False, success=ok)

    def close(self) -> None:
        pass


PROGRAM = f'''
import numpy as np

def solve(robot):
    view = robot.look()
    table = robot.table(view)
    det = robot.objects(view, robot.find(view, "red block"), table)[0]
    box = robot.box(det.points, table)
    grasp = robot.plan(robot.grasps("arm", det.points, table, scene=view.points), table, view.points)
    if grasp is None:
        print("no grasp:", robot.why())
        return
    res = robot.run(grasp)
    up = res["tcp"]["arm"].copy()
    up[2, 3] += 0.12
    robot.run(robot.plan(robot.Goal("lift", "arm", tcp=up), table, view.points))
    obj_in_tcp = np.linalg.inv(grasp.goal.tcp) @ box.pose
    obj_points = (np.linalg.inv(box.pose) @ np.c_[det.points, np.ones(len(det.points))].T).T[:, :3]
    place = robot.place_at("arm", obj_in_tcp, obj_points, {TARGET!r}, yaw=0.0, table=table, hover=0.005)
    plan = robot.plan(place, table, robot.look().points)
    if plan is None:
        print("no place:", robot.why())
        return
    robot.run(plan)
    print("holding after place:", robot.holding("arm"))
'''


def task(max_steps=600):
    return TaskInstance(spec=TaskSpec(id="testing/kit_place", version="1", max_steps=max_steps,
                                      instruction="put the red block at the marked spot"), seed=0)


def test_code_program_with_the_real_toolkit(tmp_path):
    agent = CodeProgramAgent(source=PROGRAM, timeout_s=120)
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(KitEnv(tmp_path), agent, trace=store, image_every=0).run(task())
    out = agent.last
    assert out["error"] is None and "holding after place: False" in out["output"], out
    assert rec.outcome.status == "success" and rec.outcome.termination_reason == "agent_done: solve returned"
    tr = store.read(rec.outcome.episode_id)
    assert sum(1 for _ in tr.of_type("safety.review")) == rec.outcome.steps > 50
    assert not any(e["payload"]["modified"] for e in tr.of_type("safety.review"))


def test_code_llm_with_the_real_toolkit_and_its_api_reference(tmp_path):
    model = ScriptedChat([
        "Look first.\n```python\nview = robot.look()\nrobot.show(view, 'cameras')\nprint(sorted(view.cams))\n```",
        f"Now the program.\n```python\n{PROGRAM}\nsolve(robot)\nfinish()\n```",
    ])
    agent = CodeLLMAgent(model=model, max_turns=4, block_timeout_s=120)
    store = TraceStore(tmp_path / "run")
    rec = EpisodeRunner(KitEnv(tmp_path), agent, trace=store, image_every=0).run(task())
    tr = store.read(rec.outcome.episode_id)
    turns = [e["payload"] for e in tr.of_type("agent.code.turn")]
    assert turns[0]["output"].strip() == "['back', 'front', 'side']" and turns[0]["images"][0]["image"].ndim == 3
    assert rec.outcome.status == "success", turns[-1]
    system = model.calls[0][0]["content"]
    assert "robot.why(" in system and "robot.place_at(" in system             # hylo.manip.api_reference
