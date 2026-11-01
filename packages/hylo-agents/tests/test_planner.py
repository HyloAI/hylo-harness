import numpy as np

from hylo.agents.baseline import PControllerAgent
from hylo.agents.planner import Plan, Subgoal, SubgoalAgent
from hylo.core.kernel import EpisodeRunner
from hylo.core.task import TaskInstance
from hylo.core.trace import TraceStore
from hylo.testing.point_reach import PointReachEnv, reach_tasks


class ScriptedPlanner:
    name = "test.planner"

    def __init__(self):
        self.done_calls = 0

    def plan(self, instruction, observation):
        return Plan(subgoals=[Subgoal(instruction="first half", max_steps=5), Subgoal(instruction="second half", max_steps=500)],
                    rationale="two halves")

    def done(self, subgoal, observation):
        self.done_calls += 1
        return False, "not yet", {}


class Recorder(PControllerAgent):
    def __init__(self):
        super().__init__(kp=2.0, horizon=3, execution_steps=3)
        self.instructions = []

    def act(self, observation):
        self.instructions.append(observation.instruction)
        return super().act(observation)


def test_subgoals_feed_the_inner_agent_and_switch_on_budget(tmp_path):
    inner, planner = Recorder(), ScriptedPlanner()
    agent = SubgoalAgent(inner, planner, check_every=1)
    store = TraceStore(tmp_path)
    task = next(t for t in reach_tasks() if t.id.endswith("reach_fixed"))
    EpisodeRunner(PointReachEnv(), agent, trace=store).run(TaskInstance(spec=task, seed=1), episode_id="ep")
    assert inner.instructions[0] == "first half" and "second half" in inner.instructions
    decisions = [e["payload"] for e in store.read("ep").of_type("agent.decision")]
    assert decisions[0]["kind"] == "plan" and decisions[1]["kind"] == "subgoal_switch"
    assert "budget" in decisions[1]["rationale"] and planner.done_calls >= 1
    assert "planner" in agent.spec().components
