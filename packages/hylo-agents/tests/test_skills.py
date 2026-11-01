import numpy as np

from hylo.agents.planner import IdentityPlanner, SubgoalAgent
from hylo.agents.skills import SkillAgent
from hylo.agents.steering import SteeredAgent
from hylo.core.agent import AgentSpec, EpisodeContext, EpisodeOutcome
from hylo.core.experience import Skill, SkillTrigger
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Observation


class Sampler:
    def __init__(self):
        self.instructions = []

    def spec(self):
        return AgentSpec(name="fake", version="0", action_dim=7, components={"policy": "fake@0"})

    def reset(self, ctx):
        pass

    def default_execution_steps(self):
        return 10

    def sample(self, observation, seeds, noise_scale=1.0):
        self.instructions.append(observation.instruction)
        return np.zeros((len(seeds or [0]), 20, 7))


class Library:
    def __init__(self, skills):
        self.skills = skills

    def match(self, instruction, *, task_tags=(), env=None, kinds=None):
        return [(s, {"a": "soup", "b": "sauce"}) for s in self.skills if not kinds or s.kind in kinds]

    def all(self, status=None):
        return self.skills

    def snapshot(self):
        return "abc"


SKILLS = [Skill(name="steer", version="1", description="exec 25", kind="steering", status="verified",
                trigger=SkillTrigger(task_tags=("libero_10",)), body={"steering": {"execution_steps": 25}}),
          Skill(name="both", version="1", description="two picks", kind="plan", status="verified",
                trigger=SkillTrigger(), body={"subgoals": ["pick the {a}", "pick the {b}"]})]


def test_skills_apply_for_one_episode_and_are_undone():
    sampler, events = Sampler(), []
    steered = SteeredAgent(sampler, {"n_samples": 1})
    agent = SkillAgent(SubgoalAgent(steered, IdentityPlanner()), Library(SKILLS))
    task = TaskInstance(spec=TaskSpec(id="t", version="1", instruction="put both", tags=("libero_10",)), seed=0)
    ctx = EpisodeContext(episode_id="e", task=task, env_spec=None, emit=lambda t, p: events.append((t, p)), seed=0)
    agent.reset(ctx)
    assert steered.cfg.execution_steps == 25
    chunk = agent.act(Observation(state={}, instruction="put both"))
    assert chunk.execution_steps == 20                                   # the skill set 25; the chunk has 20 steps
    assert sampler.instructions[-1] == "pick the soup"                   # the plan skill's first subgoal
    assert [p["skill"] for t, p in events if t == "skill.applied"] == ["steer@1", "both@1"]
    agent.on_episode_end(EpisodeOutcome(episode_id="e", task=task, status="success", success=True, steps=1,
                                        termination_reason="x"))
    assert steered.cfg.execution_steps is None and isinstance(agent.inner.planner, IdentityPlanner)
    assert "library(abc,verified=2" in agent.spec().components["skills"]
