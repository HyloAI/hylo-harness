import numpy as np

from hylo.testing.conformance import check_agent, check_environment
from hylo.testing.point_reach import PointReachEnv, reach_tasks


class BrokenEnv(PointReachEnv):
    def reset(self, task):
        super().reset(task)
        self.pos = self.pos + np.random.default_rng().normal(size=2)   # unseeded despite declaring seedable
        return self._observe()


def test_the_test_double_passes_and_a_broken_env_fails():
    task = reach_tasks()[1]
    good = check_environment(lambda: PointReachEnv(), task)
    assert good.ok, good.summary()
    bad = check_environment(lambda: BrokenEnv(), task)
    assert not bad.ok and any(c.name.startswith("reset is deterministic") and not c.passed for c in bad.checks)


def test_baseline_agent_conforms():
    from hylo.agents.baseline import PControllerAgent  # noqa: F401  (hylo-agents is a test-time dependency)

    rep = check_agent(PControllerAgent(kp=2.0), PointReachEnv(), reach_tasks()[0])
    assert rep.ok, rep.summary()
