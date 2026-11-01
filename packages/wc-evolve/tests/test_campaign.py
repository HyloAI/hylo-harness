import yaml

from worldcurrent.evolve.campaign import EvolveConfig, approve, load_state, rollback, run_campaign, status


def write_experiment(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump({
        "schema_version": "wc.experiment/v0.1", "name": "double",
        "benchmark": {"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
        "env": {"type": "testing.point_reach"}, "agent": {"type": "testing.p_controller", "params": {"kp": 0.5}}}))
    return p


def cfg(tmp_path, **kw):
    return EvolveConfig(name="t", experiment=str(write_experiment(tmp_path)), out_dir=str(tmp_path / "camp"),
                        proposer={"type": "tests.gain"}, **kw)


def test_manual_approval_parks_then_resumes(tmp_path, monkeypatch):
    from worldcurrent.core.registry import registry
    from worldcurrent.evolve.candidate import Candidate

    class Gain:
        name = "tests.gain"

        def propose(self, diagnosis, c, parent_id, archive):
            kp = c.agent.params["kp"]
            return [(Candidate(patch={"agent.params.kp": kp * 4}, rationale="", parent_id=parent_id, generator=self.name),
                     "a higher gain reaches the goal before the time limit")]

    registry.register("generators", "tests.gain")(Gain)
    c = cfg(tmp_path, cycles=2, approval="manual", stop_after_no_progress=1)
    s1 = run_campaign(c, log=lambda m: None)
    assert s1.cycles_done == 1 and "manual approval" in s1.stopped
    assert status(c)["pending_approval"] is not None and status(c)["active"] is None
    promoted = approve(c, by="tester")
    assert status(c)["active"] == promoted
    s2 = run_campaign(c, log=lambda m: None)                      # resumes at cycle 2 from the promoted parent
    assert s2.cycles_done == 2 and load_state(c).history[1]["cycle"] == 2
    assert rollback(c, reason="field report", by="tester") is None
