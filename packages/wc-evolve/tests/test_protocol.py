from worldcurrent.eval.experiment import run_experiment
from worldcurrent.eval.scoring import one_sided_exact_mcnemar
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.evolve.candidate import Candidate, CandidateRegistry
from worldcurrent.evolve.protocol import Archive, ProtocolConfig, holm, rsi_splits, run_cycle
from worldcurrent.evolve.steering import diagnose_traces


def test_rsi_splits_are_disjoint_and_replicates_keep_init_states():
    s = rsi_splits(50, (20, 30, 40), replicates=2)
    all_seeds = [x for v in s.values() for x in v]
    assert len(all_seeds) == len(set(all_seeds)) == 100
    assert {x % 50 for x in s["gate"]} == set(range(30, 40))


def test_holm_is_stricter_than_unadjusted():
    assert holm([0.03, 0.04], 0.05) == [False, False]      # 0.03 > 0.05 / 2: unadjusted tests would pass both
    assert holm([0.01, 0.04], 0.05) == [True, True]


class GainProposer:
    name = "test.gain"

    def propose(self, diagnosis, cfg, parent_id, archive):
        kp = cfg.agent.params["kp"]
        return [(Candidate(patch={"agent.params.kp": kp * f}, rationale="", parent_id=parent_id, generator=self.name),
                 f"kp x{f} reduces timeouts") for f in (1.0, 4.0)] + \
               [(Candidate(patch={"safety.max_delta": 9.0}, rationale="", parent_id=parent_id, generator=self.name),
                 "removing the rate limit is faster")]


def test_cycle_selects_on_search_val_confirms_on_gate_and_reports_heldout(tmp_path):
    cfg = ExperimentConfig(name="p", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"], "protocol": "rsi"}},
                           env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})
    reg, arc = CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "archive.jsonl")
    seen_splits = []

    def evaluate(c, split, label):
        seen_splits.append(split)
        return run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)

    rep = run_cycle(cfg, reg, arc, GainProposer(), evaluate, diagnose_traces, protocol=ProtocolConfig())
    verdicts = {e.candidate.patch.get("agent.params.kp", "safety"): e.verdict for e in rep.entries}
    assert verdicts["safety"] == "rejected"                       # protected scope, never evaluated
    assert verdicts[2.0] == "promoted" and rep.promoted is not None
    assert rep.heldout["used_for_selection"] is False
    assert seen_splits[:2] == ["search", "search_val"] and "gate" in seen_splits and seen_splits[-1] == "heldout"
    assert len(arc.entries()) == 3 and all(e.hypothesis for e in arc.entries())
    # a proposal identical to an archived candidate is rejected before any rollout
    class Repeat:
        name = "test.repeat"

        def propose(self, diagnosis, cfg, parent_id, archive):
            return [(Candidate(patch={"agent.params.kp": 2.0}, rationale="", parent_id=parent_id, generator=self.name), "again")]

    n_evals = len(seen_splits)
    rep2 = run_cycle(cfg, reg, arc, Repeat(), evaluate, diagnose_traces)
    assert rep2.entries[0].verdict == "rejected" and "duplicate" in rep2.entries[0].evidence["stage0"]
    assert len(seen_splits) - n_evals == 2                         # only the parent's search + search_val runs


def test_mcnemar_sanity():
    assert one_sided_exact_mcnemar(6, 0) < 0.05


def test_alpha_spending_sums_to_alpha_and_is_strict_early():
    from worldcurrent.evolve.protocol import spend

    levels = [spend(0.05, k, 3, 3.0) for k in (1, 2, 3)]
    assert abs(sum(levels) - 0.05) < 1e-12 and levels[0] < levels[1] < levels[2]


def test_group_sequential_gate_stops_at_the_first_conclusive_look(tmp_path):
    cfg = ExperimentConfig(name="p", benchmark={"type": "testing/reach-v0",
                                                "params": {"tags": ["reach"], "protocol": "rsi", "gate_looks": 3}},
                           env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 0.5}})
    splits = []

    def evaluate(c, split, label):
        splits.append(split)
        return run_experiment(c, split=split, run_root=tmp_path / "runs", label=label)

    rep = run_cycle(cfg, CandidateRegistry(tmp_path / "reg"), Archive(tmp_path / "a.jsonl"), GainProposer(), evaluate,
                    diagnose_traces, protocol=ProtocolConfig(gate_looks=3))
    promoted = next(e for e in rep.entries if e.verdict == "promoted")
    looks = [s for s in promoted.stages if s.stage == 3]
    assert len(looks) == 1 and "look 1/3" in looks[0].notes      # a large effect passes at the first look
    assert "gate_r1" not in splits                                # later looks were never spent
