"""Skill library: a gated campaign's outcome becomes a verified skill, which the skills agent applies to new scenes.

What it shows
    The RFC-0003 skill path on MuJoCo planar reach, end to end and without an LLM:
    1. a small flow-matching policy (trained from expert demonstrations) deployed with a mis-tuned sampler
       (1 sample at noise scale 3.0): the parent;
    2. an improvement campaign (RFC-0002 protocol: search_val selection, group-sequential gate on the gate
       split, held-out report) decides on steering candidates: those of the built-in `rule.steering` proposer
       plus two noise-scale reductions from a 20-line proposer registered by this script (it stands in for the
       LLM proposer of the recorded maturity campaign, which proposed the same kind of change);
    3. every decided candidate becomes a skill whose status follows the evidence (hylo.evolve.experience):
       promoted -> verified (gate statistics attached), gate failure with effect <= 0 -> refuted,
       promising but not significant -> proposed;
    4. the frozen library is given to the `skills` agent wrapped around the unchanged parent: at reset it matches
       verified skills to the task (env + task tags) and applies the steering patch for that episode only
       (a `skill.applied` trace event);
    5. before / after on fresh scenes: the standard validation split, seeds disjoint from every campaign split,
       paired per (task, seed).

    `hylo skills ARCHIVE --out LIB` recognises only the steering path of a remote agent
    (agent.params.params.steering); this local campaign patches agent.params.steering, so the example calls the
    same functions (`skill_from_entry`, `classify`) with that path.

Command (from the harness repository root)
    uv run python examples/memory/04_skill_library.py [--quick] [--out DIR]

Runtime: about 20 s with --quick, about 1 min in full mode (16 worker threads, CPU).
Requires: hylo-env-mujoco (hylo-bench), hylo-policies, hylo-learn, hylo-evolve.
"""

from __future__ import annotations

import json
import shutil

import yaml

from _common import Timer, paired, parse_args, print_table, rate, reach_config, require_plugin, train_flow_policy, write_result

SLUG = "skill_library"
PARENT_STEERING = {"n_samples": 1, "noise_scale": 3.0}
STEERING_PATH = "agent.params.steering"


def main() -> None:
    args = parse_args(SLUG, __doc__)
    require_plugin("envs", "mujoco.planar_reach")
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry
    from hylo.eval.experiment import run_experiment
    from hylo.evolve.campaign import EvolveConfig, run_campaign
    from hylo.evolve.experience import classify, read_archive, skill_from_entry
    from hylo.memory.skills import SkillLibrary

    registry.register("generators", "examples.noise_scale")(_noise_scale_proposer())
    clock = Timer()
    rsi_states, epochs, fresh = (50, 150, 10) if args.quick else (250, 300, 100)
    art = train_flow_policy(args.out, rsi_states=rsi_states, epochs=epochs, workers=args.workers)
    parent = {"type": "steered", "params": {"sampler": {"type": "policy.flow", "params": {"path": str(art)}},
                                            "steering": PARENT_STEERING}}

    # 1-2. the campaign (config files written next to the results, so `hylo evolve status DIR` works on them)
    camp = args.out / "campaign"
    shutil.rmtree(camp, ignore_errors=True)
    camp.mkdir(parents=True)
    exp = reach_config(parent, rsi_states=rsi_states, workers=args.workers, name="flow-reach-noisy", gate_looks=2)
    (camp / "experiment.yaml").write_text(yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False))
    evolve = {"schema_version": "hylo.evolve/v0.1", "name": "flow-reach-skills", "experiment": "experiment.yaml",
              "proposer": {"type": "examples.noise_scale", "params": {"path": STEERING_PATH}},
              "protocol": {"alpha": 0.05, "gate_looks": 2, "stage1_pairs": min(20, rsi_states // 5)},
              "cycles": 1, "approval": "auto", "out_dir": str(camp)}
    (camp / "evolve.yaml").write_text(yaml.safe_dump(evolve, sort_keys=False))
    state = run_campaign(EvolveConfig.load(camp / "evolve.yaml"), log=lambda m: print(m, flush=True))

    # 3. archive -> skill library (status from the evidence only)
    lib_dir = args.out / "skills"
    shutil.rmtree(lib_dir, ignore_errors=True)
    lib = SkillLibrary(str(lib_dir))
    skills = []
    for entry in read_archive(camp):
        skill = skill_from_entry(entry, task_tags=("reach",), env="mujoco.planar_reach", steering_path=STEERING_PATH)
        if skill is None:
            continue
        status, evidence = classify(entry)
        sid = lib.add(skill)
        (lib.transition(sid, status, evidence) if status != "proposed" else lib.annotate(sid, evidence))
        body = {k: v for k, v in skill.body["steering"].items() if PARENT_STEERING.get(k, _DEFAULTS.get(k)) != v}
        skills.append({"skill": sid, "status": status, "changes": json.dumps(body), "verdict": entry.verdict,
                       "effect": evidence.get("effect"), "p": evidence.get("p_value"), "split": evidence.get("gate_split")})
    n_md = lib.export_skill_md(str(args.out / "SKILL.md"))
    digest = lib.snapshot()

    # 4-5. the unchanged parent with and without the frozen library, on fresh scenes
    fresh_cfg = ExperimentConfig(name="flow-reach-fresh", benchmark={"type": "mujoco/reach-v0", "params": {"validation": fresh}},
                                 env={"type": "mujoco.planar_reach"}, agent=parent, workers=args.workers)
    with_skills = fresh_cfg.with_overrides({"agent": {"type": "skills", "params": {
        "inner": parent, "skills": {"type": "skill_library", "params": {"path": str(lib_dir), "frozen": True}}}}})
    before = run_experiment(fresh_cfg, split="validation", run_root=args.out / "fresh", label="parent")
    after = run_experiment(with_skills, split="validation", run_root=args.out / "fresh", label="parent+skills")
    applied = _applied(after.run_dir)
    cmp_ = paired(before.per_key(), after.per_key())

    result = {"example": SLUG, "quick": args.quick, "rsi_states": rsi_states, "parent_steering": PARENT_STEERING,
              "campaign": {"dir": str(camp), "stopped": state.stopped, "history": state.history},
              "skills": skills, "library_digest": digest, "skill_md_exported": n_md,
              "fresh_scenes": {"split": "validation (standard protocol)", "parent": rate(before.per_key().values()),
                               "parent_plus_skills": rate(after.per_key().values()), "paired": cmp_,
                               "skill_applied_events": applied,
                               "steps_mean": {"parent": before.metrics["steps_mean"], "parent_plus_skills": after.metrics["steps_mean"]}},
              "runtime_s": clock()}
    write_result(args.out, result)
    heldout = (state.history[-1].get("heldout") if state.history else None) or {}
    print_table(skills, title=f"skill library {digest} (from the campaign archive; {n_md} verified skill(s) exported as SKILL.md)")
    if heldout:
        print(f"campaign held-out report: {json.dumps(heldout)}")
    print_table([{"agent": "parent (noise 3.0)", **_row(result["fresh_scenes"]["parent"]), "steps": before.metrics["steps_mean"]},
                 {"agent": "parent + skills agent", **_row(result["fresh_scenes"]["parent_plus_skills"]),
                  "steps": after.metrics["steps_mean"]}],
                title=f"fresh scenes ({cmp_['n_pairs']} pairs): diff {cmp_['diff']:+.3f}, +{cmp_['wins']} / -{cmp_['losses']}, "
                      f"p = {cmp_['p_two_sided']}; skill.applied in {applied} episodes")
    print(f"\nresult: {args.out / 'result.json'}  ({clock()} s)")


def _noise_scale_proposer():
    """rule.steering's candidates plus noise-scale reductions of the current steering config (full config patches,
    so each one is expressible as a steering skill)."""
    from hylo.evolve.candidate import Candidate
    from hylo.evolve.steering import SteeringRuleProposer, _get

    class NoiseScaleProposer(SteeringRuleProposer):
        name = "examples.noise_scale@0.1"

        def __init__(self, path: str = STEERING_PATH, factors: tuple[float, ...] = (0.5, 0.33), **kw) -> None:
            super().__init__(path=path, **kw)
            self.factors = tuple(factors)

        def propose(self, diagnosis, cfg, parent_id, archive):
            out = super().propose(diagnosis, cfg, parent_id, archive)
            current = dict(_get(cfg.model_dump(mode="json"), self.path) or {})
            seen = {json.dumps(e.candidate.patch, sort_keys=True) for e in archive}
            for f in self.factors:
                ns = round(float(current.get("noise_scale", 1.0)) * f, 3)
                hyp = f"sampling noise {current.get('noise_scale', 1.0)} is too wide for this policy; {ns} keeps samples near the data"
                c = Candidate(patch={self.path: {**current, "noise_scale": ns}}, rationale=hyp, parent_id=parent_id,
                              generator=self.name, evidence={"buckets": diagnosis.get("buckets", {})})
                if json.dumps(c.patch, sort_keys=True) not in seen:
                    out.append((c, hyp))
            return out

    return NoiseScaleProposer


_DEFAULTS = {"n_samples": 1, "selector": "first", "noise_scale": 1.0, "execution_steps": None, "seed_noise": True,
             "rules": [], "compare_steps": 10, "still_eps": 0.0015, "gripper_closed_below": 0.06, "eef_key": "eef_pos",
             "gripper_key": "gripper_qpos"}


def _row(r: dict) -> dict:
    return {"success": r["success"], "95% CI": f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]", "n": r["n"]}


def _applied(run_dir) -> int:
    from hylo.core.trace import TraceStore

    store = TraceStore(run_dir)
    return sum(1 for eid in store.episode_ids() if store.read(eid).first("skill.applied") is not None)


if __name__ == "__main__":
    main()
