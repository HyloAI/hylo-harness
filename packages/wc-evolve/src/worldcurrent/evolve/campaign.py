"""Multi-cycle improvement campaigns: config, resume, budgets, manual approval (RFC-0002 §5).

A campaign repeats `run_cycle`: each promoted candidate becomes the next cycle's parent. State lives in the campaign
directory (registry, archive, campaign_state.json, per-cycle runs), so `run_campaign` resumes where it stopped.
With approval="manual", a candidate that passes the gate is parked in pending_approval.json and promoted only by an
explicit `approve()` (e.g. `wc evolve approve DIR`): the default for anything that will run on hardware.
Every change of the active version is published as `active.json` (core.deployment), which is all a deploying
process (the policy server) reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.config import ComponentRef, ExperimentConfig
from worldcurrent.core.contract import Contract
from worldcurrent.core.deployment import ActiveVersion, write_active
from worldcurrent.core.registry import registry as plugins

from .candidate import CandidateRegistry
from .protocol import Archive, CycleReport, ProtocolConfig, run_cycle
from .steering import diagnose_traces

EVOLVE_SCHEMA = "wc.evolve/v0.1"


class EvolveConfig(Contract):
    schema_version: Literal["wc.evolve/v0.1"] = EVOLVE_SCHEMA
    name: str
    experiment: str                                          # path to a wc.experiment config, relative to this file
    experiment_overrides: dict[str, Any] = Field(default_factory=dict)
    proposer: ComponentRef = Field(default_factory=lambda: ComponentRef(type="rule.steering"))
    diagnosis: dict[str, Any] = Field(default_factory=dict)  # kwargs for diagnose_traces
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    cycles: int = Field(default=1, ge=1)
    stop_after_no_progress: int = Field(default=2, ge=1)     # consecutive cycles without a promotion
    budget_s: float | None = None                            # wall clock for the whole campaign
    approval: Literal["auto", "manual"] = "auto"
    out_dir: str = "runs/evolve"

    @classmethod
    def load(cls, path: str | Path) -> "EvolveConfig":
        """A relative `experiment` path is resolved against the campaign file (else the working directory, as
        configs written before 2026-11-01 assumed), so a campaign runs from any directory."""
        import yaml

        cfg = cls.model_validate(yaml.safe_load(Path(path).read_text()))
        exp = Path(cfg.experiment).expanduser()
        if not exp.is_absolute():
            near = Path(path).resolve().parent / exp
            exp = near if near.exists() or not exp.exists() else exp.resolve()
        return cfg.model_copy(update={"experiment": str(exp)})


class CampaignState(Contract):
    name: str
    cycles_done: int = 0
    no_progress: int = 0
    elapsed_s: float = 0.0
    history: list[dict[str, Any]] = Field(default_factory=list)
    stopped: str | None = None


def _paths(cfg: EvolveConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    return {"root": root, "state": root / "campaign_state.json", "pending": root / "pending_approval.json",
            "registry": root / "registry", "archive": root / "archive.jsonl"}


def publish_active(cfg: EvolveConfig) -> ActiveVersion:
    """Write the campaign's active version (resolved agent spec) to `<out_dir>/active.json`."""
    reg = CandidateRegistry(_paths(cfg)["registry"])
    patch = reg.effective_patch() if reg.active else {}
    base = ExperimentConfig.load(cfg.experiment).with_overrides(cfg.experiment_overrides)
    active = ActiveVersion(campaign=cfg.name, version=reg.active, agent=base.with_overrides(patch).agent.model_dump(mode="json"),
                           patch=patch, lineage=tuple(reg.lineage()) if reg.active else ())
    write_active(cfg.out_dir, active)
    return active


def load_state(cfg: EvolveConfig) -> CampaignState:
    p = _paths(cfg)["state"]
    return CampaignState.model_validate_json(p.read_text()) if p.exists() else CampaignState(name=cfg.name)


def run_campaign(cfg: EvolveConfig, *, log: Callable[[str], None] = print,
                 evaluate: Callable[[ExperimentConfig, str, str], Any] | None = None) -> CampaignState:
    paths = _paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    (paths["root"] / "evolve_config.json").write_text(cfg.model_dump_json(indent=2))
    publish_active(cfg)
    state = load_state(cfg)
    if paths["pending"].exists():
        state = state.model_copy(update={"stopped": "waiting for manual approval"})
        log(f"pending approval: {json.loads(paths['pending'].read_text())['candidate_id']} (wc evolve approve {paths['root']})")
        return state
    base = ExperimentConfig.load(cfg.experiment).with_overrides(cfg.experiment_overrides)
    reg, archive = CandidateRegistry(paths["registry"]), Archive(paths["archive"])
    proposer = plugins.build("generators", cfg.proposer.model_dump())
    t_start = time.time() - state.elapsed_s

    while state.cycles_done < cfg.cycles:
        if cfg.budget_s is not None and time.time() - t_start > cfg.budget_s:
            state = state.model_copy(update={"stopped": "budget exhausted"})
            break
        k = state.cycles_done + 1
        cycle_dir = paths["root"] / f"cycle-{k:02d}"

        def _evaluate(c: ExperimentConfig, split: str, label: str):
            t = time.time()
            r = (evaluate or (lambda cc, s, lb: run_experiment(cc, split=s, run_root=cycle_dir / "runs", label=lb)))(c, split, label)
            log(f"[cycle {k}] {label:<34} {split:<10} success={r.metrics['success_mean']:.3f} "
                f"n={r.metrics['n_episodes']} ({time.time() - t:.0f}s)")
            return r

        def _approve(candidate, gate) -> bool:
            if cfg.approval == "auto":
                return True
            paths["pending"].write_text(json.dumps({"candidate_id": candidate.id, "gate": gate, "cycle": k}, default=str, indent=2))
            log(f"[cycle {k}] {candidate.id} passed the gate; waiting for manual approval")
            return False

        log(f"[cycle {k}] parent = {reg.active or 'base config'}")
        rep: CycleReport = run_cycle(base, reg, archive, proposer, _evaluate,
                                     lambda r: diagnose_traces(r, **cfg.diagnosis), protocol=cfg.protocol, approve=_approve)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        (cycle_dir / "cycle_report.json").write_text(rep.model_dump_json(indent=2))
        progressed = rep.promoted is not None
        state = state.model_copy(update={
            "cycles_done": k, "no_progress": 0 if progressed else state.no_progress + 1,
            "elapsed_s": round(time.time() - t_start, 1),
            "history": [*state.history, {"cycle": k, "promoted": rep.promoted, "heldout": rep.heldout,
                                         "verdicts": {e.candidate.id: e.verdict for e in rep.entries},
                                         "elapsed_s": rep.elapsed_s}]})
        paths["state"].write_text(state.model_dump_json(indent=2))
        publish_active(cfg)
        log(f"[cycle {k}] done: promoted={rep.promoted} heldout={json.dumps(rep.heldout)}")
        if paths["pending"].exists():
            state = state.model_copy(update={"stopped": "waiting for manual approval"})
            break
        if state.no_progress >= cfg.stop_after_no_progress:
            state = state.model_copy(update={"stopped": f"no promotion in {state.no_progress} consecutive cycles"})
            break
    else:
        state = state.model_copy(update={"stopped": "all cycles done"})
    paths["state"].write_text(state.model_dump_json(indent=2))
    return state


def approve(cfg: EvolveConfig, *, by: str) -> str:
    paths = _paths(cfg)
    pending = json.loads(paths["pending"].read_text())
    CandidateRegistry(paths["registry"]).promote(pending["candidate_id"], gate=pending["gate"], approved_by=by)
    paths["pending"].unlink()
    publish_active(cfg)
    return pending["candidate_id"]


def status(cfg: EvolveConfig) -> dict[str, Any]:
    paths = _paths(cfg)
    reg = CandidateRegistry(paths["registry"])
    return {"state": load_state(cfg).model_dump(), "active": reg.active, "lineage": reg.lineage() if reg.active else [],
            "effective_patch": reg.effective_patch() if reg.active else {},
            "pending_approval": json.loads(paths["pending"].read_text()) if paths["pending"].exists() else None,
            "archive_size": len(Archive(paths["archive"]).entries())}


def rollback(cfg: EvolveConfig, *, reason: str, by: str) -> str | None:
    restored = CandidateRegistry(_paths(cfg)["registry"]).rollback(reason=reason, by=by)
    publish_active(cfg)
    return restored


def shadow_config(cfg: EvolveConfig, candidate_id: str, *, background: bool = True) -> ExperimentConfig:
    """The active version, executing, with `candidate_id` riding along as a shadow that never acts (agents.shadow).

    The step before trying a candidate on hardware: run this on the robot (or in simulation) and read
    agents.shadow.shadow_report to see how often and how far the candidate would have acted differently."""
    reg = CandidateRegistry(_paths(cfg)["registry"])
    base = ExperimentConfig.load(cfg.experiment).with_overrides(cfg.experiment_overrides)
    active = base.with_overrides(reg.effective_patch())
    patch = reg.get(candidate_id)["candidate"]["patch"]
    outside = [k for k in patch if not k.startswith("agent.")]
    if outside:
        raise ValueError(f"{candidate_id} changes more than the agent ({outside}); it cannot run as a shadow")
    candidate = active.with_overrides(patch)
    return active.with_overrides({"agent": {"type": "shadow", "params": {
        "active": active.agent.model_dump(mode="json"), "shadow": candidate.agent.model_dump(mode="json"),
        "background": background}}})
