"""Optional phase-guided program evolution, independently implemented from a design study.

Plan observable phases -> develop and repair solve(robot) -> seal evidence-linked repair
notes -> return a code candidate to Hylo's existing gate. No upstream HarnessPAI code is
included, no weight training is performed, and development success never activates code.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from hylo.core.artifact import Provenance, SourceRef
from hylo.core.registry import registry
from hylo.reasoners.llm.chat import ChatError, Usage, build_chat

from .generator import ResearchProposer
from .phase_memory import PhasePlans, PhaseRepairMemory, RepairNotes, validate_notes
from .research import ResearchConfig, ResearchSession, dev_cases
from .skills import read_program, write_program
from .turns import Budget

PLAN = """Plan observable phases for a robot control program. Return JSON only:
{"phases": [{"name": "...", "goal": "...", "completion": "...", "fallback": "...", "step_budget": 100}]}.
Use 1-12 phases. Completion checks must use deployment observations or public tool results,
never hidden simulator state, task IDs or evaluator success. Propose bounded handovers and
fallbacks. Use a frozen action-model tool only if the supplied toolkit actually provides
one; otherwise use its available control tools. Retrieved repairs are hypotheses with
limited provenance, not universal instructions. Do not infer phase competence from a
whole-task score. This is a plan for code generation, not an executable state machine."""

DISTILL = """Summarize reusable repairs from the supplied development submissions and the
selected program. Return JSON only: {"repairs": [{"phase": "a supplied phase name",
"trigger": "observable failure pattern", "repair": "what changed and why it may help",
"limitation": "where it may not apply", "failure_refs": ["submission-1/case-0"],
"success_refs": ["submission-2/case-0"]}]}.
Use at most 8 notes, or an empty list when no repair is supported. Each note must cite a
failed case and a successful replay of that same case in the selected program. Cite only
provided evidence IDs. Do not claim cross-task transfer or independent phase validation.
Do not include executable code or hard-coded simulator coordinates in the repair text."""


class HarnessEvolutionProposer:
    """A selectable strategy, distinct from the existing research.llm proposer.

    `budget.turns` includes planning and optional distillation calls. Session submissions,
    resets and total model token/cost budgets remain bounded. Token/cost limits are checked
    between calls, as in ResearchSession; one model call can exceed the remaining estimate.
    """

    def __init__(self, model="ollama:qwen3.5:35b-a3b", *, dev: int = 6, budget=None,
                 out_root="runs/harness-evolution", repair_campaigns=(), tools=None,
                 family=None, program_params=None, reply_tokens: int = 2048,
                 block_timeout_s: float = 60, program_timeout_s: float = 300):
        self.model = build_chat(model)
        self.budget = Budget.parse(budget) if budget is not None else Budget(turns=40, submits=6, resets=12)
        if self.budget.turns < 3 or reply_tokens < 1:
            raise ValueError("reserve at least three model calls and a positive reply token budget")
        self.config = ResearchConfig(dev=dev, tools=tools, family=family, program_params=dict(program_params or {}),
                                     reply_tokens=reply_tokens, block_timeout_s=block_timeout_s,
                                     program_timeout_s=program_timeout_s)
        self.memory = PhaseRepairMemory(repair_campaigns)
        self.root, self.reply_tokens = Path(out_root), reply_tokens
        self.name = f"harness.program[{self.model.name}]@0.1"
        self.last = {}

    def _json(self, system, payload, schema, out, label):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, default=str)}]
        reply = self.model.chat(messages, max_tokens=self.reply_tokens)
        self.usage.add(reply)
        (out / f"{label}.json").write_text(json.dumps({"messages": messages, "reply": reply.text,
                                                     "usage": reply.accounting()}, indent=2) + "\n")
        text = reply.text.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        return schema.model_validate(json.loads(text))

    def _remaining(self):
        tokens = None if self.budget.tokens is None else self.budget.tokens - self.usage.tokens
        cost = None if self.budget.cost_usd is None else self.budget.cost_usd - self.usage.cost_usd
        if (tokens is not None and tokens <= 0) or (cost is not None and cost <= 0):
            raise ValueError("model budget exhausted")
        return tokens, cost

    def propose(self, diagnosis, cfg, parent_id, archive):
        out = self.root / f"proposal-{uuid.uuid4().hex[:12]}"
        out.mkdir(parents=True, exist_ok=False)
        self.usage = Usage()
        self.last = {"directory": str(out), "status": "planning"}
        try:
            return self._propose(diagnosis, cfg, parent_id, out)
        except (ChatError, ValueError) as exc:
            self.last.update(status="rejected", error=str(exc))
            return []
        finally:
            self.last["usage"] = self.usage.as_dict()
            (out / "strategy.json").write_text(json.dumps(self.last, indent=2, default=str) + "\n")

    def _propose(self, diagnosis, cfg, parent_id, out):
        bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
        cases = dev_cases(bench, self.config.dev)
        if not cases:
            raise ValueError("no development cases")
        # Only read a diagnosis after checking its manifest; never forward arbitrary caller metadata.
        failures = {}
        if diagnosis.get("run_dir"):
            path = Path(diagnosis["run_dir"])
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("benchmark", {}).get("split") != "search":
                raise ValueError("program evolution diagnosis requires the search split")
            failures = {k: diagnosis[k] for k in ("buckets", "bucket_rate") if k in diagnosis}
        instructions = list(dict.fromkeys(case.spec.instruction for case in cases))
        recalled = self.memory.retrieve(" ".join(instructions))
        # Source locations and gate records are provenance, not model input.
        notes = [{k: r[k] for k in ("id", "phase", "trigger", "repair", "limitation")} for r in recalled]
        parent = ResearchProposer._parent_program(cfg)
        self._remaining()
        phases = self._json(PLAN, {"instructions": instructions, "failures": failures,
                                  "parent_program": parent, "repair_notes": notes}, PhasePlans, out, "plan")
        if len({p.name for p in phases.phases}) != len(phases.phases):
            raise ValueError("phase names must be unique")
        # Once phases exist, recall by their labels and goals as well as the task wording.
        # Keep every source shown during planning in provenance even if not selected again.
        used = {row["id"]: row for row in recalled}
        scoped = {}
        skipped = list(self.memory.skipped)
        for phase in phases.phases:
            for row in self.memory.retrieve(phase.goal, phase=phase.name):
                scoped[row["id"]] = row
            skipped.extend(self.memory.skipped)
        selected_notes = sorted(scoped.values(), key=lambda row: (-row["score"], row["id"]))[:6]
        for row in selected_notes:
            used[row["id"]] = row
        recalled = list(used.values())
        notes = [{k: r[k] for k in ("id", "phase", "trigger", "repair", "limitation")}
                 for r in selected_notes] or notes
        tokens, cost = self._remaining()
        tools = self.config.tools
        if tools is None and cfg.agent.type in ("code.llm", "code.program"):
            tools = cfg.agent.params.get("tools")
        guidance = ("Implement the following phase plan using the actual toolkit. Completed program revisions "
                    "are automatically evaluated on all development cases within the submission budget. "
                    "Explicit submit() is also available and is not duplicated automatically. "
                    "When a parent is already defined, "
                    "your first code block must call submit() to establish its development failures. Otherwise "
                    "submit an initial program before improving it. Inspect failed development cases, repair and "
                    "resubmit. Match the provided toolkit signatures exactly, including required arm arguments; "
                    "inspect perception results before indexing them. Code must implement bounded "
                    "loops and observation-based phase completion checks; the plan alone does not enforce them. "
                    "Evaluate the full submitted program, not just a process exit.\n" + json.dumps({
                        "phases": phases.model_dump()["phases"], "repair_notes": notes}, indent=2))
        rc = self.config.model_copy(update={"tools": tools, "guidance": guidance, "parent_program": parent,
            "auto_submit": True,
            "min_dev_successes": len(cases), "budget": self.budget.model_copy(update={
                "turns": self.budget.turns - 2, "tokens": tokens, "cost_usd": cost})})
        session = ResearchSession(cfg, self.model, out / "development", config=rc)
        summary = session.run()
        self.usage.calls += session.usage.calls
        self.usage.tokens_in += session.usage.tokens_in
        self.usage.tokens_out += session.usage.tokens_out
        self.usage.cost_usd += session.usage.cost_usd
        self.usage.cost_known = self.usage.cost_known and session.usage.cost_known
        self.usage.latency_s += session.usage.latency_s
        self.last.update(session=summary, retrieved=[r["id"] for r in recalled], skipped=skipped)
        candidate = session.candidate(parent_id, self.name)
        if candidate is None:
            self.last["status"] = "no_complete_development_solution"
            return []
        selected = session.frozen["submission"]
        evidence = {f"submission-{sub['submission']}/case-{r['case']}": {
            "key": r["key"], "success": r["success"], "selected": sub["submission"] == selected,
            "reason": r["reason"], "error": r["error"], "output": r["output"]}
            for sub in session.submissions for r in sub["results"]}
        source, old_manifest, meta = read_program(session.frozen["artifact"])
        repairs = RepairNotes()
        if any(not r["success"] for r in evidence.values()):
            try:
                self._remaining()
                repairs = self._json(DISTILL, {"phases": phases.model_dump()["phases"],
                    "evidence": evidence, "selected_program": source}, RepairNotes, out, "distillation")
                validate_notes(repairs, phases, evidence)
            except (ChatError, ValueError) as exc:
                # A malformed note must not poison a valid program or enter future retrieval.
                repairs = RepairNotes()
                self.last["distillation_rejected"] = str(exc)
        pack = {"phases": phases.model_dump()["phases"], "repairs": repairs.model_dump()["repairs"],
                "evidence": evidence}
        sources = [SourceRef(kind="artifact", id=old_manifest.digest, path=session.frozen["artifact"], split="search")]
        sources += [SourceRef(kind="run", id=sub["run_id"], path=sub["dir"], split="search", n_episodes=sub["cases"])
                    for sub in session.submissions]
        sources += [SourceRef(kind="artifact", id=r["program_digest"], path=r["program_artifact"], split="search")
                    for r in recalled]
        parent_artifact = cfg.agent.params.get("artifact") if cfg.agent.type == "code.program" else None
        if parent_artifact:
            from hylo.core.store import Store, is_ref
            parent_path = Store().path(parent_artifact) if is_ref(parent_artifact) else parent_artifact
            _, parent_manifest, _ = read_program(parent_path)
            sources.append(SourceRef(kind="artifact", id=parent_manifest.digest, path=str(parent_path), split="search"))
        artifact = (out / "program").resolve()
        manifest = write_program(artifact, source, meta={**meta, "phase_repairs": pack},
            provenance=Provenance(builder="hylo.harness.program@0.1", sources=sources,
                                  params={"model": self.model.name, "retrieved": [r["id"] for r in recalled]}),
            metrics=old_manifest.metrics)
        patch = dict(candidate.patch)
        patch["agent.params"] = {**patch["agent.params"], "artifact": str(artifact)}
        candidate = candidate.model_copy(update={"patch": patch, "evidence": {**candidate.evidence,
            "artifact": {"id": str(artifact), "path": str(artifact), "kind": "program", "role": "behaviour",
                         "digest": manifest.digest, "splits": ["search"]},
            "phase_count": len(phases.phases), "repair_count": len(repairs.repairs), "strategy": str(out)}})
        self.last.update(status="candidate", candidate=candidate.model_dump(mode="json"))
        return [(candidate, candidate.rationale)]
