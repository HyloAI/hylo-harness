"""`research.llm`: a research session as an improvement-campaign generator (RFC-0006 part 5).

Each proposal round runs one research session on the parent's experiment (development cases of its search split)
and proposes the frozen program as one code candidate; the campaign's cascade, gate, held-out report and approval
decide the rest. The session's model never sees search_val, gate or held-out outcomes.

    proposer: {type: research.llm, params: {model: {backend: ollama, model: qwen3.5:35b-a3b, vision: true},
               dev: 6, budget: {turns: 40, submits: 6, resets: 30, cost_usd: 2.0}, out_root: runs/research}}

When the parent is a `code.program` (show_parent, default), the session is shown the parent's program as the one to
improve: it may start from it, and the gate still decides against the parent itself.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from hylo.core.config import ExperimentConfig
from hylo.core.improvement import ArchiveEntry, Candidate
from hylo.reasoners.llm.chat import ChatModel, build_chat

from .research import ResearchConfig, ResearchSession
from .turns import Budget

CODE_AGENTS = ("code.llm", "code.program")


class ResearchProposer:
    def __init__(self, model: str | Mapping[str, Any] | ChatModel = "ollama:qwen3.5:35b-a3b", dev: int = 6,
                 budget: str | Mapping[str, Any] | None = None, out_root: str = "runs/research",
                 tools: str | Mapping[str, Any] | None = None, block_timeout_s: float = 60.0,
                 program_timeout_s: float = 300.0, min_dev_successes: int = 1, family: str | None = None,
                 program_params: Mapping[str, Any] | None = None, demonstration: str | None = None,
                 skills: str | Mapping[str, Any] | None = None, reply_tokens: int | None = None,
                 show_parent: bool = True) -> None:
        self.model = build_chat(model)
        self.rc = ResearchConfig(dev=dev, budget=Budget.parse(budget) if budget is not None else
                                 Budget(turns=40, resets=30, submits=6), tools=tools, block_timeout_s=block_timeout_s,
                                 program_timeout_s=program_timeout_s, min_dev_successes=min_dev_successes,
                                 family=family, program_params=dict(program_params or {}), demonstration=demonstration,
                                 skills=skills, reply_tokens=reply_tokens)
        self.out_root, self.show_parent = Path(out_root), show_parent
        self.name = f"research.llm[{self.model.name}]@0.1"
        self.last: dict[str, Any] = {}

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        rc = self.rc
        if rc.tools is None and cfg.agent.type in CODE_AGENTS and cfg.agent.params.get("tools") is not None:
            rc = rc.model_copy(update={"tools": cfg.agent.params["tools"]})   # the parent's toolkit
        parent = self._parent_program(cfg) if self.show_parent else None
        if parent:                                       # the session improves the parent's program
            rc = rc.model_copy(update={"parent_program": parent})
        n = sum(1 for e in archive if e.candidate.generator == self.name)
        tag = f"{parent_id or 'base'}-r{n}"
        out = self.out_root / tag
        if out.exists():
            out = self.out_root / f"{tag}-{uuid.uuid4().hex[:6]}"
        session = ResearchSession(cfg, self.model, out, config=rc, name=f"{cfg.name}-{tag}")
        self.last = session.run()
        cand = session.candidate(parent_id, self.name)
        return [(cand, cand.rationale)] if cand is not None else []


    @staticmethod
    def _parent_program(cfg: ExperimentConfig) -> str | None:
        """The parent's program source when the parent is a code.program (inline source or a sealed artifact)."""
        if cfg.agent.type != "code.program":
            return None
        p = cfg.agent.params
        if p.get("source"):
            return str(p["source"])
        if p.get("artifact"):
            from hylo.core.store import Store, is_ref

            from .skills import read_program

            art = p["artifact"]
            return read_program(Store().path(art) if is_ref(art) else art)[0]
        return None


__all__ = ["ResearchProposer"]
