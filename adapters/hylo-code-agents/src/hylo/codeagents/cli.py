"""`hylo research`: run a research session on an experiment's development cases and freeze the best program.

    hylo research CONFIG [--dev N] [--budget turns=40,cost=2,submits=6,resets=30] [--model SPEC] [--out DIR]
                  [--tools REF] [--set KEY=VALUE ...] [--evaluate search_val gate heldout [--with-parent]]
    hylo research skills CAMPAIGN_DIR --library DIR [--family NAME]   promoted programs -> program skills

SPEC is a chat-model spec: JSON ({"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": true}) or a shorthand
("ollama:gpt-oss:20b", "anthropic:<model>", "claude_code", "codex"); default: the experiment's code.llm model.
--evaluate runs the frozen program on those splits after the session (the model never sees these results).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from hylo.core.config import ExperimentConfig


def _model(spec: str | None, cfg: ExperimentConfig) -> Any:
    if spec:
        return json.loads(spec) if spec.lstrip().startswith("{") else spec
    if cfg.agent.type == "code.llm" and cfg.agent.params.get("model"):
        return cfg.agent.params["model"]
    raise SystemExit("hylo research: pass --model (the experiment's agent is not code.llm with a model)")


def _skills(argv: list[str]) -> int:
    from hylo.core.registry import registry

    from .skills import promote_from_campaign

    ap = argparse.ArgumentParser(prog="hylo research skills")
    ap.add_argument("campaign")
    ap.add_argument("--library", required=True)
    ap.add_argument("--family")
    a = ap.parse_args(argv)
    lib = registry.build("memory", {"type": "skill_library", "params": {"path": a.library}})
    print(json.dumps({"added": promote_from_campaign(a.campaign, lib, family=a.family)}, indent=2))
    return 0


def research(argv: list[str]) -> int:
    """research sessions: develop a program on development cases, freeze it as a code candidate"""
    if argv and argv[0] == "skills":
        return _skills(argv[1:])
    from .research import ResearchConfig, ResearchSession
    from .turns import Budget

    ap = argparse.ArgumentParser(prog="hylo research", description=__doc__.split("\n")[0])
    ap.add_argument("config", help="experiment config (YAML)")
    ap.add_argument("--dev", type=int, default=6, help="development cases from the search split")
    ap.add_argument("--budget", default="turns=40,resets=30,submits=6", help="turns=,tokens=,cost=,resets=,submits=")
    ap.add_argument("--model", help="chat model spec (JSON or shorthand)")
    ap.add_argument("--out", help="session directory (default: <output_dir>/research/<name>)")
    ap.add_argument("--name")
    ap.add_argument("--tools", help="toolkit factory module:attr (default: the experiment agent's, else hylo.manip)")
    ap.add_argument("--set", action="append", default=[], help="override a research setting, e.g. program_timeout_s=120")
    ap.add_argument("--evaluate", nargs="*", default=[], help="splits to run the frozen program on afterwards")
    ap.add_argument("--with-parent", action="store_true", help="also run the experiment's own agent and pair them")
    a = ap.parse_args(argv)
    cfg = ExperimentConfig.load(a.config)
    settings: dict[str, Any] = {"dev": a.dev, "budget": Budget.parse(a.budget)}
    tools = a.tools or (cfg.agent.params.get("tools") if cfg.agent.type in ("code.llm", "code.program") else None)
    if tools:
        settings["tools"] = tools
    for kv in a.set:
        k, _, v = kv.partition("=")
        settings[k] = yaml.safe_load(v)
    name = a.name or f"{cfg.name}-research"
    from pathlib import Path

    out = Path(a.out) if a.out else Path(cfg.output_dir) / "research" / name
    session = ResearchSession(cfg, _model(a.model, cfg), out, config=ResearchConfig.model_validate(settings), name=name,
                              log=lambda m: print(m, file=sys.stderr))
    summary = session.run()
    if a.evaluate:
        session.evaluate(list(a.evaluate), with_parent=a.with_parent)
    s = session.summary
    print(json.dumps({"session": s["session"], "out": str(out), "stopped": s["stopped"], "turns": s["turns"],
                      "usage": s["usage"], "submissions": [(x["submission"], f"{x['successes']}/{x['cases']}")
                                                           for x in s["submissions"]],
                      "frozen": s["frozen"], "evaluation": s.get("evaluation")}, indent=2, default=str))
    return 0 if summary.get("frozen") else 1


__all__ = ["research"]
