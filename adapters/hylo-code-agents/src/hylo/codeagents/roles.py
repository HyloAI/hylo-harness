"""Role campaigns (RFC-0008 part 9): a coordinator works through a set of tasks with worker and verifier roles over
a tool surface, keeps a ledger of every tool call, and admits skills only after an independent replay.

    hylo roles run CAMPAIGN.yaml [--model ollama:qwen3.5:35b-a3b] [--out runs/roles/NAME]
    hylo roles status OUT

A campaign file lists tasks, each a stage (a hylo.stage/v0.1 file the task starts from), a request for the worker,
a goal (a check predicate) and a partition:

```text
  dev       pending -> working -> verifying -> done        a worker session works until its check holds
                              \\-> retrying -> blocked       one retry, with the failure reasons and matching skills
            after a dev task is done, its worker's successful tool calls become a proposed skill (a recipe) with the
            failure signature the task started with; a verifier (no model) replays the recipe on a fresh copy of the
            task and checks the goal, at most twice; only a passing replay promotes the skill into the library
  heldout   pending -> heldout -> done | failed            evaluated last, with the library frozen; never used to
            propose or verify skills, and never shown to a dev worker
```

Rules: every task runs on its own copy of its stage (the campaign's stage root), so tasks never see each other's
changes; every tool call goes into `ledger/<task>/<run>.jsonl` (tool, argument digest, ok, error code, receipt
status) and counts against the task's budget (`max_tool_calls`; the call that crosses it fails with
`campaign_budget_exceeded`); the coordinator takes the pending dev task with the fewest goal atoms first.
Progress is written to `status.json` after every transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from hylo.core.experience import Skill, SkillTrigger
from hylo.core.registry import registry
from hylo.core.stage import Stage, StageStore
from hylo.core.tooling import Surface, ToolError, ToolReply
from hylo.reasoners.llm.chat import ChatModel

from .session import WORLD_SYSTEM, ToolSession, _model

TRANSITIONS = {"pending": {"working", "heldout"}, "working": {"verifying", "retrying", "blocked"},
               "retrying": {"verifying", "blocked"}, "verifying": {"done", "blocked"},
               "heldout": {"done", "failed"}, "done": set(), "blocked": set(), "failed": set()}


class Ledger:
    """A surface wrapper that records every call of one run and enforces the call budget."""

    def __init__(self, inner: Surface, path: Path, max_calls: int | None) -> None:
        self.inner, self.path, self.max_calls = inner, path, max_calls
        self.calls: list[dict[str, Any]] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def call(self, name: str, args: Mapping[str, Any] | None) -> ToolReply:
        if self.max_calls is not None and len(self.calls) >= self.max_calls:
            r = ToolReply(name, False, error=ToolError("campaign_budget_exceeded",
                                                       f"this run may make {self.max_calls} tool calls",
                                                       retry="never").as_dict())
        else:
            r = self.inner.call(name, args)
        row = {"seq": len(self.calls) + 1, "time": round(time.time(), 3), "tool": name, "kind": self.inner.kind(name),
               "args_sha": hashlib.sha256(json.dumps(args or {}, sort_keys=True, default=str).encode()).hexdigest()[:16],
               "ok": r.ok, "error": (r.error or {}).get("code") if not r.ok else None}
        res = r.result or {}
        receipts = res.get("receipts") if isinstance(res, dict) else None
        if receipts:
            row["receipts"] = [x.get("status") for x in receipts]
        self.calls.append({**row, "arguments": dict(args or {})})
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return r

    def __getattr__(self, name: str) -> Any:            # specs, kind, names, digest, manual, ...
        return getattr(self.inner, name)

    def __contains__(self, name: str) -> bool:
        return name in self.inner


def goal_atoms(goal: Mapping[str, Any]) -> int:
    if goal.get("op") in ("all", "any"):
        return sum(goal_atoms(t) for t in goal.get("terms") or [])
    if goal.get("op") == "not":
        return goal_atoms(goal["term"])
    return 1


def recipe_from(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The successful acting calls of a run, in order, without generation-bound fields (they are re-read on replay)."""
    steps = []
    for c in calls:
        if c["kind"] == "act" and c["ok"] and c["tool"] not in ("world_reset",):
            args = json.loads(json.dumps(c["arguments"]))
            for a in args.get("actions") or []:
                a.pop("expected_generation", None)
                a.pop("action_id", None)
            if c["tool"] == "stage_edit":
                args.pop("expected_revision", None)
            steps.append({"tool": c["tool"], "arguments": args})
    return steps


def replay(surface: Surface, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run a recipe: stage edits against the current revision, actions against the current generation."""
    out = []
    for s in steps:
        args = json.loads(json.dumps(s["arguments"]))
        if s["tool"] == "stage_edit":
            args["expected_revision"] = surface.call("stage_show", {}).result["revision"]
        for a in args.get("actions") or []:
            a["expected_generation"] = surface.call("world_state", {}).result["generation"]
        r = surface.call(s["tool"], args)
        out.append({"tool": s["tool"], "ok": r.ok, "error": (r.error or {}).get("code")})
    return out


class RoleCampaign:
    def __init__(self, spec: Mapping[str, Any], out: str | Path, model: ChatModel | None,
                 surface_factory: Callable[[str, str], Surface] | None = None) -> None:
        self.spec, self.out, self.model = dict(spec), Path(out), model
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage_root = self.out / "stages"
        self.surface_name = spec.get("surface", "mujoco.world")
        self.factory = surface_factory or (lambda root, stage: registry.build(
            "surfaces", {"type": self.surface_name, "params": {"root": root, "stage": stage}}))
        from hylo.memory.skills import SkillLibrary

        self.library = SkillLibrary(str(self.out / "skills"))
        self.tasks = {t["id"]: {**t, "state": "pending", "runs": [], "partition": t.get("partition", "dev")}
                      for t in spec["tasks"]}
        self.budget = int(spec.get("max_tool_calls", 60))
        self.base = Path(spec.get("base_dir", "."))
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ bookkeeping
    def transition(self, tid: str, state: str, **info: Any) -> None:
        t = self.tasks[tid]
        if state not in TRANSITIONS[t["state"]]:
            raise ValueError(f"task {tid}: {t['state']} -> {state} is not allowed")
        if state == "working" and t["partition"] != "dev":
            raise ValueError(f"task {tid} is held out: it is only evaluated")
        t["state"] = state
        self.events.append({"time": round(time.time(), 3), "task": tid, "state": state, **info})
        self.save()

    def save(self) -> None:
        status = {"campaign": self.spec.get("name"), "surface": self.surface_name,
                  "tasks": {k: {kk: v for kk, v in t.items() if kk not in ("stage", "goal")} for k, t in self.tasks.items()},
                  "skills": self.library.index(status=None), "events": self.events}
        (self.out / "status.json").write_text(json.dumps(status, indent=1, default=str))

    def fresh(self, tid: str, run: str) -> tuple[Surface, str]:
        """A new copy of the task's stage (its own stage name) and a surface on it."""
        t = self.tasks[tid]
        name = f"{tid}_{run}".replace("-", "_").lower()
        store = StageStore(self.stage_root)
        if (self.stage_root / name).exists():
            shutil.rmtree(self.stage_root / name)
        src = t["stage"]
        data = json.loads((self.base / src).read_text()) if isinstance(src, str) else dict(src)
        store.create(Stage.model_validate({**data, "name": name, "revision": 0}), note=f"task {tid} run {run}")
        return self.factory(str(self.stage_root), name), name

    def check(self, surface: Surface, goal: Mapping[str, Any]) -> dict[str, Any]:
        state = surface.call("world_state", {})
        if not state.ok:
            b = surface.call("world_build", {})
            if not b.ok:
                return {"satisfied": False, "reasons": ["BUILD_FAILED"], "error": b.error}
        r = surface.call("check", {"goal": goal})
        return r.result if r.ok else {"satisfied": False, "reasons": ["CHECK_FAILED"], "error": r.error}

    # ------------------------------------------------------------------ roles
    def worker(self, tid: str, run: str, extra: str = "", frozen: bool = False) -> dict[str, Any]:
        t = self.tasks[tid]
        surface, stage = self.fresh(tid, run)
        led = Ledger(surface, self.out / "ledger" / tid / f"{run}.jsonl", t.get("max_tool_calls", self.budget))
        built = surface.call("world_build", {})
        start = self.check(surface, t["goal"]) if built.ok else {"reasons": []}
        skills = self.library.for_failure(start.get("reasons") or [], kinds=("recipe",))
        hints = ""
        if skills:
            hints = "\n\nVerified skills for this failure signature (replay their steps if they fit):\n" + "\n".join(
                f"- {s.name}: {s.description} steps={json.dumps(s.body.get('steps'))[:1500]}" for s in skills[:3])
        request = (f"{t['request']}\n\nThe stage '{stage}' is already built into a world. The goal, checked with the "
                   f"check tool, is: {json.dumps(t['goal'])}. Work until check says satisfied, then answer briefly."
                   f"{hints}{extra}")
        sess = ToolSession(self.model, led, system=WORLD_SYSTEM, out=self.out / "traces" / tid / run,
                           max_calls=int(t.get("max_model_calls", 30)))
        answer = sess.ask(request)
        sess.close()
        end = self.check(surface, t["goal"])
        rec = {"run": run, "answer": answer[:1000], "calls": len(led.calls), "satisfied": bool(end.get("satisfied")),
               "reasons": end.get("reasons"), "start_reasons": start.get("reasons"), "skills_offered":
                   [s.name for s in skills], "recipe": recipe_from(led.calls)}
        t["runs"].append({k: v for k, v in rec.items() if k != "recipe"})
        return rec

    def verifier(self, tid: str, skill: Skill, tries: int = 2) -> dict[str, Any]:
        """Replay the recipe on fresh copies of a dev task (no model) and check the goal."""
        t = self.tasks[tid]
        if t["partition"] != "dev":
            raise ValueError("verifiers only use dev tasks")
        results = []
        for k in range(tries):
            surface, _ = self.fresh(tid, f"verify{k + 1}")
            surface.call("world_build", {})
            steps = replay(surface, skill.body["steps"])
            res = self.check(surface, t["goal"])
            results.append({"try": k + 1, "satisfied": bool(res.get("satisfied")), "reasons": res.get("reasons"),
                            "steps_ok": sum(1 for s in steps if s["ok"]), "steps": len(steps)})
            if res.get("satisfied"):
                break
        return {"passed": any(r["satisfied"] for r in results), "tries": results}

    # ------------------------------------------------------------------ the coordinator
    def run(self) -> dict[str, Any]:
        dev = sorted((t for t in self.tasks.values() if t["partition"] == "dev"), key=lambda t: goal_atoms(t["goal"]))
        for t in dev:
            tid = t["id"]
            self.transition(tid, "working")
            rec = self.worker(tid, "w1")
            if not rec["satisfied"]:
                self.transition(tid, "retrying", reasons=rec["reasons"])
                rec = self.worker(tid, "w2", extra=f"\n\nAn earlier attempt ended with the check failing for "
                                                   f"{rec['reasons']}. Change the approach.")
                if not rec["satisfied"]:
                    self.transition(tid, "blocked", reasons=rec["reasons"])
                    continue
            self.transition(tid, "verifying")
            sig = tuple(sorted(rec["start_reasons"] or ()))
            if rec["recipe"] and sig and not self.library.for_failure(list(sig), kinds=("recipe",)):
                name = f"{tid}-recipe"
                skill = Skill(name=name, version="1", description=f"what solved {tid}: {t['request'][:160]}",
                              trigger=SkillTrigger(reason_codes=sig), kind="recipe",
                              body={"surface": self.surface_name, "steps": rec["recipe"]},
                              evidence={"proposed_from": tid, "calls": rec["calls"]})
                sid = self.library.add(skill)
                v = self.verifier(tid, skill)
                if v["passed"]:
                    self.library.transition(sid, "verified", {"replay": v["tries"]})
                else:
                    self.library.transition(sid, "refuted", {"replay": v["tries"]})
                self.events.append({"time": round(time.time(), 3), "task": tid, "skill": sid,
                                    "verified": v["passed"], "replay": v["tries"]})
            self.transition(tid, "done")
        for t in self.tasks.values():
            if t["partition"] != "heldout":
                continue
            self.transition(t["id"], "heldout")
            rec = self.worker(t["id"], "h1", frozen=True)
            self.transition(t["id"], "done" if rec["satisfied"] else "failed", reasons=rec["reasons"])
        self.save()
        return json.loads((self.out / "status.json").read_text())


def main(argv: Sequence[str] | None = None) -> int:
    """role campaigns: worker and verifier sessions over dev and held-out tasks, skills by replay"""
    ap = argparse.ArgumentParser(prog="hylo roles", description="role campaigns: worker / verifier / coordinator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("campaign")
    r.add_argument("--model", default="ollama:qwen3.5:35b-a3b")
    r.add_argument("--vision", action="store_true")
    r.add_argument("--out")
    s = sub.add_parser("status")
    s.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        st = json.loads((Path(a.out) / "status.json").read_text())
        for tid, t in st["tasks"].items():
            print(f"{tid:24} {t['partition']:8} {t['state']:10} runs={len(t['runs'])}")
        for sk in st["skills"]:
            print(f"skill {sk.get('id')}  {sk.get('status')}  signature {sk.get('trigger', {}).get('reason_codes')}")
        return 0
    spec = yaml.safe_load(Path(a.campaign).read_text())
    spec.setdefault("base_dir", str(Path(a.campaign).resolve().parent))
    out = a.out or f"runs/roles/{spec.get('name', 'campaign')}"
    st = RoleCampaign(spec, out, _model(a.model, a.vision)).run()
    print(json.dumps({k: v["state"] for k, v in st["tasks"].items()}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["Ledger", "RoleCampaign", "goal_atoms", "recipe_from", "replay"]
