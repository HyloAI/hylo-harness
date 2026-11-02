"""Research sessions (RFC-0006 part 5): a model develops a program on development cases, then it is frozen.

The model works in one persistent namespace across episodes of the benchmark's `search` split (development
cases). Besides the toolkit (`robot`, on the live episode) it has:

    cases            the development cases: [{"case": i, "instruction": ...}]
    reset(case=None) end the live episode, return its outcome (success, steps, reason), start `case` (default: again)
    submit(source=None)  validate a program defining solve(robot) (default: the imports, functions and constants the
                     blocks defined) and run it, without the model (code.program), on a
                     fresh episode of every development case; returns each case's outcome and printed output
    finish()         end the session

The submission with the most development successes (the earliest on a tie) is frozen into a sealed program
artifact (provenance: the search split only) and becomes a `Candidate(kind="code")` whose patch sets
`agent.type: code.program` and the artifact: from there it is an ordinary candidate (search_val, gate, held-out,
approval). The model never sees anything but development cases. Budgets: turns, model tokens and cost, resets,
submissions. Everything goes to the session directory:

    log.jsonl            one record per turn, reset, submission and the end (images as PNG files under images/)
    live/                traces of the live episodes
    submissions/sub-NN/  solve.py, results.json and the traces of each submission's runs
    frozen/<name>/       the sealed program artifact
    summary.json         turns, usage, dev outcomes per submission, the frozen program and the candidate
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from pydantic import Field

from hylo.core.artifact import Provenance, SourceRef, check_splits
from hylo.core.config import ExperimentConfig
from hylo.core.contract import Contract
from hylo.core.improvement import Candidate
from hylo.core.kernel import EpisodeRunner
from hylo.core.registry import registry
from hylo.core.task import TaskInstance
from hylo.core.trace import TraceStore
from hylo.eval.experiment import env_factory
from hylo.reasoners.llm.chat import ChatModel, Usage, build_chat, describe, png_bytes

from .agents import CodeProgramAgent, make_tools, tool_status
from .bridge import BridgeAgent, EpisodeBridge, LiveExecutor, TimedExecutor
from .prompt import RULES, NO_VISION, VISION, api_reference, budget_text, compact, robot_description
from .sandbox import CodeRejected, Sandbox, defines, truncate, validate
from .skills import load_program_skills, write_program
from .turns import Budget, CodeLoop

DEV_SPLIT = "search"


class ResearchConfig(Contract):
    dev: int = Field(default=6, ge=1)                # development cases taken from the search split
    budget: Budget = Field(default_factory=lambda: Budget(turns=40, resets=30, submits=6))
    tools: str | dict[str, Any] | None = None
    block_timeout_s: float = 60.0
    program_timeout_s: float = 300.0
    act_wait_s: float = 86400.0                      # the live episode waits while the model thinks (no sim
                                                     # time passes): a slow model must not time it out
    reply_tokens: int | None = None
    max_history_turns: int = 30
    max_context_chars: int | None = 80_000          # prompt size cap (a served model's context window)
    min_dev_successes: int = 1                       # freeze nothing below this
    family: str | None = None                        # task family recorded with the program (default: common tags)
    program_params: dict[str, Any] = Field(default_factory=dict)   # extra code.program params in the candidate
    demonstration: str | None = None
    parent_program: str | None = None                # the program being improved (a campaign's parent), if any
    skills: str | dict[str, Any] | None = None
    max_case_output: int = 1200


def dev_cases(bench: Any, n: int) -> list[TaskInstance]:
    """The first n instances of the search split, interleaved across tasks (seed 0 of every task, then seed 1, ...)."""
    check_splits([DEV_SPLIT])
    order = {t.id: i for i, t in enumerate(bench.tasks)}
    inst = sorted(bench.instances(DEV_SPLIT), key=lambda i: (i.episode_index, order[i.spec.id]))
    return inst[:n]


SYSTEM = """You are developing a Python program that makes a robot solve a family of tasks. You experiment on
development cases, then submit programs; the best submission is frozen and later evaluated on unseen cases of the
same family that you will never see.

{robot}

Toolkit API (`robot`, acting on the live development episode):
{api}

{rules}

Research functions (in your namespace):
- `cases`: the development cases, a list of {{"case": i, "instruction": ...}}. A live episode of one case is running.
- `reset(case=None)`: ends the live episode and returns its outcome {{"case", "success", "steps", "reason"}}, then
  starts case `case` (default: the same case again). Your variables survive resets.
- `submit(source=None)`: submits a program that defines `solve(robot)`; it is validated, then run without you on a
  fresh episode of every development case. `submit()` with no argument submits what your blocks defined at the top
  level: every import, function definition and constant assignment (the latest of each name), which must include
  `solve(robot)`. Or pass the complete program as a string. The program does not see your other variables. Defining
  `solve` does not submit it: only submit() evaluates it. Returns {{"submission", "successes", "cases", "results": [per case: success,
  steps, reason, output, error]}}. A program can call finish() to end its episode early; returning also ends it.
- `finish()`: ends the session. The submission with the most development successes is frozen.

Write general code: read the instruction from robot.status()["instruction"], measure the scene every time, and never
hard-code positions seen in one case.{skills}{demo}{parent}"""


class ResearchSession:
    def __init__(self, cfg: ExperimentConfig, model: str | Mapping[str, Any] | ChatModel, out_dir: str | Path, *,
                 config: ResearchConfig | Mapping[str, Any] | None = None, name: str | None = None,
                 log: Any = None) -> None:
        self.cfg = cfg
        self.rc = config if isinstance(config, ResearchConfig) else ResearchConfig.model_validate(dict(config or {}))
        self.model = build_chat(model)
        self.out = Path(out_dir)
        self.name = name or f"{cfg.name}-research-{uuid.uuid4().hex[:6]}"
        self.say = log or (lambda msg: None)
        self.submissions: list[dict[str, Any]] = []
        self.resets = 0
        self.finished = False
        self.frozen: dict[str, Any] | None = None
        self.summary: dict[str, Any] = {}
        self.usage, self.turns, self.current = Usage(), 0, 0
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------ log
    def _log(self, type_: str, payload: Mapping[str, Any]) -> None:
        rec = {"type": type_, "t": round(time.time(), 3), **payload}
        if rec.get("images"):
            refs = []
            for i, img in enumerate(rec["images"]):
                f = self.out / "images" / f"turn-{rec.get('turn', 0):03d}-{i}.png"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(png_bytes(img["image"]))
                refs.append({"note": img.get("note", ""), "file": str(f.relative_to(self.out))})
            rec["images"] = refs
        with self._log_lock, open(self.out / "log.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ live episodes
    def _runner(self, env: Any, agent: Any, store: TraceStore) -> EpisodeRunner:
        return EpisodeRunner(env, agent, trace=store, safety=self.cfg.safety, timing=self.cfg.timing,
                             record_privileged=self.cfg.trace.record_privileged, image_every=self.cfg.trace.image_every)

    def _start(self, case: int) -> None:
        inst = self.cases[case]
        bridge = EpisodeBridge(self.env_spec, inst, act_wait_s=self.rc.act_wait_s)
        agent = BridgeAgent(bridge, components={"model": describe(self.model), "session": self.name})
        runner = self._runner(self.env, agent, self.live_store)
        box: dict[str, Any] = {}

        def kernel() -> None:
            try:
                box["record"] = runner.run(inst)
            except Exception as exc:  # noqa: BLE001 - the kernel failed outside its own error handling
                box["error"] = repr(exc)
                bridge.cancel()

        thread = threading.Thread(target=kernel, daemon=True, name="hylo-research-kernel")
        thread.start()
        self.live.bridge, self._live, self.current = bridge, (bridge, thread, box), case
        bridge.wait_ready(120.0)
        if "error" in box:                                # e.g. the environment failed to reset: say why
            raise RuntimeError(f"the live episode of case {case} could not start: {box['error'][:500]}")

    def _end_live(self, reason: str) -> dict[str, Any]:
        bridge, thread, box = self._live
        out = bridge.finish(reason)
        thread.join(timeout=120.0)
        rec = {"case": self.current, "success": out.get("success"), "steps": out.get("steps"), "reason": out.get("reason")}
        if "error" in box:
            rec["error"] = box["error"][:300]
        return rec

    # ------------------------------------------------------------------ the model's functions
    def reset(self, case: int | None = None) -> dict[str, Any]:
        b = self.rc.budget
        if b.resets is not None and self.resets >= b.resets:
            raise RuntimeError(f"reset budget used up ({b.resets})")
        case = self.current if case is None else int(case)
        if not 0 <= case < len(self.cases):
            raise ValueError(f"case must be 0..{len(self.cases) - 1}")
        with self.sandbox.paused():                     # resetting is robot time, not the block's
            outcome = self._end_live("reset")
            self.resets += 1
            self._start(case)
        self._log("reset", {"outcome": outcome, "next_case": case})
        return outcome

    def submit(self, source: str | None = None) -> dict[str, Any]:
        b = self.rc.budget
        if b.submits is not None and len(self.submissions) >= b.submits:
            raise RuntimeError(f"submission budget used up ({b.submits})")
        if source is None or callable(source):        # what the blocks defined: imports, functions, constants
            source = self.sandbox.program_source("solve")
            if source is None:
                raise ValueError("no block has defined solve(robot) yet: define it, then call submit()")
        if not isinstance(source, str):
            raise TypeError("submit(source) takes the program as a string (or nothing: the functions you defined)")
        tree = validate(source)
        if not defines(tree, "solve"):
            raise CodeRejected("the program must define a top-level function solve(robot)")
        k = len(self.submissions) + 1
        d = self.out / "submissions" / f"sub-{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "solve.py").write_text(source)
        with self.sandbox.paused():
            results = self._run_submission(source, d)
        n = sum(1 for r in results if r["success"])
        run_id = f"{self.name}-sub{k:02d}"
        (d / "results.json").write_text(json.dumps({"submission": k, "successes": n, "results": results}, indent=2, default=str))
        (d / "manifest.json").write_text(json.dumps({"run_id": run_id, "benchmark": {**self.bench_ref, "split": DEV_SPLIT},
                                                     "config_digest": self.cfg.digest()}, indent=2))
        sub = {"submission": k, "successes": n, "cases": len(results), "dir": str(d), "run_id": run_id,
               "source": source, "results": results}
        self.submissions.append(sub)
        self._log("submit", {k2: v for k2, v in sub.items() if k2 != "source"})
        self.say(f"[{self.name}] submission {k}: {n}/{len(results)} development successes")
        return {"submission": k, "successes": n, "cases": len(results),
                "results": [{kk: r[kk] for kk in ("case", "success", "steps", "reason", "output", "error")} for r in results]}

    def _run_submission(self, source: str, d: Path) -> list[dict[str, Any]]:
        if self.sub_env is None:
            self.sub_env = env_factory(self.cfg)()
        store = TraceStore(d)
        results = []
        try:
            for i, inst in enumerate(self.cases):
                agent = CodeProgramAgent(source=source, tools=self.rc.tools, timeout_s=self.rc.program_timeout_s,
                                         record_output=self.rc.max_case_output)
                rec = self._runner(self.sub_env, agent, store).run(inst)
                o = rec.outcome
                results.append({"case": i, "key": inst.key, "episode_id": o.episode_id, "status": o.status,
                                "success": bool(o.success), "steps": o.steps, "reason": o.termination_reason,
                                "output": truncate(agent.last.get("output", ""), self.rc.max_case_output),
                                "error": agent.last.get("error")})
        finally:
            store.close()
        return results

    def finish(self) -> str:
        self.finished = True
        best = self.best()
        return (f"session over; submission {best['submission']} ({best['successes']}/{best['cases']}) will be frozen"
                if best else "session over; nothing was submitted")

    def best(self) -> dict[str, Any] | None:
        if not self.submissions:
            return None
        return max(self.submissions, key=lambda s: (s["successes"], -s["submission"]))

    # ------------------------------------------------------------------ the session
    def _remaining(self, loop: CodeLoop) -> dict[str, Any]:
        b = self.rc.budget
        left: dict[str, Any] = {"turns": b.turns - loop.turn}
        if b.resets is not None:
            left["resets"] = b.resets - self.resets
        if b.submits is not None:
            left["submissions"] = b.submits - len(self.submissions)
        if b.tokens is not None:
            left["model_tokens"] = b.tokens - loop.usage.tokens
        if b.cost_usd is not None:
            left["cost_usd"] = round(b.cost_usd - loop.usage.cost_usd, 4)
        return left

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        self.bench = registry.build("benchmarks", self.cfg.benchmark.model_dump(exclude={"split"}))
        self.bench_ref = {"id": self.bench.id, "version": self.bench.version, "digest": self.bench.digest()}
        self.cases = dev_cases(self.bench, self.rc.dev)
        self.env = env_factory(self.cfg)()
        self.sub_env = None
        self.env_spec = self.env.spec()
        self.live_store = TraceStore(self.out / "live")
        self.live = LiveExecutor()
        self.sandbox = sandbox = Sandbox({}, block_timeout_s=self.rc.block_timeout_s)
        holder: dict[str, CodeLoop] = {}
        tools = make_tools(self.rc.tools, self.env_spec, TimedExecutor(self.live, sandbox),
                           lambda img, note="": holder["loop"].show(img, note))
        self.instructions = self._instructions()
        cases = [{"case": i, "instruction": self.instructions[i]} for i in range(len(self.cases))]
        names: dict[str, Any] = {"robot": tools, "cases": cases, "reset": self.reset, "submit": self.submit,
                                 "finish": self.finish}
        skills_text = ""
        if self.rc.skills:
            spec = self.rc.skills if isinstance(self.rc.skills, Mapping) else \
                {"type": "skill_library", "params": {"path": self.rc.skills, "frozen": True}}
            lib = registry.build("memory", dict(spec))
            ns, skills_text, used = load_program_skills(lib, " ".join(c["instruction"] for c in cases),
                                                        lambda src, entry: sandbox.define(src, entry))
            if vars(ns):
                names["skills"] = ns
            self._log("skills", {"retrieved": used})
        sandbox.add(**names)
        if self.rc.parent_program:                       # the program to improve, defined as a first block
            res = sandbox.run(self.rc.parent_program)
            self._log("parent", {"defined": res.error is None, "error": res.error, "output": res.output[-500:]})
        vision = bool(getattr(self.model, "vision", False))
        system = SYSTEM.format(
            robot=robot_description(self.env_spec.robot, self.env_spec), api=api_reference(tools),
            rules=RULES.format(block_s=self.rc.block_timeout_s, vision=VISION if vision else NO_VISION),
            skills=("\n\nVerified skills (`skills.<name>(robot, **kw)`, usable live; a submission cannot call them):\n"
                    + skills_text) if skills_text else "",
            demo=(f"\n\nA program that solved a similar task:\n```python\n{self.rc.demonstration.strip()}\n```"
                  if self.rc.demonstration else ""),
            parent=(f"\n\nThe current program for this family, which you are asked to improve (it does not solve "
                    f"every case). It is already defined in your namespace (solve and its helpers, as if you had run "
                    f"it as your first block): run solve(robot) on live cases to find where it fails, redefine only "
                    f"what needs changing, then submit():\n```python\n{self.rc.parent_program.strip()}\n```"
                    if self.rc.parent_program else ""))
        stopped = None
        try:
            self._start(0)

            def status() -> dict[str, Any]:
                st = {"live_case": self.current, **tool_status(tools, self.live)}
                if st.get("ended"):                      # a model may keep acting on a finished episode otherwise
                    st["note"] = "the live episode has ended: call reset(case) to start a new one"
                return st

            loop = holder["loop"] = CodeLoop(self.model, sandbox, system=system, first="",
                                             emit=lambda rec: self._log("turn", rec), status=status,
                                             reply_tokens=self.rc.reply_tokens, max_history_turns=self.rc.max_history_turns,
                                             max_context_chars=self.rc.max_context_chars)
            loop.messages[1]["content"] = "\n".join([
                f"Development cases: {compact(cases, 3000)}", f"Live episode: case 0. Robot status: {compact(status())}",
                budget_text(self._remaining(loop)), "Write your first block."])
            while not self.finished:
                stopped = loop.exceeded(self.rc.budget) or loop.done
                if stopped:
                    break
                self.live.bridge.provenance = {"source": "research", "session": self.name, "turn": loop.turn + 1}
                loop.step(self._remaining(loop), extra={"live_case": self.current})
            stopped = stopped or "finish() called"
            self._end_live("session over")
            self.usage, self.turns = loop.usage, loop.turn
        finally:
            self.live_store.close()
            for env in (self.env, self.sub_env):
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        self.frozen = self._freeze()
        self.summary = {
            "session": self.name, "config": self.cfg.name, "config_digest": self.cfg.digest(), "benchmark": self.bench_ref,
            "split": DEV_SPLIT, "dev_cases": [{"case": i, "key": c.key, "instruction": self.instructions[i]}
                                              for i, c in enumerate(self.cases)],
            "model": describe(self.model), "budget": self.rc.budget.model_dump(exclude_none=True), "stopped": stopped,
            "turns": self.turns, "usage": self.usage.as_dict(), "resets": self.resets,
            "submissions": [{"submission": s["submission"], "successes": s["successes"], "cases": s["cases"],
                             "outcomes": [{k: r[k] for k in ("case", "success", "steps", "reason")} for r in s["results"]]}
                            for s in self.submissions],
            "frozen": self.frozen, "elapsed_s": round(time.time() - t0, 1)}
        self._log("end", {"stopped": stopped, "frozen": self.frozen, "usage": self.usage.as_dict()})
        self.write_summary()
        return self.summary

    def _instructions(self) -> list[str]:
        """Each development case's own instruction, as its episode states it (an environment realises the task's
        template per seed, e.g. which blocks in which order): the template alone would hide what is asked."""
        out = []
        for inst in self.cases:
            try:
                out.append(self.env.reset(inst).instruction or inst.spec.instruction)
            except Exception:  # noqa: BLE001 - the template is better than nothing
                out.append(inst.spec.instruction)
        return out

    def write_summary(self) -> None:
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=str))

    # ------------------------------------------------------------------ freezing
    def family(self) -> str:
        if self.rc.family:
            return self.rc.family
        tags = [set(c.spec.tags) for c in self.cases]
        common = sorted(set.intersection(*tags)) if tags else []
        return "+".join(common) or self.bench.id

    def _freeze(self) -> dict[str, Any] | None:
        best = self.best()
        if best is None or best["successes"] < self.rc.min_dev_successes:
            return None
        art = (self.out / "frozen" / f"{self.name}-sub{best['submission']:02d}").resolve()
        instructions = list(dict.fromkeys(self.instructions))
        meta = {"name": self.family(), "family": self.family(), "instructions": instructions,
                "tasks": sorted({c.spec.id for c in self.cases}), "session": self.name, "submission": best["submission"],
                "model": describe(self.model), "description": f"solves tasks like: {instructions[0]}",
                "dev": [{k: r[k] for k in ("key", "success", "steps")} for r in best["results"]]}
        prov = Provenance(builder="hylo.codeagents.research@0.1",
                          sources=[SourceRef(kind="run", id=best["run_id"], split=DEV_SPLIT, path=best["dir"],
                                             n_episodes=best["cases"])],
                          params={"benchmark": self.bench_ref, "dev_cases": [c.key for c in self.cases],
                                  "model": describe(self.model), "session": self.name})
        m = write_program(art, best["source"], meta=meta, provenance=prov, name=art.name,
                          metrics={"dev_successes": best["successes"], "dev_cases": best["cases"]})
        return {"artifact": str(art), "digest": m.digest, "submission": best["submission"],
                "dev_successes": best["successes"], "dev_cases": best["cases"]}

    def candidate(self, parent_id: str | None = None, generator: str | None = None) -> Candidate | None:
        """The frozen program as a code candidate (None when nothing was frozen)."""
        if self.frozen is None:
            return None
        from hylo.core.store import Store

        ref = Store().ref_for(self.frozen["artifact"])
        params = {"artifact": ref, "timeout_s": self.rc.program_timeout_s, **self.rc.program_params}
        if self.rc.tools is not None:
            params["tools"] = self.rc.tools
        f = self.frozen
        hyp = (f"a program written by {describe(self.model)} in research session {self.name} solved "
               f"{f['dev_successes']}/{f['dev_cases']} development cases of {self.family()}; it generalises to unseen "
               f"init states of the family")
        return Candidate(patch={"agent.type": "code.program", "agent.params": params}, rationale=hyp, parent_id=parent_id,
                         generator=generator or f"research.llm[{self.model.name}]@0.1", kind="code",
                         evidence={"artifact": {"id": ref, "path": f["artifact"], "kind": "program", "role": "behaviour",
                                                "digest": f["digest"], "splits": [DEV_SPLIT]},
                                   "dev": {"successes": f["dev_successes"], "cases": f["dev_cases"],
                                           "submission": f["submission"]},
                                   "session": str(self.out), "turns": self.turns, "usage": self.usage.as_dict()})

    # ------------------------------------------------------------------ after the session (never shown to the model)
    def evaluate(self, splits: list[str], *, with_parent: bool = False) -> dict[str, Any]:
        """Run the frozen program on other splits (search_val, gate, heldout) and record the results in summary.json."""
        from hylo.eval.experiment import run_experiment
        from hylo.eval.scoring import paired_compare

        cand = self.candidate()
        if cand is None:
            return {}
        frozen_cfg = self.cfg.with_overrides(dict(cand.patch))
        out: dict[str, Any] = {}
        for split in splits:
            r = run_experiment(frozen_cfg, split=split, run_root=self.out / "eval", label=f"frozen-{split}")
            row: dict[str, Any] = {"success_mean": r.metrics.get("success_mean"), "n": r.metrics.get("n_episodes"),
                                   "run_dir": str(r.run_dir)}
            if with_parent:
                p = run_experiment(self.cfg, split=split, run_root=self.out / "eval", label=f"parent-{split}")
                row["parent_success_mean"] = p.metrics.get("success_mean")
                row["paired"] = paired_compare(r.per_key(), p.per_key()).model_dump()
            out[split] = row
            self.say(f"[{self.name}] frozen program on {split}: success {row['success_mean']}")
        self.summary["evaluation"] = out
        self.write_summary()
        return out


__all__ = ["DEV_SPLIT", "ResearchConfig", "ResearchSession", "dev_cases"]
