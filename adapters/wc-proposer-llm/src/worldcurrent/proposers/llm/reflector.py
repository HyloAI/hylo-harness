"""LLM reflector (RFC-0003 M4): distil lessons from the search split's successes and failures.

Like ReasoningBank it learns from failures as well as successes (pitfalls next to strategies); like ExpeL / ACE it
never rewrites the lesson book, it emits operations (ADD / UPVOTE / DOWNVOTE / EDIT / RETIRE) that LessonBook
applies deterministically. It reads exactly what the steering proposer reads (search-split diagnostics and
timelines), never search_val, gate or held-out data, so lessons cannot carry selection-split information into
proposals. Every ADD must cite episodes it was shown; uncited or out-of-scope operations are dropped.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from worldcurrent.core.trace import TraceStore

from worldcurrent.reasoners.llm.client import LLMClient, LLMError
from .proposer import _timeline

SYSTEM = """You maintain a lesson book for a robot manipulation agent (a frozen VLA policy with a steering wrapper).
From episode timelines (grasp steps, still periods, end-effector heights, outcome) you extract short, reusable
lessons: "strategy" lessons describe what successful episodes did, "pitfall" lessons describe failure patterns to
avoid. A lesson is useful only if it is specific (names the phase, object or signal) and could change a decision.

You may only:
- ADD a new lesson: {"op": "ADD", "scope": "global" | "task:<task id from the input>", "kind": "strategy"|"pitfall",
  "title": "<= 12 words", "content": "<= 60 words", "evidence": ["<episode id from the input>", ...]}
- UPVOTE or DOWNVOTE an existing lesson the new episodes confirm or contradict: {"op": "UPVOTE"|"DOWNVOTE",
  "id": "<lesson id>", "evidence": ["<episode id>"]}
- EDIT an existing lesson's content: {"op": "EDIT", "id": "<lesson id>", "content": "..."}
Do not restate existing lessons as new ones; vote on them instead. Use scope "global" only for patterns seen in
several tasks. Reply with JSON only: {"ops": [...]}"""


class LLMReflector:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434", api: str = "ollama",
                 max_new: int = 4, per_task: int = 2, max_tasks: int = 6, cache_dir: str | None = None,
                 max_tokens: int = 1500, timeout_s: float = 600.0, **client_kwargs: Any) -> None:
        # one call per cycle with a long prompt: a CPU-hosted model on a busy machine needs minutes, not seconds
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, max_tokens=max_tokens,
                                timeout_s=timeout_s, **client_kwargs)  # type: ignore[arg-type]
        self.max_new, self.per_task, self.max_tasks = max_new, per_task, max_tasks
        self.name = f"llm.reflector[{model}]@0.1"
        self.rejected: list[dict[str, Any]] = []

    def evidence(self, diagnosis: Mapping[str, Any]) -> dict[str, Any]:
        """Per task: a few failed and successful search-split episodes, as timelines."""
        store = TraceStore(diagnosis["run_dir"])
        log = json.loads((store.root / "evallog.json").read_text())
        if json.loads((store.root / "manifest.json").read_text()).get("benchmark", {}).get("split") != "search":
            raise ValueError("the reflector only reads search-split runs")
        by_task: dict[str, dict[str, list[str]]] = {}
        for s in log["samples"]:
            if s["status"] in ("success", "failure"):
                by_task.setdefault(s["task_id"], {"success": [], "failure": []})[s["status"]].append(s["episode_id"])
        mixed = sorted(by_task, key=lambda t: -min(len(by_task[t]["success"]), len(by_task[t]["failure"])))
        tasks = []
        for t in mixed[: self.max_tasks]:
            eps = {k: v[: self.per_task] for k, v in by_task[t].items()}
            tasks.append({"task_id": t, "success_rate": round(len(by_task[t]["success"]) / max(1, sum(map(len, by_task[t].values()))), 2),
                          "episodes": [{"id": e, **_timeline(store.read(e))} for k in ("failure", "success") for e in eps[k]]})
        return {"run_id": log["run_id"], "failure_buckets": diagnosis.get("bucket_rate"), "tasks": tasks}

    def reflect(self, diagnosis: Mapping[str, Any], book: Any) -> list[dict[str, Any]]:
        ev = self.evidence(diagnosis)
        shown = {e["id"] for t in ev["tasks"] for e in t["episodes"]}
        task_ids = {t["task_id"] for t in ev["tasks"]}
        existing = [{"id": l.id, "scope": l.scope, "kind": l.kind, "title": l.title, "content": l.content,
                     "helpful": l.helpful, "harmful": l.harmful} for l in book.all("active")][:30]
        user = json.dumps({"existing_lessons": existing, "evidence": ev, "max_new_lessons": self.max_new}, default=str)
        try:
            res = self.client.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        except LLMError as exc:
            self.rejected = [{"error": str(exc)}]
            return []
        ops, self.rejected, n_new = [], [], 0
        known = {l["id"] for l in existing}
        for op in res.content.get("ops") or []:
            kind = str(op.get("op", "")).upper()
            cites = [e for e in op.get("evidence", []) if e in shown]
            refs = [{"run_id": ev["run_id"], "episode_id": e, "split": "search"} for e in cites]
            if kind == "ADD":
                scope = str(op.get("scope", ""))
                if n_new >= self.max_new or not cites or not (scope == "global" or scope.removeprefix("task:") in task_ids):
                    self.rejected.append({"op": op, "error": "uncited, out of scope, or over the limit"})
                    continue
                if op.get("kind") not in ("strategy", "pitfall") or not op.get("title") or not op.get("content"):
                    self.rejected.append({"op": op, "error": "incomplete"})
                    continue
                ops.append({"op": "ADD", "scope": scope, "kind": op["kind"], "title": str(op["title"])[:120],
                            "content": str(op["content"])[:600], "evidence": refs})
                n_new += 1
            elif kind in ("UPVOTE", "DOWNVOTE") and op.get("id") in known and cites:
                ops.append({"op": kind, "id": op["id"], "evidence": refs})
            elif kind == "EDIT" and op.get("id") in known and op.get("content"):
                ops.append({"op": "EDIT", "id": op["id"], "content": str(op["content"])[:600]})
            else:
                self.rejected.append({"op": op, "error": "invalid"})
        return ops
