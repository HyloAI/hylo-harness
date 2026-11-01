"""LLM proposer for steering candidates (RFC-0002 step 3).

The model reads what the search split shows (failure buckets, per-episode timelines of grasps and stalls, a few
successes for contrast), the current steering config with its schema bounds, and the archive's earlier hypotheses
with their verdicts. It never sees search_val, gate or held-out numbers. Every proposal must validate against
SteeringConfig (field bounds, rule schema); invalid proposals are dropped and counted.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from worldcurrent.agents.steering import SteeringConfig
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.trace import TraceStore
from worldcurrent.core.improvement import ArchiveEntry, Candidate
from worldcurrent.evolve.steering import STEERING_PATH

from worldcurrent.reasoners.llm.client import LLMClient, LLMError

SYSTEM = """You improve a frozen robot manipulation policy (a VLA) WITHOUT changing its weights, by editing the
configuration of a steering wrapper around it. You see diagnostics from failed and successful episodes and propose
a few alternative steering configurations, each with ONE falsifiable hypothesis about which failures it fixes and why.

The steering wrapper:
- samples action chunks from the policy with seeded noise; n_samples > 1 draws several chunks and selector
  "gripper_vote_medoid" executes the medoid of the gripper-majority samples (cost grows linearly with n_samples);
- executes `execution_steps` actions of each 50-step chunk before asking the policy again (null = all 50);
- noise_scale scales the policy's sampling noise (1.0 = as trained);
- rules: runtime triggers. feature in [still_steps, grasp_attempts, disagreement, steps], op in [>=, >, <=, <],
  action "resample" (params: noise_scale) or "backoff" (params: steps, dz, open_gripper); cooldown and max_fires
  are in control steps / per episode. disagreement requires n_samples >= 2.

Experience (when present): `lessons` are notes distilled from earlier episodes, each with an id and helpful/harmful
votes; `skills` are earlier steering configurations with a status: verified (shown to help), refuted (shown not to
help), proposed (tried, not yet shown either way). Build on verified skills and well-supported lessons, do not
re-propose refuted ones, and list the ids of the lessons each candidate relies on in "lessons_used".

Rules of the game: propose changes that plausibly generalise across tasks and init states; do not repeat an
earlier hypothesis that was refuted; keep each candidate a coherent single idea.
Reply with JSON only: {"candidates": [{"hypothesis": "...", "steering": {<complete steering config>},
"lessons_used": ["<lesson id>", ...]}]}"""


def _timeline(trace, eef_key: str = "eef_pos", gripper_key: str = "gripper_qpos", closed_below: float = 0.06,
              still_eps: float = 0.0015) -> dict[str, Any]:
    obs = [e["payload"]["state"] for e in trace.events if e["type"] == "observation"]
    instr = next((e["payload"].get("instruction") for e in trace.events if e["type"] == "observation"), "")
    end = trace.last("episode.end")["payload"]
    out: dict[str, Any] = {"instruction": instr, "status": end["status"], "steps": end["steps"]}
    if obs and eef_key in obs[0] and gripper_key in obs[0]:
        eef = np.array([o[eef_key] for o in obs])
        g = np.array([o[gripper_key] for o in obs])
        width = g[:, 0] - g[:, -1]
        closed = width < closed_below
        out["grasp_steps"] = [int(i) for i in np.flatnonzero(np.diff(closed.astype(int)) == 1)][:8]
        still = np.linalg.norm(np.diff(eef, axis=0), axis=1) < still_eps
        periods, start = [], None
        for i, s in enumerate(still):
            if s and start is None:
                start = i
            if (not s or i == len(still) - 1) and start is not None:
                if i - start >= 15:
                    periods.append([int(start), int(i - start)])
                start = None
        out["still_periods"] = periods[:6]                       # [start step, length]
        out["final_eef_z"] = round(float(eef[-1, 2]), 3)
        out["min_eef_z"] = round(float(eef[:, 2].min()), 3)
    return out


class LLMSteeringProposer:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434", api: str = "ollama",
                 n_candidates: int = 4, n_failure_examples: int = 6, n_success_examples: int = 2,
                 cache_dir: str | None = None, path: str = STEERING_PATH, max_tokens: int = 1500, **client_kwargs: Any) -> None:
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, max_tokens=max_tokens,
                                **client_kwargs)  # type: ignore[arg-type]
        self.n, self.nf, self.ns, self.path = n_candidates, n_failure_examples, n_success_examples, path
        self.name = f"llm.steering[{model}]@0.1"
        self.last_prompt: str = ""
        self.rejected: list[dict[str, Any]] = []

    def _evidence(self, diagnosis: Mapping[str, Any]) -> dict[str, Any]:
        store = TraceStore(diagnosis["run_dir"]) if diagnosis.get("run_dir") else None
        fails = [eid for ids in diagnosis.get("examples", {}).values() for eid in ids]
        fails = list(dict.fromkeys(fails))[: self.nf]
        succ = list(diagnosis.get("success_examples", []))[: self.ns]
        ev: dict[str, Any] = {"success_rate": diagnosis.get("success"), "failure_buckets": diagnosis.get("bucket_rate"),
                              "failures_per_task": diagnosis.get("per_task")}
        if store is not None:
            ev["failed_episodes"] = [_timeline(store.read(e)) for e in fails]
            ev["successful_episodes"] = [_timeline(store.read(e)) for e in succ]
        return ev

    def propose(self, diagnosis: Mapping[str, Any], cfg: ExperimentConfig, parent_id: str | None,
                archive: list[ArchiveEntry]) -> list[tuple[Candidate, str]]:
        node: Any = cfg.model_dump(mode="json")
        for part in self.path.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        current = SteeringConfig.model_validate(node or {}).model_dump(mode="json")
        history = [{"hypothesis": e.hypothesis, "verdict": e.verdict,
                    "change": e.candidate.patch.get(self.path)} for e in archive[-12:]]   # verdicts only, no numbers
        payload = {"diagnostics_from_search_split": self._evidence(diagnosis), "current_steering": current,
                   "earlier_candidates": history, "propose": self.n}
        for key in ("lessons", "skills"):                      # added by the experience layer (evolve.experience)
            if diagnosis.get(key):
                payload[key] = diagnosis[key]
        lesson_ids = {l["id"] for l in diagnosis.get("lessons") or []}
        user = json.dumps(payload, default=str)
        self.last_prompt = user
        try:
            res = self.client.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        except LLMError as exc:
            self.rejected = [{"error": str(exc)}]
            return []
        out, self.rejected = [], []
        seen = {json.dumps(e.candidate.patch, sort_keys=True) for e in archive}
        for item in (res.content.get("candidates") or [])[: self.n]:
            try:
                steering = SteeringConfig.model_validate(item.get("steering") or {}).model_dump(mode="json")
                hyp = str(item.get("hypothesis", "")).strip()
                if not hyp:
                    raise ValueError("missing hypothesis")
            except Exception as exc:  # noqa: BLE001 - invalid proposals are data
                self.rejected.append({"item": item, "error": str(exc)[:300]})
                continue
            if steering == current:
                self.rejected.append({"item": item, "error": "no change"})
                continue
            used = [i for i in item.get("lessons_used") or [] if i in lesson_ids]
            c = Candidate(patch={self.path: steering}, rationale=hyp, parent_id=parent_id, generator=self.name,
                          evidence={"llm": {"model": res.model, "prompt_hash": res.prompt_hash, "cached": res.cached},
                                    **({"lessons": used} if used else {})})
            if json.dumps(c.patch, sort_keys=True) in seen:
                self.rejected.append({"item": item, "error": "duplicate of an archived candidate"})
                continue
            out.append((c, hyp))
        return out
