"""LLM/VLM subgoal planner: a text LLM decomposes the instruction; a VLM checks completion from the camera.

    agent: {type: planner.subgoal, params: {inner: {...}, planner: {type: planner.llm, params: {...}}}}
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from worldcurrent.agents.planner import Plan, Subgoal
from worldcurrent.core.types import Observation

from .client import LLMClient, LLMError

PLAN_PROMPT = """You plan for a robot arm controlled by a vision-language-action policy that was trained on short
tabletop instructions such as "pick up the alphabet soup and place it in the basket", "open the top drawer of the
cabinet", "put the bowl on the plate", "turn on the stove". Decompose the instruction into at most {max_subgoals}
sequential sub-instructions in that style, one object or one action each. If it is already a single step, return it
unchanged as the only sub-instruction. Keep object names exactly as written in the instruction.
Instruction: "{instruction}"
Reply with JSON only: {{"subgoals": ["...", "..."], "rationale": "<one sentence>"}}"""

DONE_PROMPT = """The image shows a robot workspace from a fixed camera. Has this sub-task been completed?
Sub-task: "{subgoal}"
Answer conservatively: only "true" if the end state is clearly visible.
Reply with JSON only: {{"done": true|false, "rationale": "<one short sentence>"}}"""


class LLMSubgoalPlanner:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434", api: str = "ollama",
                 camera: str = "agentview", max_subgoals: int = 3, subgoal_max_steps: int = 260,
                 plan_prompt: str = PLAN_PROMPT, done_prompt: str = DONE_PROMPT, cache_dir: str | None = None,
                 **client_kwargs: Any) -> None:
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, **client_kwargs)  # type: ignore[arg-type]
        self.camera, self.max_subgoals, self.subgoal_max_steps = camera, max_subgoals, subgoal_max_steps
        self.plan_prompt, self.done_prompt = plan_prompt, done_prompt
        self.name = f"planner.llm[{model}]@0.1"

    def plan(self, instruction: str, observation: Observation) -> Plan:
        try:
            res = self.client.chat_json([{"role": "user", "content": self.plan_prompt.format(
                instruction=instruction, max_subgoals=self.max_subgoals)}])
            goals = [str(g).strip() for g in res.content.get("subgoals", []) if str(g).strip()][: self.max_subgoals]
            if not goals:
                raise ValueError("empty plan")
            return Plan(subgoals=[Subgoal(instruction=g, max_steps=self.subgoal_max_steps) for g in goals],
                        rationale=str(res.content.get("rationale", ""))[:300],
                        meta={"model": res.model, "cached": res.cached, "latency_s": round(res.latency_s, 2)})
        except (LLMError, ValueError) as exc:
            return Plan(subgoals=[Subgoal(instruction=instruction, max_steps=10_000)], rationale=f"fallback: {exc}")

    def done(self, subgoal: Subgoal, observation: Observation) -> tuple[bool, str, dict[str, Any]]:
        img = observation.images.get(self.camera)
        if img is None:
            return False, f"no {self.camera} image", {}
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
        try:
            res = self.client.chat_json([{"role": "user", "content": self.done_prompt.format(subgoal=subgoal.instruction)}],
                                        images=[buf.getvalue()])
        except LLMError as exc:
            return False, f"VLM unavailable: {exc}", {}
        done = res.content.get("done") is True or str(res.content.get("done")).lower() == "true"
        return done, str(res.content.get("rationale", ""))[:200], {"cached": res.cached, "latency_s": round(res.latency_s, 2)}
