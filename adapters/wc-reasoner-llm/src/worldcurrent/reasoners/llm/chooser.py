"""LLM-backed object chooser: grounds a free-form instruction to one of the perceived objects.

Drop-in replacement for `chooser.rule_color` (same Chooser protocol), selected purely by config:
    agent: {type: composable.object_reach, params: {chooser: {type: chooser.llm, params: {model: ...}}}}
"""

from __future__ import annotations

from typing import Any

from worldcurrent.core.reasoning import Choice, SceneObject
from worldcurrent.core.types import Observation

from .client import LLMClient, LLMError

SYSTEM = (
    "You ground natural-language instructions for a robot arm working on a table. You receive the "
    "instruction and a list of objects with their RGB colour (0-1 floats and hex) and position in metres. "
    "Decide which single object the instruction refers to. "
    'Reply with JSON only: {"object_id": "<one of the ids>", "rationale": "<one short sentence>"}.'
)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


class LLMChooser:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434",
                 api: str = "ollama", cache_dir: str | None = None, **client_kwargs: Any) -> None:
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, **client_kwargs)  # type: ignore[arg-type]
        self.name = f"chooser.llm[{model}]@0.1"

    def choose(self, instruction: str, objects: list[SceneObject], robot_xy: tuple[float, float],
               obs: Observation | None = None) -> Choice:
        if not objects:
            return Choice(object_id=None, rationale="no objects perceived or remembered")
        lines = [f"- {o.id}: rgb=({o.rgb[0]:.2f}, {o.rgb[1]:.2f}, {o.rgb[2]:.2f}) {_hex(o.rgb)}, "
                 f"position=({o.position[0]:.2f}, {o.position[1]:.2f}) m" for o in objects]
        user = f"Instruction: {instruction}\nObjects:\n" + "\n".join(lines)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            res = self.client.chat_json(messages)
        except LLMError as exc:
            return Choice(object_id=None, rationale=f"LLM unavailable: {exc}", meta={"error": str(exc)})
        oid = res.content.get("object_id")
        meta = {"model": res.model, "latency_s": round(res.latency_s, 3), "cached": res.cached,
                "prompt_hash": res.prompt_hash, "tokens_in": res.tokens_in, "tokens_out": res.tokens_out}
        if oid not in {o.id for o in objects}:
            return Choice(object_id=None, rationale=f"model returned an unknown id {oid!r}", meta={**meta, "raw": res.raw})
        return Choice(object_id=oid, rationale=str(res.content.get("rationale", ""))[:300], meta=meta)
