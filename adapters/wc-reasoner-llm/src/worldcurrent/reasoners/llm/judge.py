"""Zero-shot VLM success judge (RFC-0004 A3): a vision-language model looks at the final camera frames and the task
instruction and says whether the task is complete. No training; trusted only after `calibrate_judge` against
simulator outcomes. Sees exactly what a real robot would record (no privileged state, no episode length).

    instrument: {type: judge.vlm, params: {model: qwen3.5:35b-a3b, cameras: [agentview, wrist]}}
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from worldcurrent.core.instrument import Judgement
from worldcurrent.core.trace import EpisodeTrace

from .client import LLMClient, LLMError

PROMPT = """You see the final state of a robot manipulation attempt: {views}.
Task instruction: "{instruction}"
Decide whether the task is fully completed in this final state. Every object named in the instruction must be in the
stated place or state (e.g. inside the basket, on the plate, drawer closed, stove turned on). An object still held by
the gripper or merely near its target does not count.
Reply with JSON only: {{"completed": true or false, "confidence": <probability from 0 to 1 that the task is completed>,
"reason": "<at most 20 words>"}}"""

VIEW_NAMES = {"agentview": "a third-person camera view", "wrist": "the robot's wrist camera view"}


def _png(img: Any, flip: bool) -> bytes:
    from PIL import Image

    arr = np.asarray(img, dtype=np.uint8)
    if flip:                                     # for envs that record raw (upside-down) renders; LIBERO's adapter
        arr = arr[::-1, ::-1]                    # already stores upright frames
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class VLMSuccessJudge:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434", api: str = "ollama",
                 cameras: tuple[str, ...] = ("agentview", "wrist"), flip: bool = False, cache_dir: str | None = None,
                 timeout_s: float = 600.0, max_tokens: int = 200, **client_kwargs: Any) -> None:
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, timeout_s=timeout_s,
                                max_tokens=max_tokens, **client_kwargs)  # type: ignore[arg-type]
        self.cameras, self.flip = tuple(cameras), flip
        self.name = f"judge.vlm[{model}]@0.1"

    def judge(self, episode: EpisodeTrace) -> Judgement:
        last = None
        for ev in episode.of_type("observation"):
            p = ev["payload"]
            if all(c in (p.get("images") or {}) for c in self.cameras):
                last = p
        if last is None:
            return Judgement(p_success=0.0, info={"error": "no frames with all cameras"})
        views = ", then ".join(VIEW_NAMES.get(c, c) for c in self.cameras)
        msg = PROMPT.format(views=views, instruction=str(last.get("instruction", "")).strip())
        try:
            res = self.client.chat_json([{"role": "user", "content": msg}],
                                        images=[_png(last["images"][c], self.flip) for c in self.cameras])
        except LLMError as exc:
            return Judgement(p_success=0.5, info={"error": str(exc)[:200]})   # uninformative, visible in the report
        c = res.content
        done = c.get("completed") is True or str(c.get("completed")).lower() == "true"
        try:
            conf = float(c.get("confidence"))
        except (TypeError, ValueError):
            conf = 1.0 if done else 0.0
        conf = min(1.0, max(0.0, conf))
        if done and conf < 0.5 or (not done and conf > 0.5):     # the model may report confidence in its own verdict
            conf = 1.0 - conf
        return Judgement(p_success=conf, info={"completed": done, "reason": str(c.get("reason", ""))[:160],
                                               "cached": res.cached, "latency_s": round(res.latency_s, 1)})
