"""VLM object chooser with set-of-mark prompting.

The model receives only the camera image, with each candidate object marked by its number; it never
sees colour values. Object positions are projected into the image with the pinhole parameters the
environment publishes in `Observation.extra["camera_params"][camera]`.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hylo.core.reasoning import Choice, SceneObject
from hylo.core.types import Observation

from .client import LLMClient, LLMError

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PROMPT = (
    "The image is a top-down view of a table with a robot arm. Candidate objects are marked with white number tags "
    "{marks}. Instruction: \"{instruction}\". Which numbered object does the instruction refer to? "
    'Reply with JSON only: {{"marker": <number>, "rationale": "<one short sentence about what you see>"}}.'
)


def project(points_xyz: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    """World points -> pixel (u, v) for a pinhole camera looking along its local -z (MuJoCo convention)."""
    c, R = np.asarray(cam["pos"]), np.asarray(cam["xmat"]).reshape(3, 3)
    f = (cam["height"] / 2) / np.tan(np.radians(cam["fovy_deg"]) / 2)
    pc = (np.asarray(points_xyz) - c) @ R          # rows: R^T (p - c)
    depth = -pc[:, 2]
    return np.stack([cam["width"] / 2 + f * pc[:, 0] / depth, cam["height"] / 2 - f * pc[:, 1] / depth], axis=1)


def mark(image: np.ndarray, uv: np.ndarray, labels: list[str]) -> Image.Image:
    img = Image.fromarray(image.astype(np.uint8))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT, 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    for (u, v), lab in zip(uv, labels):
        d.rounded_rectangle((u + 12, v - 14, u + 36, v + 14), 5, fill="white", outline="black")
        d.text((u + 24, v), lab, font=font, fill="black", anchor="mm")
    return img


class VLMChooser:
    def __init__(self, model: str = "qwen3.5:35b-a3b", base_url: str = "http://127.0.0.1:11434", api: str = "ollama",
                 camera: str = "top", cache_dir: str | None = None, **client_kwargs: Any) -> None:
        self.client = LLMClient(model=model, base_url=base_url, api=api, cache_dir=cache_dir, **client_kwargs)  # type: ignore[arg-type]
        self.camera = camera
        self.name = f"chooser.vlm[{model}, set-of-mark]@0.1"

    def choose(self, instruction: str, objects: list[SceneObject], robot_xy: tuple[float, float],
               obs: Observation | None = None) -> Choice:
        if not objects:
            return Choice(object_id=None, rationale="no objects perceived or remembered")
        if obs is None or self.camera not in obs.images or "camera_params" not in obs.extra:
            return Choice(object_id=None, rationale=f"no '{self.camera}' camera image in the observation")
        cam = obs.extra["camera_params"][self.camera]
        uv = project(np.array([[*o.position, 0.0] for o in objects]), cam)
        labels = [o.id.split("_")[-1] for o in objects]
        marked = mark(obs.images[self.camera], uv, labels)
        buf = io.BytesIO()
        marked.save(buf, format="PNG")
        prompt = PROMPT.format(marks=", ".join(labels), instruction=instruction)
        try:
            res = self.client.chat_json([{"role": "user", "content": prompt}], images=[buf.getvalue()])
        except LLMError as exc:
            return Choice(object_id=None, rationale=f"VLM unavailable: {exc}", meta={"error": str(exc)})
        by_label = dict(zip(labels, objects))
        picked = by_label.get(str(res.content.get("marker")))
        meta = {"model": res.model, "latency_s": round(res.latency_s, 3), "cached": res.cached, "prompt_hash": res.prompt_hash,
                "tokens_in": res.tokens_in, "tokens_out": res.tokens_out, "camera": self.camera,
                "marked_image": np.asarray(marked)}   # stored in the trace (as a blob) to show what the model saw
        if picked is None:
            return Choice(object_id=None, rationale=f"model returned an unknown marker {res.content.get('marker')!r}", meta=meta)
        return Choice(object_id=picked.id, rationale=str(res.content.get("rationale", ""))[:300], meta=meta)
