"""`segment`: text prompts -> instance masks and scores (SAM 3 through transformers, on the CPU).

    hylo tool segment IMAGE -p "red can" -p basket [--threshold 0.5] [--out DIR]

Python:
    seg = segment(rgb, ["red can", "basket"])            # Segmentation: masks (N, H, W), scores, labels, boxes
    robot = Tools(spec, executor, segmenter=Sam3Segmenter())   # the hylo.manip segmenter interface

Model: facebook/sam3 (gated: request access on Hugging Face and log in). About 4 s per prompt on a 558 x 558 image
with 32 CPU threads once loaded; the image encoder runs once per image, each prompt reuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .runner import ToolResult, ToolServer, run_worker

MODEL = "facebook/sam3"
LICENCE = "SAM License (Meta; custom, gated)"


@dataclass
class Segmentation:
    masks: np.ndarray        # (N, H, W) bool
    scores: np.ndarray       # (N,)
    labels: list[str]        # the prompt each instance answers
    boxes: np.ndarray        # (N, 4) xyxy pixels
    result: ToolResult

    def __len__(self) -> int:
        return len(self.labels)


def _parse(res: ToolResult) -> Segmentation:
    a = res.arrays("masks")
    return Segmentation(a["masks"].astype(bool), a["scores"], [str(x) for x in a["labels"]], a["boxes"], res)


def _request(prompts: Sequence[str] | str, threshold: float, mask_threshold: float, model: str,
             overlay: bool) -> dict[str, Any]:
    return {"prompts": [prompts] if isinstance(prompts, str) else list(prompts), "threshold": threshold,
            "mask_threshold": mask_threshold, "model": model, "overlay": overlay}


def segment(image: Any, prompts: Sequence[str] | str, *, threshold: float = 0.5, mask_threshold: float = 0.5,
            model: str = MODEL, overlay: bool = True, server: ToolServer | None = None, **run_kw: Any) -> Segmentation:
    """Instance masks for each text prompt. `image`: path, (H, W, 3) uint8 array or PIL image. `server`: a running
    ToolServer("segment") to reuse a loaded model (the model param is then fixed by the server)."""
    req = _request(prompts, threshold, mask_threshold, model, overlay)
    res = server.call(req, {"image": image}) if server else run_worker("segment", req, {"image": image}, **run_kw)
    return _parse(res)


class Sam3Segmenter:
    """The hylo.manip segmenter interface on SAM 3: `segment(view, labels) -> list[hylo.manip.Mask]` (one call per
    camera, all labels at once), and `__call__(rgb, labels) -> [(label, mask, score)]` for plain images. The model
    stays loaded in a persistent worker until `close()`."""

    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5, min_pixels: int = 15,
                 threads: int | None = None, persistent: bool = True) -> None:
        self.threshold, self.mask_threshold, self.min_pixels = threshold, mask_threshold, min_pixels
        self.threads, self.persistent = threads, persistent
        self._server: ToolServer | None = None

    def _run(self, rgb: np.ndarray, labels: Sequence[str]) -> Segmentation:
        if self.persistent and self._server is None:
            self._server = ToolServer("segment", threads=self.threads)
            self._server.start()
        return segment(rgb, list(labels), threshold=self.threshold, mask_threshold=self.mask_threshold,
                       overlay=False, server=self._server, threads=self.threads)

    def __call__(self, rgb: np.ndarray, labels: Sequence[str]) -> list[tuple[str, np.ndarray, float]]:
        if not labels:
            return []
        s = self._run(rgb, labels)
        return [(l, m, float(sc)) for l, m, sc in zip(s.labels, s.masks, s.scores) if m.sum() >= self.min_pixels]

    def segment(self, view: Any, labels: Sequence[str]) -> list[Any]:
        from hylo.manip.datatypes import Mask
        out = []
        for name, cam in view.cams.items():
            for label, m, score in self(cam.rgb, labels):
                out.append(Mask(label=label, camera=name, mask=m, score=score))
        return out

    def close(self) -> None:
        if self._server:
            self._server.close()
            self._server = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def cli(argv: list[str]) -> int:
    from . import _cli
    ap = _cli.parser("segment", __doc__)
    ap.add_argument("image")
    ap.add_argument("-p", "--prompt", action="append", required=True, help="text prompt (repeatable)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    return _cli.guard(lambda: _cli.finish(segment(a.image, a.prompt, threshold=a.threshold,
                                                  mask_threshold=a.mask_threshold, model=a.model,
                                                  **_cli.run_kw(a)).result, a.out))
