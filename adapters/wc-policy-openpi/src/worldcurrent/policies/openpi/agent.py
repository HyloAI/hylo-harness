"""OpenPIAgent: an Agent backed by an openpi policy server (`scripts/serve_policy.py` in openpi), so pi0 /
pi0-FAST / pi0.5 checkpoints run in openpi's own JAX or PyTorch stack on a GPU box while the worldcurrent
kernel, safety shield and trace stay here.

    agent: {type: openpi, params: {url: "ws://gpu-box:8000", preset: libero, execution_steps: 5}}

The input mapping follows openpi's examples/libero/main.py: both cameras resized-with-pad to 224, state =
eef_pos + eef_axis_angle + gripper_qpos (8), prompt = the instruction; the server returns {"actions": (H, 7)}
and the client executes the first `execution_steps` ("replan_steps" there) before asking again.
Our LIBERO adapter already rotates images by 180 degrees as openpi does.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

from worldcurrent.core.agent import AgentSpec, EpisodeContext, EpisodeOutcome
from worldcurrent.core.errors import AgentError
from worldcurrent.core.remote import wire
from worldcurrent.core.types import ActionChunk, Observation, StepResult

# openpi key -> source: "camera:<name>", "state:<k1>,<k2>,...", or "instruction"
PRESETS: dict[str, dict[str, str]] = {
    "libero": {"observation/image": "camera:agentview", "observation/wrist_image": "camera:wrist",
               "observation/state": "state:eef_pos,eef_axis_angle,gripper_qpos", "prompt": "instruction"},
}


def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Aspect-preserving bilinear resize, zero padded and centred (tf.image.resize_with_pad semantics)."""
    from PIL import Image

    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    ratio = max(w / width, h / height)
    nh, nw = int(h / ratio), int(w / ratio)
    small = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((nw, nh), resample=Image.BILINEAR)
    canvas = Image.new(small.mode, (width, height), 0)
    canvas.paste(small, (max(0, int((width - nw) / 2)), max(0, int((height - nh) / 2))))
    return np.asarray(canvas)


class OpenPIAgent:
    def __init__(self, url: str = "ws://127.0.0.1:8000", preset: str | None = "libero",
                 inputs: dict[str, str] | None = None, image_size: int | None = 224, execution_steps: int = 5,
                 action_dim: int = 7, api_key: str | None = None, connect_timeout_s: float = 300.0,
                 weights_license: str = "see openpi checkpoint licence (Gemma terms for PaliGemma-based models)") -> None:
        self.url, self.image_size, self.execution_steps, self.action_dim = url, image_size, execution_steps, action_dim
        self.inputs = dict(PRESETS[preset]) if preset else {}
        self.inputs.update(inputs or {})
        if not self.inputs:
            raise ValueError("give a preset or an inputs mapping")
        self.weights_license = weights_license
        self._connect(api_key, connect_timeout_s)

    def _connect(self, api_key: str | None, timeout_s: float) -> None:
        import websockets.exceptions
        import websockets.sync.client as wsc

        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None   # openpi's header scheme
        direct = {"legacy": True} if "legacy" in inspect.signature(wsc.connect).parameters else {}
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                # openpi servers disable compression and frame-size limits
                self._ws = wsc.connect(self.url, compression=None, max_size=None, additional_headers=headers, **direct)
                break
            except websockets.exceptions.InvalidHandshake as exc:
                raise AgentError(f"{self.url} rejected the connection: {exc}") from exc
            except OSError as exc:
                if time.monotonic() > deadline:
                    raise AgentError(f"openpi server {self.url} unreachable: {exc!r}") from exc
                time.sleep(2.0)
        self.metadata = wire.unpackb(self._ws.recv())     # the server speaks first: its metadata dict
        if not isinstance(self.metadata, dict):
            raise AgentError(f"unexpected openpi hello: {type(self.metadata).__name__}")

    def spec(self) -> AgentSpec:
        cams = tuple(src.split(":", 1)[1] for src in self.inputs.values() if src.startswith("camera:"))
        state = tuple(k for src in self.inputs.values() if src.startswith("state:") for k in src.split(":", 1)[1].split(","))
        meta = {k: str(v)[:120] for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))}
        return AgentSpec(name="openpi.remote", version="0.1", required_state=state, required_cameras=cams,
                         action_dim=self.action_dim, weights_license=self.weights_license,
                         components={"server": self.url, "execution_steps": str(self.execution_steps),
                                     **{f"server.{k}": v for k, v in meta.items()}})

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def request(self, obs: Observation) -> dict[str, Any]:
        req: dict[str, Any] = {}
        for key, src in self.inputs.items():
            kind, _, name = src.partition(":")
            if kind == "camera":
                img = np.asarray(obs.images[name], dtype=np.uint8)
                req[key] = resize_with_pad(img, self.image_size, self.image_size) if self.image_size else img
            elif kind == "state":
                req[key] = np.concatenate([np.asarray(obs.state[k], dtype=np.float32).reshape(-1) for k in name.split(",")])
            elif src == "instruction":
                req[key] = str(obs.instruction or "")
            else:
                raise ValueError(f"bad input source {src!r} for {key!r}")
        return req

    def act(self, observation: Observation) -> ActionChunk:
        t = time.perf_counter()
        try:
            self._ws.send(wire.packb(self.request(observation)))
            reply = self._ws.recv()
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"openpi infer over {self.url} failed: {exc!r}") from exc
        if isinstance(reply, str):                        # openpi sends the traceback as a text frame
            raise AgentError(f"openpi server error:\n{reply}")
        out = wire.unpackb(reply)
        actions = np.asarray(out["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None]
        actions = actions[:, : self.action_dim]
        timing = out.get("server_timing", {}) if isinstance(out.get("server_timing"), dict) else {}
        return ActionChunk(actions=actions, execution_steps=min(self.execution_steps, len(actions)),
                           provenance={"source": "openpi", "horizon": int(len(actions)),
                                       "roundtrip_ms": round((time.perf_counter() - t) * 1000, 1),
                                       "infer_ms": float(timing.get("infer_ms", 0.0))})

    def on_step(self, result: StepResult) -> None:
        pass

    def on_episode_end(self, outcome: EpisodeOutcome) -> None:
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass
