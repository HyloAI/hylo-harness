# hylo: worker-side (runs in a tool worker's own environment; never imported by Hylo packages)
"""Worker side of the Hylo tool runner (imported by the worker scripts next to it; never by Hylo itself).

A worker script calls `main(load=..., handle=...)`:

    python worker.py REQUEST.json     one request: load, handle, write OUT_DIR/result.json
    python worker.py --serve          load once, then one request path per stdin line; each reply is one stdout line
                                      starting with RESPONSE_PREFIX

`load(params)` returns the model state (params: the first request's params, {} in serve mode); `handle(state, req)`
returns the JSON outputs and saves files through `req`. Raise `Blocked(message, action)` when a licence, login or
file the user must provide is missing.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

RESPONSE_PREFIX = "@@hylo-tool "


class Blocked(RuntimeError):
    def __init__(self, message: str, action: str) -> None:
        super().__init__(message)
        self.action = action


class Request:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.params: dict[str, Any] = body.get("params", {})
        self.inputs: dict[str, str] = body.get("inputs", {})
        self.out_dir = Path(body["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, str] = {}
        self.provenance: dict[str, Any] = {}
        self.timing: dict[str, float] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def input(self, name: str, required: bool = True) -> str | None:
        p = self.inputs.get(name)
        if p is None and required:
            raise ValueError(f"missing input {name!r}")
        return p

    def path(self, filename: str, name: str | None = None) -> Path:
        """An output path under out_dir, registered as output file `name` (default: the stem)."""
        p = self.out_dir / filename
        self.files[name or Path(filename).stem] = str(p)
        return p

    def save_npz(self, name: str = "arrays", **arrays: Any) -> Path:
        import numpy as np
        p = self.path(f"{name}.npz", name)
        np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrays.items()})
        return p

    def save_png(self, name: str, image: Any) -> Path:
        import numpy as np
        from PIL import Image
        p = self.path(f"{name}.png", name)
        img = image if hasattr(image, "save") else Image.fromarray(np.asarray(image))
        img.save(p, optimize=True)
        return p


# ---------------------------------------------------------------------- helpers for the workers
def threads() -> int:
    return int(os.environ.get("HYLO_TOOLS_THREADS", "32"))


def setup_torch() -> Any:
    """torch on the CPU: bounded threads, fp32, CUDA hidden. Returns the torch module."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(threads())
    torch.set_default_dtype(torch.float32)
    return torch


def load_rgb(path: str) -> Any:
    """An image file as an (H, W, 3) uint8 array."""
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def load_array(path: str, key: str | None = None) -> Any:
    import numpy as np
    if path.endswith(".npz"):
        with np.load(path) as z:
            return z[key or z.files[0]]
    return np.load(path)


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


def overlay_masks(rgb: Any, masks: Any, captions: list[str] | None = None, alpha: float = 0.5) -> Any:
    """The image with each mask tinted in its own colour and outlined, captions at the mask's top-left."""
    import numpy as np
    from PIL import Image, ImageDraw
    out = np.asarray(rgb, np.float32).copy()
    edges = []
    for i, m in enumerate(masks):
        m = np.asarray(m, bool)
        c = np.array(PALETTE[i % len(PALETTE)], np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
        e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        edges.append((e, c))
    for e, c in edges:
        out[e] = c
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    if captions:
        d = ImageDraw.Draw(img)
        for i, (m, text) in enumerate(zip(masks, captions)):
            ys, xs = np.nonzero(np.asarray(m, bool))
            if len(xs):
                x, y = int(xs.min()), max(0, int(ys.min()) - 11)
                d.rectangle([x, y, x + 6 * len(text) + 2, y + 10], fill=PALETTE[i % len(PALETTE)])
                d.text((x + 1, y - 1), text, fill=(0, 0, 0))
    return img


def hf_revision(repo: str) -> str | None:
    """The cached commit of a Hugging Face repo (refs/main), when known."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip() if ref.exists() else None
    except Exception:  # noqa: BLE001
        return None


def hf_access(repo: str, filename: str) -> str:
    """'ok', 'gated' (403: access not granted to the logged-in user), 'missing' or 'offline' for one repo file."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return "offline"
    try:
        get_hf_file_metadata(hf_hub_url(repo, filename))
        return "ok"
    except GatedRepoError:
        return "gated"
    except RepositoryNotFoundError:
        return "missing"
    except Exception:  # noqa: BLE001
        return "offline"


def versions(*modules: str) -> dict[str, str]:
    out = {}
    for m in modules:
        try:
            from importlib.metadata import version
            out[m] = version(m)
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------- main loop
def _reply(req: Request | None, ok: bool, outputs: Any = None, error: str | None = None,
           blocked: str | None = None) -> dict[str, Any]:
    r: dict[str, Any] = {"ok": ok}
    if req is not None:
        r.update({"files": req.files, "provenance": req.provenance, "timing": req.timing})
    if ok:
        r["outputs"] = outputs or {}
    else:
        r["error"], r["blocked"] = error, blocked
    return r


def _handle(state: Any, handle: Callable, body: dict[str, Any], load_s: float | None) -> dict[str, Any]:
    req = Request(body)
    try:
        t0 = time.perf_counter()
        outputs = handle(state, req)
        req.timing["infer_s"] = round(time.perf_counter() - t0, 3)
        if load_s is not None:
            req.timing["load_s"] = round(load_s, 3)
        req.provenance.setdefault("device", "cpu")
        req.provenance.setdefault("threads", threads())
        return _reply(req, True, outputs)
    except Blocked as exc:
        return _reply(req, False, error=str(exc), blocked=exc.action)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _reply(req, False, error=f"{type(exc).__name__}: {exc}")


def main(*, handle: Callable[[Any, Request], dict[str, Any]], load: Callable[[dict[str, Any]], Any] | None = None) -> int:
    args = sys.argv[1:]
    if args and args[0] == "--serve":
        return _serve(handle, load)
    if not args:
        print("usage: worker.py REQUEST.json | --serve", file=sys.stderr)
        return 2
    body = json.loads(Path(args[0]).read_text())
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        state = load(body.get("params", {})) if load else None
    except Blocked as exc:
        reply = _reply(Request(body), False, error=str(exc), blocked=exc.action)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        reply = _reply(Request(body), False, error=f"load failed: {type(exc).__name__}: {exc}")
    else:
        reply = _handle(state, handle, body, time.perf_counter() - t0)
    (out_dir / "result.json").write_text(json.dumps(reply, indent=2))
    return 0 if reply["ok"] else 1


def _serve(handle: Callable, load: Callable | None) -> int:
    real_stdout = sys.stdout

    def send(r: dict[str, Any]) -> None:
        real_stdout.write("\n" + RESPONSE_PREFIX + json.dumps(r) + "\n")
        real_stdout.flush()

    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            state = load({}) if load else None
    except Blocked as exc:
        send({"ok": False, "error": str(exc), "blocked": exc.action})
        return 1
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        send({"ok": False, "error": f"load failed: {type(exc).__name__}: {exc}"})
        return 1
    send({"ok": True, "ready": True, "load_s": round(time.perf_counter() - t0, 3)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        body = json.loads(Path(line).read_text())
        with contextlib.redirect_stdout(sys.stderr):
            reply = _handle(state, handle, body, None)
        (Path(body["out_dir"]) / "result.json").write_text(json.dumps(reply, indent=2))
        send(reply)
    return 0
