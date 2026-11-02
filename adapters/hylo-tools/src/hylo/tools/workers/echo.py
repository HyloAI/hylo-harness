# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pillow>=10"]
# ///
"""A test worker: echoes its parameters, reports the image shape and mean, writes a flipped copy."""

from __future__ import annotations

from _common import load_rgb, main


def handle(state, req):
    out = {"params": req.params, "state": state}
    if req.input("image", required=False):
        img = load_rgb(req.input("image"))
        out["shape"], out["mean"] = list(img.shape), float(img.mean())
        req.save_png("flipped", img[:, ::-1])
    req.provenance.update({"model": "echo", "licence": "Apache-2.0"})
    return out


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=lambda params: "loaded"))
