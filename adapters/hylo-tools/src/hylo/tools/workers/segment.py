# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch>=2.5", "torchvision", "transformers>=5.0", "huggingface_hub", "pillow>=10", "numpy>=1.26"]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Open-vocabulary instance segmentation with SAM 3 (transformers Sam3Model) on the CPU.

params: prompts [str], threshold (instance score, default 0.5), mask_threshold (0.5), model (facebook/sam3),
        max_instances (per prompt, default 20), overlay (bool, default true)
inputs: image
outputs: instances [{label, score, box_xyxy, area}], masks.npz (masks (N, H, W) bool, scores, labels, boxes)
"""

from __future__ import annotations

import numpy as np

from _common import Blocked, hf_access, hf_revision, load_rgb, main, overlay_masks, setup_torch, versions

LICENCE = "SAM License (Meta; custom, gated on Hugging Face)"


def load(params):
    torch = setup_torch()
    from PIL import Image  # noqa: F401
    from transformers import Sam3Model, Sam3Processor
    repo = params.get("model", "facebook/sam3")
    try:
        model = Sam3Model.from_pretrained(repo, torch_dtype=torch.float32).to("cpu").eval()
        processor = Sam3Processor.from_pretrained(repo)
    except OSError as exc:
        if hf_access(repo, "config.json") == "gated":
            raise Blocked(f"{repo}: access not granted (403)",
                          f"request access at https://huggingface.co/{repo} and log in (hf auth login)") from exc
        raise
    return {"torch": torch, "model": model, "processor": processor, "repo": repo}


def handle(st, req):
    torch, model, processor = st["torch"], st["model"], st["processor"]
    prompts = req.get("prompts") or []
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("segment: pass at least one text prompt")
    thr, mthr = float(req.get("threshold", 0.5)), float(req.get("mask_threshold", 0.5))
    kmax = int(req.get("max_instances", 20))
    from PIL import Image
    rgb = load_rgb(req.input("image"))
    image = Image.fromarray(rgb)
    with torch.inference_mode():
        img_in = processor(images=image, return_tensors="pt")
        vision = model.get_vision_features(pixel_values=img_in["pixel_values"])
        masks, scores, labels, boxes = [], [], [], []
        per_prompt = {}
        for p in prompts:
            txt = processor(text=p, return_tensors="pt")
            out = model(vision_embeds=vision, input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
            res = processor.post_process_instance_segmentation(
                out, threshold=thr, mask_threshold=mthr, target_sizes=img_in["original_sizes"].tolist())[0]
            order = np.argsort(-res["scores"].numpy())[:kmax]
            per_prompt[p] = len(order)
            for i in order:
                masks.append(res["masks"][i].numpy().astype(bool))
                scores.append(float(res["scores"][i]))
                boxes.append(res["boxes"][i].numpy().astype(np.float32))
                labels.append(p)
    H, W = rgb.shape[:2]
    m = np.stack(masks) if masks else np.zeros((0, H, W), bool)
    req.save_npz("masks", masks=m, scores=np.array(scores, np.float32), labels=np.array(labels, dtype="U64"),
                 boxes=np.array(boxes, np.float32).reshape(-1, 4))
    if req.get("overlay", True):
        req.save_png("overlay", overlay_masks(rgb, m, [f"{l} {s:.2f}" for l, s in zip(labels, scores)]))
    req.provenance.update({"model": st["repo"], "revision": hf_revision(st["repo"]), "licence": LICENCE,
                           "backend": "transformers.Sam3Model", "dtype": "float32",
                           "packages": versions("torch", "transformers")})
    return {"instances": [{"label": l, "score": round(s, 4), "box_xyxy": [round(float(v), 1) for v in b],
                           "area": int(mm.sum())} for l, s, b, mm in zip(labels, scores, boxes, m)],
            "per_prompt": per_prompt, "image_size": [H, W]}


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
