"""RGB-only region proposals for set-of-marks grounding in simple tabletop scenes.

Reuses Hylo's existing colour segmenter. No depth, environment segmentation, camera calibration,
actor pose, or task ID is accepted. Proposals are image regions, not certified object instances.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def propose(rgb):
    from hylo.manip.datatypes import CameraView, View
    from hylo.manip.perception import ColourSegmenter
    image = np.asarray(rgb, dtype=np.uint8)
    view = View(cams={"front": CameraView(image, None, None, None)}, points=np.empty((0, 3)))
    labels = ["red", "green", "blue", "cyan", "purple", "pink", "yellow", "white", "gray"]
    masks = ColourSegmenter(min_pixels=10).segment(view, labels)
    candidates = []
    height, width = image.shape[:2]
    for mask in masks:
        ys, xs = np.nonzero(mask.mask)
        if len(xs) > height * width * 0.15 or min(np.ptp(xs), np.ptp(ys)) < 2 or np.ptp(xs) > width * 0.5:
            continue
        x, y = float(xs.mean()), float(ys.mean())
        if any((x - c["x"]) ** 2 + (y - c["y"]) ** 2 < 9 for c in candidates):
            continue
        crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(float)
        white_fraction = float(((crop.min(-1) > 200) & (np.ptp(crop, axis=-1) < 35)).mean())
        candidates.append({"id": len(candidates) + 1, "color": mask.label,
                           "x": x, "y": y, "area": len(xs),
                           "white_fraction": white_fraction,
                           "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    candidates = sorted(candidates, key=lambda c: c["area"], reverse=True)[:32]
    for i, candidate in enumerate(candidates):
        candidate["id"] = i + 1
    # Nearest-neighbor enlargement keeps tiny objects visible without changing source centers.
    annotated = Image.fromarray(image).resize((width * 3, height * 3), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(annotated)
    for candidate in candidates:
        x, y = candidate["x"] * 3, candidate["y"] * 3
        label_x, label_y = min(max(x + 12, 0), width * 3 - 20), max(y - 25, 0)
        draw.line((x, y, label_x, label_y + 7), fill="black", width=2)
        draw.rectangle((label_x, label_y, label_x + 19, label_y + 15), fill="yellow", outline="black")
        draw.text((label_x + 2, label_y + 2), str(candidate["id"]), fill="black")
    return candidates, annotated


def filter_targets(candidates, target, purpose, image_shape):
    """Conservative appearance priors for an explicitly named color or landing marker."""
    import re
    words = set(re.findall(r"[a-z]+", target.lower()))
    colored = {"red", "green", "blue", "cyan", "purple", "pink", "yellow"}
    if "DESTINATION" in purpose and "target" in words:
        return [c for c in candidates if c["color"] in colored and c["white_fraction"] >= 0.10], "colored_ring_destination"
    if words & {"cube", "block"} and not words & {"container", "bin", "hiding", "hidden"}:
        colors = words & colored
        if len(colors) == 1:
            return [c for c in candidates if c["color"] in colors], "explicit_color"
    if "button" in words:
        caps = []
        for c in candidates:
            x0, y0, x1, y1 = c["bbox"]
            aspect = (x1 - x0 + 1) / (y1 - y0 + 1)
            if c["color"] == "gray" and 0.5 <= aspect <= 2 and c["area"] <= image_shape[0] * image_shape[1] * 0.015:
                caps.append(c)
        if caps:
            return caps, "gray_button_cap_candidates"
    return candidates, "unfiltered"
