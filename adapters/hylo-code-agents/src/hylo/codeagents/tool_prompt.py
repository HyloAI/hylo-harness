"""The prompt of the `tools.llm` agent (RFC-0007 part 4), in named layers so a run records exactly what the model was
told (`bundle_digest`, and one digest per layer):

```text
  protocol     how turns work: read tools batch, one act call alone per reply, receipts, handles go stale
  evidence     how to treat what you know: seen, measured, inferred, executed, confirmed; what to do when unsure
  boundary     what the agent cannot see or decide: simulator state, the task's own success check
  embodiment   the robot, generated from its RobotSpec: arms, hands, TCP convention, frames, cameras, notes
  task         the instruction (first user message), with the opening images
```
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from hylo.core.robot import RobotSpec

PROTOCOL = """You operate a robot through tools. You cannot see or touch anything except through them.

How a turn works:
- Read tools (look, locate, pixel_to_world, world_to_pixel, calc, grasp_options, reach_check, robot_state,
  read_artifact) only measure or compute. Call several in one reply when they do not depend on each other.
- Act tools (move_tcp, grasp, place, gripper, recover, wait) move the robot. An act call must be the ONLY call in its
  reply. It ends your turn: the next message brings its receipt (status, measured errors, holding) and new images.
- finish ends the episode. Call it when the evidence shows the task done, or when it cannot be done.
- Every reply should contain at least one tool call. Think briefly in text, then call.

Handles and staleness:
- locate returns target handles (t1, t2, ...) and grasp_options returns option handles (g3, ...). They describe the
  moment they were measured. After any motion they are stale: measure again before using positions.
- Images arrive with each observation. Earlier results and images may be stored to save context; they show the
  past. read_artifact opens one; for the present, call the tool again.

Numbers:
- Derive every number with calc (offsets, midpoints, rotations, distances). Quaternions are [x, y, z, w].
- World frame: metres, +z up. The TCP of a hand: +x is the approach direction, +y the closing direction.
"""

EVIDENCE = """Evidence rules:
1. Know how you know. A fact is seen (in an image), measured (a tool returned it), inferred (you reasoned it),
   executed (a receipt says the motion ran) or confirmed (an observation after the motion shows the effect). Only
   confirmed facts show that the task is done.
2. A receipt says what the robot did, not what happened to objects. After a grasp, holding_after_lift tells you the
   hand stopped on something; the next images and a fresh locate tell you whether the object came along.
3. When two pieces of evidence disagree, get a third: another camera, another measurement, a small diagnostic motion.
4. A failed call teaches something. Do not send the same call with the same arguments again on an unchanged scene:
   change the arguments, measure something new, or try another approach. After repeated failures of one approach,
   change the approach.
5. If you are missing something, say which tool can get it: positions (locate, pixel_to_world), whether a point is
   hidden (world_to_pixel), whether a pose is reachable (reach_check), what changed (look after acting).
6. If the instruction is ambiguous, choose the reading most consistent with the scene and say which you chose.
"""

BOUNDARY = """Boundary:
- You do not have the simulator's state, object ground truth or the task's success check. The task's own check
  decides success after the episode; finish only states your view.
- Objects can move without a command (they roll, slide, fall). Something missing from one image may be hidden.
- Tools refuse unsafe or impossible motions; a refusal that moved nothing costs no decision.
- Budgets limit decisions (act calls), model calls and time. The latest message says what is left.
"""


def embodiment_card(spec: RobotSpec, env_name: str = "", control_hz: float | None = None) -> str:
    """The robot as text, from its RobotSpec only."""
    lines = ["Robot:"]
    if env_name:
        lines.append(f"- Environment: {env_name}" + (f", control at {control_hz:g} Hz" if control_hz else ""))
    for a in spec.arms:
        h = a.hand
        base = np.asarray(a.base_in_world, float)[:3, 3]
        hand = f"{h.kind} hand"
        if h.kind == "parallel" and h.max_width:
            hand += f", opens to {h.max_width * 100:.1f} cm"
        if h.presets:
            hand += f", presets {sorted(h.presets)}"
        lines.append(f"- Arm '{a.name}': {len(a.joints)} joints, base at {np.round(base, 3).tolist()}, {hand}.")
    if len(spec.arms) > 1:
        lines.append("- Several arms: pass `arm` to every arm tool. A motion moves one arm; the others hold still and "
                     "count as obstacles.")
    lines.append("- TCP (tool centre point): where a grasped object's centre sits, between the pads (among the "
                 "fingers for a dexterous hand). Its +x axis points along the approach, +y along the closing "
                 "direction, +z completes the frame.")
    if spec.cameras:
        lines.append("- Cameras: " + ", ".join(f"{c.name} ({c.width}x{c.height}{', depth' if c.depth else ''}"
                                               + (f", on arm {c.mounted_on}" if c.mounted_on else "") + ")"
                                               for c in spec.cameras) + ". Image pixels: u right, v down.")
    if spec.table_height is not None:
        lines.append(f"- Work surface at about z = {spec.table_height:.3f} m (measure it; locate reports it).")
    if spec.mobile_base:
        lines.append("- The base can move; arm tools do not move it.")
    if spec.notes.strip():
        lines.append("- Notes: " + " ".join(spec.notes.split()))
    return "\n".join(lines)


def task_message(instruction: str, contract: str | None = None, state: Mapping[str, Any] | None = None,
                 cameras: list[str] | None = None) -> str:
    parts = [f"TASK: {instruction.strip()}"]
    if contract:
        parts.append(f"Completion contract (what the check expects, beyond the words): {contract.strip()}")
    parts.append("Before acting, separate what the task requires, what the contract adds, and what is left open.")
    if cameras:
        parts.append(f"Opening images: {', '.join(cameras)} (in that order).")
    if state:
        parts.append("Robot now: " + json.dumps(state, separators=(",", ":"), default=str))
    return "\n\n".join(parts)


def bundle(spec: RobotSpec, env_name: str = "", control_hz: float | None = None,
           extra: str = "") -> tuple[str, dict[str, str]]:
    """The system message and the digests of its layers."""
    layers = {"protocol": PROTOCOL, "evidence": EVIDENCE, "boundary": BOUNDARY,
              "embodiment": embodiment_card(spec, env_name, control_hz)}
    if extra.strip():
        layers["extra"] = extra.strip()
    digests = {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in layers.items()}
    text = "\n\n".join(layers.values())
    digests["bundle"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return text, digests


__all__ = ["BOUNDARY", "EVIDENCE", "PROTOCOL", "bundle", "embodiment_card", "task_message"]
