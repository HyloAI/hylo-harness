"""What code-writing models are told: the robot, the toolkit's API, the rules, and the feedback after each block.

Task-agnostic: the robot comes from RobotSpec, the API from the toolkit itself (hylo.manip.api_reference, else the
Tools signatures and docstrings), the task from the instruction. No task names, no success signal.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hylo.core.robot import RobotSpec
from hylo.core.types import EnvSpec

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> tuple[str | None, int]:
    """The first fenced Python block of a reply, and how many blocks there were."""
    blocks = FENCE.findall(text or "")
    if not blocks:
        return None, 0
    return blocks[0].strip("\n"), len(blocks)


def _round(x: Any, nd: int = 3) -> Any:
    if isinstance(x, np.ndarray):
        return _round(x.tolist(), nd)
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Mapping):
        return {str(k): _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def compact(value: Any, limit: int = 1500) -> str:
    """A short JSON-ish rendering of a status dict (numbers rounded, arrays as lists)."""
    try:
        text = json.dumps(_round(value), default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 20] + " ... (cut)"


def robot_description(robot: RobotSpec | None, env: EnvSpec | None) -> str:
    if robot is None:
        return "Robot: (the environment publishes no robot description)"
    lines = ["Robot:"]
    for a in robot.arms:
        h = a.hand
        hand = f"{h.kind} hand, command open={list(h.open)} closed={list(h.closed)}"
        if h.max_width:
            hand += f", opening {h.max_width:.3f} m"
        if h.presets:
            hand += f", presets {sorted(h.presets)}"
        lines.append(f"- arm '{a.name}': {len(a.joints)} joints {list(a.joints)}, end effector frame '{a.ee_frame}'; "
                     f"{hand}; home posture {[round(v, 3) for v in a.home]}")
    for c in robot.cameras:
        where = f", on arm '{c.mounted_on}'" if c.mounted_on else ""
        lines.append(f"- camera '{c.name}': {c.width}x{c.height} RGB{' + depth' if c.depth else ''}{where}")
    if robot.mobile_base:
        lines.append("- mobile base")
    if robot.table_height is not None:
        lines.append(f"- table surface near z = {robot.table_height:.3f} m (measure it; this is a hint)")
    if env is not None:
        lines.append(f"- control: {env.control_hz:g} Hz joint-position targets; every target goes through a safety "
                     f"shield that clamps it to the joint limits")
    lines.append("- frames: metres and radians, world frame z up; poses are 4x4 matrices. Tool centre point (TCP): "
                 "+x approach direction, +y closing direction of the fingers (a top-down TCP pose has +x along world -z)")
    if robot.notes.strip():
        lines.append(f"Notes: {robot.notes.strip()}")
    return "\n".join(lines)


def _first_paragraph(doc: str | None) -> str:
    return (doc or "").strip().split("\n\n")[0].replace("\n", " ").strip()


def api_reference(tools: Any) -> str:
    """The toolkit's reference, after what the toolkit says about itself first (`tools.extra_reference()`, e.g.
    privileged verbs)."""
    text = toolkit_reference(tools)
    extra = getattr(tools, "extra_reference", None)
    if callable(extra):
        text = str(extra()).strip() + "\n\n" + text
    return text


def toolkit_reference(tools: Any) -> str:
    """The toolkit's own reference (hylo.manip.api_reference) or, until it exists, signatures and docstrings."""
    try:
        from hylo.manip import api_reference as manip_reference  # type: ignore[attr-defined]
        from hylo.manip import Tools
        from hylo.manip.tools import verbs_for

        # The rich reference describes base Tools signatures and includes matching examples.
        # Custom toolkits can override those signatures. Keep the public verb list and
        # type descriptions, replace signatures and omit incompatible base examples.
        changed = any(callable(getattr(Tools, name, None)) and callable(getattr(type(tools), name, None))
                      and inspect.signature(getattr(Tools, name)) != inspect.signature(getattr(type(tools), name))
                      for name in dir(Tools) if not name.startswith("_"))
        text = manip_reference(tools)
        if text:
            if changed:
                signatures = {f"robot.{name}": f"robot.{name}{inspect.signature(getattr(tools, name))}"
                              for name in verbs_for(tools)}
                text = "\n".join(signatures.get(line.split("(", 1)[0], line)
                                 if line.startswith("robot.") else line for line in text.splitlines())
                before, _, rest = text.partition("## Example")
                if rest:
                    _, marker, types = rest.partition("## Types")
                    text = before + marker + types
            return str(text)
    except Exception:  # noqa: BLE001 - older toolkit without a reference
        pass
    lines = []
    for name in sorted(n for n in dir(type(tools)) if not n.startswith("_")):
        member = getattr(tools, name, None)
        if inspect.isclass(member):
            try:
                sig = str(inspect.signature(member))
            except (TypeError, ValueError):
                sig = "(...)"
            lines.append(f"robot.{name}{sig}: {_first_paragraph(inspect.getdoc(member))}".rstrip(": "))
        elif callable(member):
            try:
                sig = str(inspect.signature(member))
            except (TypeError, ValueError):
                sig = "(...)"
            lines.append(f"robot.{name}{sig}: {_first_paragraph(inspect.getdoc(member))}".rstrip(": "))
    return "\n".join(lines)


RULES = """How to work:
- Reply with a short thought, then exactly ONE fenced block: ```python ... ```. It runs as soon as you send it; you
  see what it printed, any error, and the robot's status, then you write the next block. There are no tools or
  functions to call through the chat: write the code as text in your reply, inside the fence.
- Variables, functions and imports persist between blocks. Preloaded: `robot` (the toolkit), `np` (numpy), `math`.
  The only imports allowed are math and numpy. No files, no network, no class definitions, no names that start
  with an underscore.
- Each block may compute for at most {block_s:g} s; time the robot spends moving is not counted.
- Every turn is one model call, which can be slow: when you are confident, do several steps in one block (perceive,
  plan, run a motion, check it) and print what you need to verify.
- `robot.run(plan)` is the only call that moves the robot: it returns when the motion is done. The simulation is
  paused while you think, so take the time you need.
- print() what you need to see: values you do not print are not shown to you.{vision}"""

VISION = "\n- `robot.show(view_or_image, note)` attaches an image you will see with the next feedback."
NO_VISION = "\n- You receive text only: images passed to robot.show are not shown to you."


def code_llm_system(*, robot: str, api: str, block_s: float, vision: bool, skills: str = "",
                    demonstration: str | None = None) -> str:
    parts = [
        "You control a robot by writing Python against a toolkit of perception, grasping, planning and motion "
        "verbs. Solve the task in the instruction with as few motions as is safe.",
        robot,
        "Toolkit API (`robot`):\n" + api,
        RULES.format(block_s=block_s, vision=VISION if vision else NO_VISION),
        "Ending: call finish() when the task is done, or when you are sure it cannot be done. The episode ends there "
        "and the task is judged from the final state. Nobody tells you whether it succeeded: check the scene "
        "yourself before you finish. Call finish() in a block of its own, after a block whose printed checks "
        "showed the task done; never in the same block as the motions (a failed step could not be retried).",
    ]
    if skills:
        parts.append("Verified skills: programs that passed a gate on tasks like the ones described; each runs a whole "
                     "task by itself. When a description fits your instruction, call that skill first (it reads the "
                     "instruction from robot.status() itself), then check the scene; write your own code only if no "
                     "skill fits or the check shows it failed. `skills` is already in your namespace (no import):\n"
                     + skills)
    if demonstration:
        parts.append("A program that solved a similar task (adapt it; do not assume the same object positions):\n"
                     f"```python\n{demonstration.strip()}\n```")
    return "\n\n".join(parts)


def budget_text(budget: Mapping[str, Any]) -> str:
    items = [f"{k.replace('_', ' ')} {v}" for k, v in budget.items() if v is not None]
    return "Budget: " + ", ".join(items) if items else ""


def task_message(instruction: str, status: Mapping[str, Any], budget: Mapping[str, Any], scene: str = "") -> str:
    return "\n".join(x for x in [f"Instruction: {instruction}", f"Robot status: {compact(status)}", scene,
                                 budget_text(budget), "Write your first block."] if x)


def scene_preview(tools: Any, vision: bool) -> tuple[str, list[Any]]:
    """What the first message says about the scene before any block ran: the segment labels the environment gives
    (if any) and, for a vision model, the cameras side by side. Saves the model a turn spent only looking."""
    look = getattr(tools, "look", None)
    if not callable(look):
        return "", []
    try:
        view = look()
    except Exception:  # noqa: BLE001 - no preview, the model looks itself
        return "", []
    lines = []
    segs = sorted(set(getattr(view, "segments", {}).values()))
    if segs:
        lines.append(f"Segment labels in view (robot.look().segments): {segs}")
    images = []
    cams = getattr(view, "cams", {})
    if vision and cams:
        rgbs = [c.rgb for c in cams.values() if getattr(c, "rgb", None) is not None]
        if rgbs:
            h = max(i.shape[0] for i in rgbs)
            images.append(np.concatenate([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in rgbs], axis=1))
            lines.append(f"Image attached: cameras {list(cams)} side by side, now.")
    return "\n".join(lines), images


def feedback(*, code: str | None, n_blocks: int, output: str, error: str | None, status: Mapping[str, Any] | None,
             images: Sequence[Mapping[str, Any]], vision: bool, remaining: Mapping[str, Any], seconds: float = 0.0) -> str:
    parts = []
    if code is None:
        parts.append("No ```python block found in your reply. Reply with exactly one fenced Python block.")
    else:
        if n_blocks > 1:
            parts.append(f"Note: only the first of your {n_blocks} blocks ran.")
        parts.append(f"Output:\n{output.rstrip()}" if output.strip() else "Output: (nothing printed)")
        if error:
            parts.append(f"Error:\n{error}")
        parts.append(f"(block ran {seconds:.1f} s)")
    if status is not None:
        parts.append(f"Robot status: {compact(status)}")
    if images:
        notes = "; ".join(i.get("note") or "(no note)" for i in images)
        parts.append(f"Images attached: {len(images)} ({notes})" if vision
                     else f"{len(images)} image(s) not shown: this model receives text only")
    left = budget_text(remaining).replace("Budget:", "Remaining:")
    if left:
        parts.append(left)
    return "\n".join(parts)


__all__ = ["FENCE", "api_reference", "budget_text", "code_llm_system", "compact", "extract_code", "feedback",
           "robot_description", "scene_preview", "task_message"]
