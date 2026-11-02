"""The replay-to-sim agent (RFC-0006 part 8): a chat model drives `ReplayPipeline`'s steps from the code-agents
sandbox through `ReplayAPI` (`hylo replay ... --agent`), sees each step's summary and overlay images, and calls
finish() when the checks are done."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from .replay import ReplayPipeline


def _json(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# ==================================================================================================== agent
AGENT_SYSTEM = """You turn a recorded manipulation video into a physics simulation and check it. You work in a
Python session (only math and numpy can be imported) through the object `replay`, whose methods run the workflow's
steps and return short summaries (dicts). Images (masks, tracked models) are attached to your next message.

Steps, normally in this order:
  replay.info()                         the recording: frames, fps, task text, what the source provides
  replay.labels(names=None)             propose object labels from the first frame (or set them: a list of str)
  replay.segment(labels=None, keep=None)  masks for the labels in frame 0; keep=[indices] keeps some of them
  replay.lift()                         depth, camera frame, table plane
  replay.hands()                        human videos only: hand -> gripper (call after lift)
  replay.track(mode="yaw", gate=0.04)   object poses through the video; mode "xyz", "yaw" or "full"
  replay.register()                     camera to robot frame, IK of the robot path (when needed)
  replay.events()                       grasp / lift / release / place from the gripper and object motion
  replay.objects()                      meshes and convex collision parts
  replay.grasps()                       with a multi-finger hand (--hand): plan grasps, bend the hand's motion
  replay.scene()                        the MuJoCo scene
  replay.replay()                       kinematic and physics replay
  replay.checks()                       pose errors, matched contact events, final-state success
  replay.video()                        the side-by-side video
  finish(note)                          end the session with a one-paragraph note on what reproduced

Check each result before moving on: are the masks on the objects the task is about (look at the image)? Did a
tracked object get lost for many frames or jump? Did registration use enough pairs with a small rmse? If a check
fails, change what you control (labels, kept masks, tracking mode or gate) and run the step again; later steps must
then be re-run too. Do not invent results: report what the checks say. Write one ```python block per reply, with a
sentence of reasoning before it. Keep blocks short and print what you need."""


class ReplayAPI:
    """What the model's programs see as `replay` (every method returns a JSON-ready summary)."""

    def __init__(self, pipe: "ReplayPipeline", sandbox: Any, show: Callable[[np.ndarray, str], None]) -> None:
        self._p, self._sb, self._show = pipe, sandbox, show

    def _run(self, fn: Callable[..., dict[str, Any]], *a: Any, **kw: Any) -> dict[str, Any]:
        self._p.images.clear()
        with self._sb.paused():
            res = fn(*a, **kw)
        for img, note in self._p.images:
            self._show(img, note)
        return json.loads(json.dumps(res, default=_json))

    def info(self) -> dict[str, Any]:
        r = self._p.rec
        return {"frames": r.T, "fps": round(r.fps, 2), "instruction": r.instruction, "agent": r.agent,
                "has_depth": r.depth is not None, "has_camera_pose": r.T_world_cam is not None,
                "has_joint_positions": r.q is not None, "has_tcp": r.tcp is not None,
                "has_gripper": r.gripper is not None, "source": r.meta.get("source")}

    def labels(self, names: list[str] | None = None) -> dict[str, Any]:
        return self._run(self._p.propose_labels, list(names) if names else None)

    def segment(self, labels: list[str] | None = None, keep: list[int] | None = None) -> dict[str, Any]:
        return self._run(self._p.segment, list(labels) if labels else None, list(keep) if keep is not None else None)

    def lift(self) -> dict[str, Any]:
        return self._run(self._p.lift_to_3d)

    def hands(self) -> dict[str, Any]:
        return self._run(self._p.hands)

    def track(self, mode: str = "yaw", gate: float = 0.04) -> dict[str, Any]:
        return self._run(self._p.track, mode, float(gate))

    def register(self) -> dict[str, Any]:
        return self._run(self._p.register)

    def events(self) -> dict[str, Any]:
        return self._run(self._p.events)

    def objects(self) -> dict[str, Any]:
        return self._run(self._p.build_objects)

    def grasps(self) -> dict[str, Any]:
        return self._run(self._p.refine_grasps)

    def scene(self) -> dict[str, Any]:
        return self._run(self._p.build_scene)

    def replay(self) -> dict[str, Any]:
        return self._run(self._p.replay)

    def checks(self) -> dict[str, Any]:
        res = self._run(self._p.checks)
        for k in ("pose_error_sim_vs_tracked",):
            res[k] = {n: {kk: vv for kk, vv in v.items() if kk != "per_frame_trans_m"} for n, v in res.get(k, {}).items()}
        return res

    def video(self) -> dict[str, Any]:
        return self._run(self._p.video)


def run_agent(pipe: "ReplayPipeline", model: Any, *, turns: int = 16, block_timeout_s: float = 120.0,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """A chat model drives the pipeline from the sandbox; turns go to DIR/agent_turns.jsonl."""
    from .sandbox import Sandbox
    from .turns import Budget, CodeLoop

    say = log or (lambda m: None)
    sandbox = Sandbox(block_timeout_s=block_timeout_s)
    finished: dict[str, Any] = {}
    turns_path = pipe.out / "agent_turns.jsonl"
    turns_path.write_text("")

    def emit(rec: dict[str, Any]) -> None:
        slim = {k: v for k, v in rec.items() if k != "images"}
        slim["images"] = [i["note"] for i in rec.get("images", [])]
        with turns_path.open("a") as f:
            f.write(json.dumps(slim, default=_json) + "\n")
        say(f"turn {rec.get('turn')}: {(rec.get('code') or rec.get('model_error') or '')[:200]!r}")

    loop = CodeLoop(model, sandbox, system=AGENT_SYSTEM,
                    first=f"Recording: {pipe.rec.name}. Task text: {pipe.rec.instruction!r}. Start with replay.info().",
                    emit=emit, status=lambda: {"steps_done": list(pipe.report["steps"])})
    api = ReplayAPI(pipe, sandbox, loop.show)

    def finish(note: str = "") -> str:
        finished["note"] = str(note)[:2000]
        return "finished"

    sandbox.add(replay=api, finish=finish)
    budget = Budget(turns=turns)
    while not finished and loop.done is None:
        why = loop.exceeded(budget)
        if why:
            loop.done = why
            break
        loop.step({"turns": budget.turns - loop.turn})
    pipe.report["agent"] = {"turns": loop.turn, "usage": loop.usage.as_dict(), "stopped": loop.done or "finish()",
                            "note": finished.get("note"), "steps_run": list(pipe.report["steps"])}
    return pipe.save()


__all__ = ["AGENT_SYSTEM", "ReplayAPI", "run_agent"]
