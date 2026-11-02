"""The robot as typed tools (RFC-0007 part 2): what the `tools.llm` agent calls, built on the manipulation toolkit.

Read tools look, measure and compute; they never move the robot and several may share one model reply. Act tools are
action proposals: each is bound to the observation step it was decided on, recorded (`agent.tool.proposal`), planned
and executed through the kernel and the SafetyShield, and answered with a measured receipt (`agent.tool.receipt`):
status (refused / arrived / stopped_short / episode_ended), steps, residual TCP error, hand state, holding. A refusal
that moved nothing does not count as a decision. `finish` ends the episode with an outcome the agent states; the
task's own success check stays the verdict and is never shown.

Frames and units: world frame, metres, +z up; orientations are quaternions in (x, y, z, w) order; the TCP follows
hylo.core.robot (+x approach, +y closing axis). Targets and grasp options are named handles (`t1`, `g3`) valid for
the observation they came from: after the robot moves they are stale and must be measured again.
"""

from __future__ import annotations

import ast
import itertools
import math
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from hylo.core.tooling import Help, Surface, ToolDef, ToolError, ToolOutput, obj

NUM = {"type": "number", "description": "number"}


def vec(n: int, what: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "number", "description": "component"}, "minItems": n, "maxItems": n,
            "description": what}


def _pose(T: np.ndarray) -> dict[str, Any]:
    T = np.asarray(T, float)
    return {"position": np.round(T[:3, 3], 4), "quat_xyzw": np.round(Rotation.from_matrix(T[:3, :3]).as_quat(), 4),
            "approach_axis": np.round(T[:3, 0], 3), "closing_axis": np.round(T[:3, 1], 3)}


def _quat_matrix(q: Any) -> np.ndarray:
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-6:
        raise ToolError("bad_quaternion", "the quaternion must be finite and non-zero", field="orientation")
    return Rotation.from_quat(q / n).as_matrix()


# ------------------------------------------------------------------------------------------------ calc


CALC_NUMPY = {"array", "dot", "cross", "sqrt", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "deg2rad",
              "rad2deg", "radians", "degrees", "mean", "median", "min", "max", "abs", "clip", "sum", "round",
              "linspace", "stack", "concatenate", "eye", "zeros", "ones", "transpose", "norm", "linalg", "inv",
              "matmul", "outer", "hypot", "sign", "floor", "ceil", "argmin", "argmax", "pi", "minimum", "maximum"}
CALC_NODES = (ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
              ast.Call, ast.Attribute, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.Compare, ast.BoolOp,
              ast.IfExp, ast.keyword, ast.Dict, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


def _rotate(quat_xyzw: Any, v: Any) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_xyzw, float)).apply(np.asarray(v, float))


CALC_HELPERS: dict[str, Callable[..., Any]] = {
    "quat_to_matrix": lambda q: Rotation.from_quat(np.asarray(q, float)).as_matrix(),
    "matrix_to_quat": lambda m: Rotation.from_matrix(np.asarray(m, float)).as_quat(),
    "quat_from_axis_angle": lambda axis, deg: Rotation.from_rotvec(
        np.asarray(axis, float) / np.linalg.norm(axis) * math.radians(deg)).as_quat(),
    "quat_multiply": lambda a, b: (Rotation.from_quat(np.asarray(a, float)) * Rotation.from_quat(np.asarray(b, float))).as_quat(),
    "rotate": _rotate,
    "euler_deg_to_quat": lambda r, p, y: Rotation.from_euler("xyz", [r, p, y], degrees=True).as_quat(),
    "quat_to_euler_deg": lambda q: Rotation.from_quat(np.asarray(q, float)).as_euler("xyz", degrees=True),
    "angle_deg": lambda a, b: math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))))),
    "dist": lambda a, b: float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))),
    "len": len, "abs": abs, "min": min, "max": max, "round": round, "float": float, "int": int,
}


def calc(code: str, bindings: Mapping[str, Any] | None = None, outputs: list[str] | None = None) -> dict[str, Any]:
    """Evaluate assignments and expressions with numbers, numpy and the rotation helpers; no loops, imports or
    definitions. Returns the named outputs, else the value of the last expression."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolError("syntax", f"line {exc.lineno}: {exc.msg}", field="code") from exc
    for node in ast.walk(tree):
        if not isinstance(node, CALC_NODES):
            raise ToolError("not_allowed", f"{type(node).__name__} is not allowed in calc (only assignments and "
                                           "expressions)", field="code")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or (node.attr not in CALC_NUMPY and node.attr not in dir(math)
                                             and node.attr not in ("T", "shape")):
                raise ToolError("not_allowed", f"attribute {node.attr!r} is not allowed", field="code")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ToolError("not_allowed", f"name {node.id!r} is not allowed", field="code")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and abs(float(node.right.value)) > 64:
                raise ToolError("not_allowed", "exponents above 64 are not allowed", field="code")
    ns: dict[str, Any] = {"np": np, "math": math, **CALC_HELPERS}
    for k, v in (bindings or {}).items():
        if not k.isidentifier() or k.startswith("_"):
            raise ToolError("bad_binding", f"binding name {k!r} is not an identifier", field="bindings")
        ns[k] = np.asarray(v, float) if isinstance(v, list) else v
    last: Any = None
    try:
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr):
                last = eval(compile(ast.Expression(stmt.value), "<calc>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
            else:
                exec(compile(ast.Module([stmt], []), "<calc>", "exec"), {"__builtins__": {}}, ns)  # noqa: S102
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError("calc_error", f"{type(exc).__name__}: {exc}"[:300], field="code") from exc
    if outputs:
        missing = [o for o in outputs if o not in ns]
        if missing:
            raise ToolError("missing_output", f"outputs not assigned: {missing}", field="outputs")
        return {o: _plain(ns[o]) for o in outputs}
    return {"value": _plain(last)}


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), 6).tolist() if v.dtype.kind in "fiub" else v.tolist()
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    if isinstance(v, (np.integer, int, bool, str)) or v is None:
        return v.item() if isinstance(v, np.generic) else v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    raise ToolError("calc_type", f"result of type {type(v).__name__} cannot be returned", field="code")


# ------------------------------------------------------------------------------------------------ surface


class RobotSurface:
    """Typed tools over one robot for one episode. `tools`: a hylo.manip Tools; `status`: the episode status
    (step, max_steps); `emit(type, payload)`: trace events; `finish(outcome, reason)`: ends the episode."""

    def __init__(self, tools: Any, *, emit: Callable[[str, Mapping[str, Any]], None] | None = None,
                 finish: Callable[[str, str], Any] | None = None, overlay_camera: str | None = None,
                 grasp_lift_m: float = 0.08, place_retreat_m: float = 0.08) -> None:
        self.t, self.emit_fn, self.finish_fn = tools, emit, finish
        self.spec = tools.spec
        self.overlay_camera = overlay_camera
        self.grasp_lift_m, self.place_retreat_m = grasp_lift_m, place_retreat_m
        self._ids = itertools.count(1)
        self._cache: tuple[int, Any, Any] | None = None
        self.targets: dict[str, dict[str, Any]] = {}
        self.options: dict[str, dict[str, Any]] = {}
        self.marks: dict[str, dict[str, Any]] = {}
        self.held: dict[str, dict[str, Any]] = {}
        self.previous: dict[str, np.ndarray] = {}
        self.decisions = 0
        self.refused = 0
        self.outcome: dict[str, Any] | None = None
        self.issues: list[str] = []
        self.surface = self._build()

    # ------------------------------------------------------------------ helpers
    def emit(self, type_: str, payload: Mapping[str, Any]) -> None:
        if self.emit_fn is not None:
            self.emit_fn(type_, payload)

    def step(self) -> int:
        return int(self.t.executor.status().get("step", 0))

    def arms(self) -> list[str]:
        return [a.name for a in self.spec.arms]

    def _arm(self, arm: str | None) -> str:
        if arm is None:
            if len(self.spec.arms) == 1:
                return self.spec.arms[0].name
            raise ToolError("arm_required", f"this robot has arms {self.arms()}; pass `arm`", field="arm")
        if arm not in self.arms():
            raise ToolError("unknown_arm", f"no arm {arm!r}; arms: {self.arms()}", field="arm")
        return arm

    def view(self) -> tuple[Any, Any]:
        """The latest view and table, measured once per observation step."""
        step = self.step()
        if self._cache is None or self._cache[0] != step:
            v = self.t.look()
            self._cache = (step, v, self.t.table(v))
        return self._cache[1], self._cache[2]

    def _cam(self, view: Any, name: str | None) -> tuple[str, Any]:
        if name is None:
            name = self.overlay_camera if self.overlay_camera in view.cams else next(iter(view.cams), None)
        if name not in view.cams:
            raise ToolError("unknown_camera", f"no camera {name!r}; cameras: {list(view.cams)}", field="camera")
        cam = view.cams[name]
        if cam.K is None or cam.T_world_cam is None:
            raise ToolError("uncalibrated", f"camera {name} has no intrinsics or pose", field="camera", retry="never")
        return name, cam

    def _fresh(self, handle: Mapping[str, Any], what: str, key: str) -> None:
        if handle["step"] != self.step():
            raise ToolError("stale", f"{what} {key} was measured at step {handle['step']}; the robot has moved since "
                                     f"(now step {self.step()}): measure again", field=what)

    # ------------------------------------------------------------------ read tools
    def robot_state(self, args: dict[str, Any]) -> dict[str, Any]:
        st = self.t.status()
        arms = {}
        for a in self.spec.arms:
            h = a.hand
            meas = np.asarray(st["hand"][a.name], float)
            frac = None
            if h.kind == "parallel" and h.open[0] != h.closed[0]:
                frac = float((meas[0] - h.closed[0]) / (h.open[0] - h.closed[0]))
            arms[a.name] = {"tcp": _pose(st["tcp"][a.name]), "joints": np.round(st["q"][a.name], 4),
                            "hand": {"kind": h.kind, "opening": None if frac is None else round(frac, 3),
                                     "measured": np.round(meas, 3), "holding": self.t.holding(a.name)},
                            "carrying": self.held.get(a.name, {}).get("target")}
        return {"step": st.get("step"), "steps_left": (st.get("max_steps") or 0) - int(st.get("step") or 0),
                "arms": arms, "decisions": self.decisions}

    def look(self, args: dict[str, Any]) -> ToolOutput:
        view, table = self.view()
        names = args.get("cameras") or list(view.cams)
        unknown = [c for c in names if c not in view.cams]
        if unknown:
            raise ToolError("unknown_camera", f"no cameras {unknown}; cameras: {list(view.cams)}", field="cameras")
        cams = {}
        for n in names:
            c = view.cams[n]
            info: dict[str, Any] = {"size": [int(c.rgb.shape[1]), int(c.rgb.shape[0])]}
            if c.K is not None:
                info["fx_fy_cx_cy"] = np.round([c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]], 2)
            if c.T_world_cam is not None:
                info["position"] = np.round(c.T_world_cam[:3, 3], 3)
                info["looking_along"] = np.round(c.T_world_cam[:3, 2], 3)
            info["depth"] = c.depth is not None
            cams[n] = info
        return ToolOutput({"step": view.step, "cameras": cams, "support_height": round(float(table.height), 4)},
                          images=[view.cams[n].rgb for n in names],
                          image_notes=[f"{n} at step {view.step}" for n in names])

    def locate(self, args: dict[str, Any]) -> ToolOutput:
        view, table = self.view()
        text = args["text"].strip()
        masks = self.t.find(view, text)
        if not masks:
            raise ToolError("not_found", f"nothing matched {text!r} in any camera; try other words (colour, shape, "
                                         "object name) or look again from another view", retry="changed_arguments")
        dets = self.t.objects(view, masks, table)[: int(args.get("max_candidates") or 3)]
        if not dets:
            raise ToolError("no_points", f"{text!r} matched pixels but no depth points above the support; it may be "
                                         "flat or out of depth range", retry="changed_arguments")
        out, boxes = [], []
        for d in dets:
            b = self.t.box(d.points, table)
            tid = f"t{next(self._ids)}"
            self.targets[tid] = {"label": d.label, "points": np.asarray(d.points), "box": b, "step": view.step}
            top = float(np.percentile(np.asarray(d.points)[:, 2], 98))
            out.append({"target": tid, "label": d.label, "score": round(float(d.score), 3),
                        "center": np.round(b.center, 4), "size": np.round(b.size, 4),
                        "yaw_deg": round(math.degrees(b.yaw), 1), "top_z": round(top, 4),
                        "bottom_z": round(float(b.bottom), 4), "points": int(len(d.points))})
            boxes.append((tid, np.asarray(d.points)))
        cam, img = self._overlay(view, boxes)
        return ToolOutput({"step": view.step, "support_height": round(float(table.height), 4), "candidates": out,
                           "note": "candidates are ranked by score; choose by what you see, not by rank"},
                          images=[img] if img is not None else [],
                          image_notes=[f"{cam}: candidates outlined and numbered"] if img is not None else [])

    def _overlay(self, view: Any, boxes: list[tuple[str, np.ndarray]]) -> tuple[str | None, Any]:
        from PIL import Image, ImageDraw

        try:
            cam_name, cam = self._cam(view, None)
        except ToolError:
            return None, None
        img = Image.fromarray(np.asarray(cam.rgb, np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(img)
        for tid, pts in boxes:
            uv, front = project(cam, pts)
            uv = uv[front]
            if len(uv) == 0:
                continue
            lo, hi = uv.min(0), uv.max(0)
            dr.rectangle([float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])], outline=(255, 230, 0), width=2)
            dr.text((float(lo[0]) + 2, max(0.0, float(lo[1]) - 11)), tid, fill=(255, 230, 0))
        return cam_name, np.asarray(img)

    def pixel_to_world(self, args: dict[str, Any]) -> dict[str, Any]:
        view, _ = self.view()
        name, cam = self._cam(view, args["camera"])
        if cam.depth is None:
            raise ToolError("no_depth", f"camera {name} has no depth", field="camera", retry="never")
        win = int(args.get("window_px") or 5)
        names = args.get("names") or []
        if names and len(names) != len(args["pixels"]):
            raise ToolError("bad_arguments", "names needs one name per pixel", field="names")
        H, W = cam.depth.shape
        out = []
        for i, (u, v) in enumerate(args["pixels"]):
            entry: dict[str, Any] = {"pixel": [u, v]}
            if not (0 <= u < W and 0 <= v < H):
                entry["error"] = f"outside the {W}x{H} image"
                out.append(entry)
                continue
            r = win // 2
            patch = cam.depth[max(0, int(v) - r): int(v) + r + 1, max(0, int(u) - r): int(u) + r + 1]
            good = patch[np.isfinite(patch) & (patch > 0)]
            if len(good) == 0:
                entry["error"] = "no depth in the window"
                out.append(entry)
                continue
            d = float(np.median(good))
            spread = float(np.percentile(good, 90) - np.percentile(good, 10))
            K, T = cam.K, cam.T_world_cam
            p_cam = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d, 1.0])
            p = (T @ p_cam)[:3]
            entry.update(world=np.round(p, 4), depth=round(d, 4), depth_spread=round(spread, 4),
                         valid_pixels=int(len(good)))
            if spread > 0.03:
                entry["warning"] = "depth varies across the window (an edge?): the point may belong to the background"
            if names:
                nm = names[i]
                old = self.marks.get(nm)
                if old is not None and old["camera"] != name:
                    gap = float(np.linalg.norm(old["world"] - p))
                    entry["cross_check_m"] = round(gap, 4)
                    if gap > 0.025:
                        entry["warning"] = (f"mark {nm} was {gap * 1000:.0f} mm away from camera {old['camera']}'s "
                                            "estimate: the two pixels are probably not the same point")
                self.marks[nm] = {"world": p, "camera": name, "step": view.step}
                entry["mark"] = nm
            out.append(entry)
        return {"step": view.step, "camera": name, "points": out}

    def world_to_pixel(self, args: dict[str, Any]) -> dict[str, Any]:
        view, _ = self.view()
        name, cam = self._cam(view, args["camera"])
        pts = np.asarray(args["points"], float)
        uv, front = project(cam, pts)
        H, W = cam.rgb.shape[:2]
        Tinv = np.linalg.inv(cam.T_world_cam)
        out = []
        for i, p in enumerate(pts):
            z = float((Tinv @ np.append(p, 1.0))[2])
            e: dict[str, Any] = {"world": np.round(p, 4), "in_front": bool(front[i])}
            if front[i]:
                u, v = uv[i]
                inside = bool(0 <= u < W and 0 <= v < H)
                e.update(pixel=np.round(uv[i], 1), in_frame=inside, expected_depth=round(z, 4))
                if inside and cam.depth is not None:
                    m = float(cam.depth[int(v), int(u)])
                    if np.isfinite(m) and m > 0:
                        e["measured_depth"] = round(m, 4)
                        e["occluded"] = bool(m < z - 0.02)
            out.append(e)
        return {"step": view.step, "camera": name, "points": out}

    def calc_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        bindings = {b["name"]: b["value"] for b in args.get("bindings") or []}
        return calc(args["code"], bindings, args.get("outputs"))

    def grasp_options(self, args: dict[str, Any]) -> dict[str, Any]:
        tid = args["target"]
        if tid not in self.targets:
            raise ToolError("unknown_target", f"no target {tid!r}; locate the object first", field="target")
        tgt = self.targets[tid]
        self._fresh(tgt, "target", tid)
        arm = self._arm(args.get("arm"))
        view, table = self.view()
        if self.spec.arm(arm).hand.kind != "parallel":
            goals = self.t.hand_grasps(tgt["points"], table, arm=arm)
        else:
            goals = self.t.grasps(tgt["points"], table, scene=view.points, arm=arm)
        if not goals:
            raise ToolError("no_grasp", f"no grasp of {tid} fits the open hand without touching the table or other "
                                        "points; it may be too wide, too flat or too close to something",
                            retry="never")
        out = []
        for g in goals[: int(args.get("max_candidates") or 3)]:
            gid = f"g{next(self._ids)}"
            reach = self.t.reach(g.tcp, arm=arm, free=g.free)
            self.options[gid] = {"goal": g, "target": tid, "arm": arm, "step": view.step}
            ax = np.asarray(g.tcp)[:3, 0]
            out.append({"option": gid, "grasp": _pose(g.tcp),
                        "pre_position": None if g.pre is None else np.round(np.asarray(g.pre)[:3, 3], 4),
                        "style": "top-down" if ax[2] < -0.7 else "side",
                        "reachable": reach["ok"], "ik_error_mm": round(reach["pos_err"] * 1000, 1)})
        return {"target": tid, "arm": arm, "options": out,
                "note": "options are stale once the robot moves; the grasp tool executes one"}

    def reach_check(self, args: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(args.get("arm"))
        T = self._target_pose(arm, args)
        r = self.t.reach(T, arm=arm)
        return {"arm": arm, "reachable": r["ok"], "position_error_mm": round(r["pos_err"] * 1000, 1),
                "rotation_error_deg": round(math.degrees(r["rot_err"]), 2), "pose": _pose(T)}

    def report_issue(self, args: dict[str, Any]) -> dict[str, Any]:
        self.issues.append(args["message"][:500])
        self.emit("agent.tool.issue", {"message": args["message"][:500], "step": self.step()})
        return {"recorded": True}

    # ------------------------------------------------------------------ act tools
    def _target_pose(self, arm: str, args: Mapping[str, Any]) -> np.ndarray:
        T = np.asarray(self.t.status()["tcp"][arm], float).copy()
        if args.get("position") is not None and args.get("delta") is not None:
            raise ToolError("bad_arguments", "give position or delta, not both", field="position")
        if args.get("position") is not None:
            T[:3, 3] = args["position"]
        elif args.get("delta") is not None:
            T[:3, 3] += np.asarray(args["delta"], float)
        if args.get("orientation") is not None and args.get("rotate") is not None:
            raise ToolError("bad_arguments", "give orientation or rotate, not both", field="orientation")
        if args.get("orientation") is not None:
            T[:3, :3] = _quat_matrix(args["orientation"])
        elif args.get("rotate") is not None:
            r = args["rotate"]
            axis = {"tcp_x": T[:3, 0], "tcp_y": T[:3, 1], "tcp_z": T[:3, 2], "world_z": np.array([0.0, 0.0, 1.0])}[r["axis"]]
            T[:3, :3] = Rotation.from_rotvec(axis * math.radians(r["degrees"])).as_matrix() @ T[:3, :3]
        return T

    def _propose(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        rec = {"action_id": f"a{next(self._ids)}", "tool": tool, "arguments": dict(args), "observation_step": self.step()}
        self.emit("agent.tool.proposal", rec)
        return rec

    def _receipt(self, prop: Mapping[str, Any], status: str, *, steps: int = 0, why: str = "",
                 stages: list[dict[str, Any]] | None = None, arm: str | None = None, **extra: Any) -> dict[str, Any]:
        counted = not (status == "refused" and steps == 0)
        if counted:
            self.decisions += 1
        else:
            self.refused += 1
        rec: dict[str, Any] = {"action_id": prop["action_id"], "tool": prop["tool"], "status": status,
                               "steps": int(steps), "counted_as_decision": counted, "step_now": self.step()}
        if why:
            rec["why"] = why
        if stages:
            rec["stages"] = stages
        if arm is not None:
            st = self.t.status()
            rec["tcp_now"] = _pose(st["tcp"][arm])
            rec["holding"] = self.t.holding(arm)
        rec.update(extra)
        self.emit("agent.tool.receipt", rec)
        return rec

    @staticmethod
    def _stage(name: str, res: Mapping[str, Any]) -> dict[str, Any]:
        err = res.get("tcp_error")
        if isinstance(err, Mapping) and err and not {"pos", "rot"} <= set(err):
            err = next(iter(err.values()))
        out: dict[str, Any] = {"stage": name, "steps": int(res.get("steps", 0))}
        if isinstance(err, Mapping) and err:
            out["tcp_error_mm"] = round(float(err["pos"]) * 1000, 1)
            out["rotation_error_deg"] = round(math.degrees(float(err["rot"])), 1)
        if res.get("interrupted"):
            out["interrupted"] = True
        if res.get("ended"):
            out["episode_ended"] = True
        return out

    @staticmethod
    def _status(stages: list[dict[str, Any]]) -> str:
        if any(s.get("episode_ended") for s in stages):
            return "episode_ended"
        if any(s.get("interrupted") for s in stages):
            return "stopped_short"
        last = stages[-1] if stages else {}
        if last.get("tcp_error_mm", 0.0) > 10.0:
            return "stopped_short"
        return "arrived"

    def _go(self, goal: Any, name: str) -> tuple[dict[str, Any] | None, str]:
        view, table = self.view()
        plan = self.t.plan(goal, table, view.points)
        if plan is None:
            return None, self.t.why() or "no plan"
        arm = goal.arm or self._arm(None)
        self.previous[arm] = np.asarray(self.t.status()["q"][arm], float).copy()
        res = self.t.run(plan)
        if res.get("error"):
            return None, str(res["error"])
        return self._stage(name, res), ""

    def move_tcp(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("move_tcp", args)
        arm = self._arm(args.get("arm"))
        if args.get("position") is None and args.get("delta") is None and args.get("orientation") is None \
                and args.get("rotate") is None:
            raise ToolError("bad_arguments", "give at least one of position, delta, orientation, rotate")
        T = self._target_pose(arm, args)
        route = args.get("route") or "free"
        verb = "reach" if route == "free" else "retreat"         # retreat: a straight line from the current TCP
        g = self.t.Goal(verb, arm, tcp=T)
        stage, why = self._go(g, f"move ({route})")
        if stage is None:
            return self._receipt(prop, "refused", why=why, arm=arm)
        return self._receipt(prop, self._status([stage]), steps=stage["steps"], stages=[stage], arm=arm, route=route)

    def grasp(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("grasp", args)
        gid = args["option"]
        if gid not in self.options:
            raise ToolError("unknown_option", f"no grasp option {gid!r}; call grasp_options first", field="option")
        opt = self.options[gid]
        self._fresh(opt, "option", gid)
        arm, goal = opt["arm"], opt["goal"]
        stage, why = self._go(goal, "approach and close")
        if stage is None:
            return self._receipt(prop, "refused", why=why, arm=arm)
        stages = [stage]
        closed_on = self.t.holding(arm)
        stage["holding_after_close"] = closed_on
        lift = float(args.get("lift_m") if args.get("lift_m") is not None else self.grasp_lift_m)
        if lift > 0 and not stage.get("episode_ended"):
            T = np.asarray(self.t.status()["tcp"][arm], float).copy()
            T[2, 3] += lift
            st2, why2 = self._go(self.t.Goal("lift", arm, tcp=T), "lift")
            stages.append(st2 or {"stage": "lift", "steps": 0, "refused": why2})
        holding = self.t.holding(arm)
        if holding:
            tgt = self.targets[opt["target"]]
            self.held[arm] = {"target": opt["target"], "label": tgt["label"], "box": tgt["box"],
                              "obj_in_tcp": np.linalg.inv(np.asarray(goal.tcp)) @ tgt["box"].pose}
        else:
            self.held.pop(arm, None)
        steps = sum(s.get("steps", 0) for s in stages)
        status = self._status(stages)
        return self._receipt(prop, status, steps=steps, stages=stages, arm=arm, holding_after_lift=holding,
                             verdict="holding the object" if holding else "the hand is empty or closed fully: "
                                                                          "the grasp missed or slipped")

    def place(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("place", args)
        arm = self._arm(args.get("arm"))
        held = self.held.get(arm)
        if held is None:
            raise ToolError("not_holding", f"arm {arm} holds nothing grasped with the grasp tool", retry="never")
        if args.get("at") is not None and args.get("on_target") is not None:
            raise ToolError("bad_arguments", "give `at` or `on_target`, not both", field="at")
        view, table = self.view()
        if args.get("on_target") is not None:
            tid = args["on_target"]
            if tid not in self.targets:
                raise ToolError("unknown_target", f"no target {tid!r}", field="on_target")
            xy = self.targets[tid]["box"].center[:2]
        elif args.get("at") is not None:
            xy = np.asarray(args["at"], float)
        else:
            raise ToolError("bad_arguments", "give `at` [x, y] or `on_target`", field="at")
        yaw = math.radians(args["yaw_deg"]) if args.get("yaw_deg") is not None else None
        goal = self.t.place_at(held["obj_in_tcp"], held["box"], xy, yaw=yaw, table=table,
                               hover=float(args.get("hover_m") if args.get("hover_m") is not None else 0.01),
                               scene=view.points, arm=arm)
        stage, why = self._go(goal, "lower and release")
        if stage is None:
            return self._receipt(prop, "refused", why=why, arm=arm)
        stages = [stage]
        self.held.pop(arm, None)
        if not stage.get("episode_ended") and self.place_retreat_m > 0:
            T = np.asarray(self.t.status()["tcp"][arm], float).copy()
            T[:3, 3] -= T[:3, 0] * self.place_retreat_m          # back out along the approach axis
            st2, why2 = self._go(self.t.Goal("retreat", arm, tcp=T), "retreat")
            stages.append(st2 or {"stage": "retreat", "steps": 0, "refused": why2})
        return self._receipt(prop, self._status(stages), steps=sum(s.get("steps", 0) for s in stages),
                             stages=stages, arm=arm, released=held["target"])

    def gripper(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("gripper", args)
        arm = self._arm(args.get("arm"))
        h = self.spec.arm(arm).hand
        cmd = {"open": h.open, "close": h.closed}.get(args["command"]) or h.presets.get(args["command"])
        if cmd is None:
            raise ToolError("bad_command", f"command must be open, close or a preset {sorted(h.presets)}",
                            field="command")
        steps = max(4, int(round(float(args.get("seconds") or 1.0) * self.t.control_hz)))
        res = self.t.run_actions(self.t.hold(steps, {arm: cmd}))
        if args["command"] == "open":
            self.held.pop(arm, None)
        stage = self._stage(f"hand {args['command']}", res)
        return self._receipt(prop, self._status([stage]) if stage.get("episode_ended") or stage.get("interrupted")
                             else "arrived", steps=stage["steps"], stages=[stage], arm=arm)

    def recover(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("recover", args)
        arm = self._arm(args.get("arm"))
        to = args.get("to") or "previous"
        if to == "previous" and arm not in self.previous:
            raise ToolError("no_previous", "nothing to go back to yet: no motion ran", field="to", retry="never")
        q = self.previous[arm] if to == "previous" else np.asarray(self.spec.arm(arm).home, float)
        stage, why = self._go(self.t.Goal("home", arm, joints=q), f"return to {to}")
        if stage is None:
            return self._receipt(prop, "refused", why=why, arm=arm)
        return self._receipt(prop, self._status([stage]), steps=stage["steps"], stages=[stage], arm=arm)

    def wait(self, args: dict[str, Any]) -> dict[str, Any]:
        prop = self._propose("wait", args)
        before = {a: np.asarray(self.t.status()["tcp"][a])[:3, 3].copy() for a in self.arms()}
        res = self.t.run_actions(self.t.hold(int(args["steps"])))
        after = {a: np.asarray(res["tcp"][a])[:3, 3] for a in self.arms()}
        drift = {a: round(float(np.linalg.norm(after[a] - before[a])) * 1000, 1) for a in before}
        stage = self._stage("wait", res)
        return self._receipt(prop, "episode_ended" if stage.get("episode_ended") else "arrived",
                             steps=stage["steps"], stages=[stage], tcp_drift_mm=drift)

    def finish(self, args: dict[str, Any]) -> dict[str, Any]:
        outcome = args["outcome"]
        if outcome == "give_up" and not args.get("category"):
            raise ToolError("category_required", "give_up needs a category", field="category")
        self.outcome = {"outcome": outcome, "reason": args["reason"][:500], "category": args.get("category"),
                        "decisions": self.decisions, "step": self.step()}
        self.emit("agent.tool.finish", self.outcome)
        if self.finish_fn is not None:
            self.finish_fn(outcome, args["reason"][:200])
        return {"ended": True}

    # ------------------------------------------------------------------ declarations
    def _build(self) -> Surface:
        arm_p = {"arm": {"type": "string", "enum": self.arms(), "description": "which arm; may be omitted when the "
                                                                                "robot has one arm"}}
        cams = [c.name for c in self.spec.cameras] or None
        cam_p = {"type": "string", "description": "camera name"} | ({"enum": cams} if cams else {})
        pose_p = {
            "position": vec(3, "target TCP position [x, y, z] in the world frame, metres"),
            "delta": vec(3, "move the TCP by [dx, dy, dz] metres in the world frame (instead of position)"),
            "orientation": vec(4, "target TCP orientation as a quaternion [x, y, z, w] (world frame)"),
            "rotate": obj({"axis": {"type": "string", "enum": ["tcp_x", "tcp_y", "tcp_z", "world_z"],
                                    "description": "axis to rotate about: the TCP's approach (x), closing (y) or "
                                                   "normal (z) axis, or the world's vertical"},
                           "degrees": {"type": "number", "minimum": -180, "maximum": 180,
                                       "description": "angle, right-handed"}}, ["axis", "degrees"])
            | {"description": "rotate the TCP from its current orientation (instead of orientation)"},
        }
        T = [
            ToolDef("robot_state", "read", Help(
                "The robot now: each arm's TCP pose, joints, hand opening and whether it is holding something.",
                use_when="after a motion, before deciding the next one; to read the TCP orientation for a move.",
                returns="step, steps_left, arms {arm: tcp {position, quat_xyzw, approach_axis, closing_axis}, "
                        "joints, hand {opening 0 closed..1 open, holding}, carrying}, decisions so far."),
                obj(), self.robot_state),
            ToolDef("look", "read", Help(
                "The current camera images with each camera's position, viewing direction and intrinsics.",
                use_when="you need to see the scene again (the images you got with the last observation are the "
                         "same moment; look does not take new pictures until the robot has moved).",
                returns="step, cameras {name: size, fx_fy_cx_cy, position, looking_along, depth}, support_height; "
                        "the images.",
                limits="depth images are not shown; measure with pixel_to_world."),
                obj({"cameras": {"type": "array", "items": {"type": "string", "description": "camera"},
                                 "description": "cameras to show (default all)"}}), self.look),
            ToolDef("locate", "read", Help(
                "Find objects described in words and measure them in 3D (segmentation, then depth fused across "
                "cameras).",
                use_when="you need an object's position and size, or a target handle for grasp_options and place.",
                returns="candidates [{target (handle), label, score, center, size (length, width, height), yaw_deg, "
                        "top_z, bottom_z}], support_height; an image with the candidates outlined.",
                limits="no object identity across steps: handles are valid until the robot moves. Ranking is by "
                       "label match, not by which object the task means.",
                errors="not_found (try other words), no_points (seen but no depth)."),
                obj({"text": {"type": "string", "maxLength": 120, "description": "what to find, e.g. 'red block'"},
                     "max_candidates": {"type": "integer", "minimum": 1, "maximum": 5,
                                        "description": "at most this many (default 3)"}}, ["text"]), self.locate),
            ToolDef("pixel_to_world", "read", Help(
                "Turn pixels of a camera image into world points using that camera's depth (median over a small "
                "window).",
                use_when="you can see a spot in an image (an edge, a slot, a mark) and need its 3D position. Name "
                         "the points to cross-check the same point seen from a second camera.",
                returns="per pixel: world [x, y, z], depth, depth_spread, valid_pixels; cross_check_m for a name "
                        "seen before from another camera; warnings at depth edges.",
                limits="does not segment; a pixel on an edge may hit the background."),
                obj({"camera": cam_p,
                     "pixels": {"type": "array", "minItems": 1, "maxItems": 8, "description": "[[u, v], ...] "
                                "pixel coordinates (u right, v down)", "items": vec(2, "[u, v]")},
                     "names": {"type": "array", "items": {"type": "string", "description": "name"},
                               "description": "optional names, one per pixel, kept as marks"},
                     "window_px": {"type": "integer", "minimum": 1, "maximum": 15,
                                   "description": "depth window size in pixels (default 5)"}},
                    ["camera", "pixels"]), self.pixel_to_world),
            ToolDef("world_to_pixel", "read", Help(
                "Project world points into a camera image, with occlusion evidence from depth.",
                use_when="to check where a measured or planned point appears in an image, or whether it is hidden.",
                returns="per point: pixel, in_front, in_frame, expected_depth, measured_depth, occluded."),
                obj({"camera": cam_p, "points": {"type": "array", "minItems": 1, "maxItems": 16,
                                                 "items": vec(3, "[x, y, z]"), "description": "world points"}},
                    ["camera", "points"]), self.world_to_pixel),
            ToolDef("calc", "read", Help(
                "Exact arithmetic: assignments and expressions with numpy (np), math and rotation helpers "
                "(quat_to_matrix, matrix_to_quat, quat_from_axis_angle(axis, deg), quat_multiply, rotate(quat, v), "
                "euler_deg_to_quat(r, p, y), quat_to_euler_deg, angle_deg(a, b), dist(a, b)).",
                use_when="any number you derive (offsets, midpoints, rotations, distances): do not compute in your "
                         "head.",
                returns="the named outputs, or the value of the last expression.",
                limits="no loops, imports, definitions or state between calls; quaternions are [x, y, z, w]."),
                obj({"code": {"type": "string", "maxLength": 4000, "description": "lines of assignments and "
                                                                                   "expressions"},
                     "bindings": {"type": "array", "description": "values to bind before running",
                                  "items": obj({"name": {"type": "string", "description": "variable name"},
                                                "value": {"type": ["number", "array"], "description": "a number or "
                                                          "a (nested) list of numbers"}}, ["name", "value"])},
                     "outputs": {"type": "array", "items": {"type": "string", "description": "name"},
                                 "description": "variables to return (default: the last expression)"}}, ["code"]),
                self.calc_tool),
            ToolDef("grasp_options", "read", Help(
                "Grasp poses for a located object that fit the open hand without touching the support or nearby "
                "points (parallel hands: top-down and side grasps; dexterous hands: multi-finger grasps).",
                use_when="before grasp: pick the option whose pose makes sense for the object and the next step.",
                returns="options [{option (handle), grasp {position, quat_xyzw, approach_axis, closing_axis}, "
                        "pre_position, style, reachable (IK), ik_error_mm}].",
                limits="no motion and no collision-free path check (grasp plans that).",
                errors="unknown_target, stale (the robot moved: locate again), no_grasp."),
                obj({"target": {"type": "string", "description": "a target handle from locate"},
                     "max_candidates": {"type": "integer", "minimum": 1, "maximum": 6, "description": "default 3"},
                     **arm_p}, ["target"]), self.grasp_options),
            ToolDef("reach_check", "read", Help(
                "Whether an arm can put its TCP at a pose (inverse kinematics only).",
                use_when="before a move to an unusual pose, or to compare orientations.",
                returns="reachable, position_error_mm, rotation_error_deg, the pose checked.",
                limits="no collision or path check."),
                obj({**pose_p, **arm_p}), self.reach_check),
            ToolDef("report_issue", "read", Help(
                "Record a problem with the tools or the interface (not with the task).",
                effect="none on the world; the message is kept with the run."),
                obj({"message": {"type": "string", "maxLength": 500, "description": "what is wrong"}}, ["message"]),
                self.report_issue),
            ToolDef("move_tcp", "act", Help(
                "Move an arm's TCP to a pose: an absolute position or a delta, and an absolute orientation, a "
                "rotation, or the current orientation.",
                use_when="free: reaching a pose around obstacles (collision-free joint path). straight: a short "
                         "straight line from where the TCP is (approach, lift, push, press, retreat).",
                effect="the robot moves; your turn ends and the next message has the receipt and new images.",
                returns="receipt {status refused | arrived | stopped_short | episode_ended, steps, stages "
                        "[{tcp_error_mm, rotation_error_deg}], tcp_now, holding, why}.",
                limits="straight moves are checked against the support and the robot only, not other objects.",
                errors="refused (no IK / no collision-free path; why says which): nothing moved, it is not counted."),
                obj({**pose_p, "route": {"type": "string", "enum": ["free", "straight"],
                                         "description": "free (default) or straight"}, **arm_p}), self.move_tcp),
            ToolDef("grasp", "act", Help(
                "Execute a grasp option: open the hand, go to the pre-grasp pose, approach in a straight line, close, "
                "lift.",
                use_when="you chose a grasp option from grasp_options at this observation.",
                effect="the robot moves and closes its hand; your turn ends.",
                returns="receipt with stages (approach and close, lift), holding_after_close, holding_after_lift.",
                errors="stale option (locate and grasp_options again), refused (no path)."),
                obj({"option": {"type": "string", "description": "a grasp option handle"},
                     "lift_m": {"type": "number", "minimum": 0, "maximum": 0.3,
                                "description": f"lift after closing (default {self.grasp_lift_m} m)"}}, ["option"]),
                self.grasp),
            ToolDef("place", "act", Help(
                "Put down the object the arm holds (grasped with grasp) so it rests at a point or on another "
                "located object, then open the hand and back away.",
                use_when="the arm is holding the object; for stacking, pass on_target with the object to stack on.",
                effect="the robot moves and releases; your turn ends.",
                returns="receipt with stages (lower and release, retreat).",
                errors="not_holding, refused (no path or no IK)."),
                obj({"at": vec(2, "[x, y] world position where the object should rest"),
                     "on_target": {"type": "string", "description": "a target handle to put the object on"},
                     "yaw_deg": {"type": "number", "minimum": -180, "maximum": 180,
                                 "description": "object yaw about +z in the world (default: keep)"},
                     "hover_m": {"type": "number", "minimum": 0, "maximum": 0.1,
                                 "description": "release height above the support (default 0.01)"}, **arm_p}),
                self.place),
            ToolDef("gripper", "act", Help(
                "Open or close a hand (or go to a named hand preset) without moving the arm.",
                effect="the hand moves for the given time; your turn ends.",
                returns="receipt with holding (closing on an object stops the fingers short: holding true)."),
                obj({"command": {"type": "string", "description": "open, close, or a hand preset name"},
                     "seconds": {"type": "number", "minimum": 0.2, "maximum": 3, "description": "default 1"},
                     **arm_p}, ["command"]), self.gripper),
            ToolDef("recover", "act", Help(
                "Move an arm back to its joint configuration before the last motion (previous) or to its home "
                "posture, on a collision-free path.",
                use_when="a motion went wrong, the arm is in the cameras' way, or you need a clean start.",
                effect="the robot moves; your turn ends."),
                obj({"to": {"type": "string", "enum": ["previous", "home"], "description": "default previous"},
                     **arm_p}), self.recover),
            ToolDef("wait", "act", Help(
                "Hold every arm and hand still while time passes.",
                use_when="something must settle (a rolling object, a door swinging).",
                effect="time passes; your turn ends.", returns="receipt with tcp_drift_mm per arm."),
                obj({"steps": {"type": "integer", "minimum": 1, "maximum": 60, "description": "control steps"}},
                    ["steps"]), self.wait),
            ToolDef("finish", "end", Help(
                "End the episode and say how it went.",
                use_when="success: the evidence you saw shows the task done (say which). give_up: it cannot be "
                         "done (category). abort: a tool or interface failure stopped you.",
                effect="the episode ends; no further calls.",
                limits="does not decide success: the task's own check does."),
                obj({"outcome": {"type": "string", "enum": ["success", "give_up", "abort"], "description": "outcome"},
                     "reason": {"type": "string", "maxLength": 500, "description": "the evidence or the reason"},
                     "category": {"type": "string", "enum": ["unreachable", "object_lost", "scene_damaged",
                                                             "repeated_failure", "other"],
                                  "description": "for give_up"}}, ["outcome", "reason"]), self.finish),
        ]
        return Surface(T, name="robot")


def project(cam: Any, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) of world points in an OpenCV-frame camera and whether each is in front of it."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    Tinv = np.linalg.inv(np.asarray(cam.T_world_cam, float))
    pc = pts @ Tinv[:3, :3].T + Tinv[:3, 3]
    front = pc[:, 2] > 1e-4
    z = np.where(front, pc[:, 2], 1.0)
    K = np.asarray(cam.K, float)
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    return uv, front


__all__ = ["RobotSurface", "calc", "project"]
