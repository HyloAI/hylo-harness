"""The checker for free-form scene programs (`hylo.freescene/v0.1`, env `mujoco.free_scene`).

    report = check_free_scene("runs/scenes_free/cable/scene.py")
    report.ok, report.text()

A free scene has no schema to validate against and no predicate language to reason about, so the checks are
behavioural: build it, run it, and try to satisfy it. Stages, in order (later ones need the earlier ones to pass):

1. program      the source passes the scene-program allow-list (validate_scene_program, below) and defines
                ROBOT, INSTRUCTION, build(scene) and success(sim)
2. load         the environment builds the scene with the robot and compiles it (build errors, MuJoCo compiler
                errors, actuators or keyframes added, randomize errors come back with the program's own lines)
3. physics      per seed: no deep start overlaps (scene-scene, scene-table, scene-robot at its home posture); 2 s of
                doing nothing: the simulation does not diverge, nothing falls off the table, rigid bodies stay
                within 2 cm and 10 degrees, cable nodes and cloth vertices within 5 cm, joints stay put, and nothing is
                still moving at the end; success(sim) runs, returns a bool, is False at the start and stays False
4. reach        on the settled state, with hylo-manip's IK on the robot's RobotSpec: every movable thing (free body,
                cable, cloth, hinged or sliding part) has a point some arm reaches with a top-down or side approach
                and a pre-pose 8 cm back, and so does every point the program's targets(sim) names
5. satisfiable  demo(sim) runs with invisible hands (physics only, the robot holds still) and success(sim) becomes
                True during it or after the hands let go; every point a hand grabbed is reachable by an arm. Without
                a demo the stage is a warning ("unverified")

What it cannot verify: that the success function means what the instruction says (a demo that satisfies a wrong
function passes), that a real gripper (rather than an invisible hand) can hold a thin cable or a cloth corner, and
that both arms can act at once where a task needs two hands at the same time (each point is checked on its own).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hylo.core.registry import registry
from hylo.core.scene import Problem
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Action

from .sandbox import ALLOWED_IMPORTS, DENIED_ATTRS, CodeChecker, CodeRejected, Sandbox
from .scene_check import CheckReport, dedupe, problem, tilt_deg

FREE_ENV = "mujoco.free_scene"
RIGID_MOVE, RIGID_TILT_DEG, SOFT_MOVE = 0.02, 10.0, 0.05      # m, degrees, m
STILL = 0.005                                                 # m over the last half second: "at rest"
DEMO_HANDS = 4

# The same guard runs twice on purpose: here, at build time, before a program reaches an environment, and in the env
# package at load time (hylo.envs.mujoco.free_program.program_problems), for programs written by hand. The boundary
# rules keep the harness and the bench from sharing the code, so keep the two allow-lists in step.
SCENE_IMPORTS = ALLOWED_IMPORTS | {"mujoco"}
SCENE_DENIED_ATTRS = DENIED_ATTRS | {
    "from_file", "to_file", "from_zip", "to_zip", "from_xml_path", "from_binary_path", "mj_saveLastXML",
    "mj_saveModel", "mj_loadXML", "mj_loadModel", "mj_printModel", "mj_printData", "mj_printSchema", "viewer",
    "rollout", "plugin_paths", "mj_loadPluginLibrary", "mj_loadAllPluginLibraries"}


class _SceneChecker(CodeChecker):
    """The robot-program checker with the scene programs' allow-list: mujoco may be imported, the model editor's
    file and plugin-loading calls may not."""

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name not in SCENE_IMPORTS:
                self._bad(node, f"import {a.name} is not allowed (math, numpy, mujoco)")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module not in SCENE_IMPORTS:
            self._bad(node, f"from {node.module} import ... is not allowed (math, numpy, mujoco)")
        for a in node.names:
            if a.name == "*" or a.name.startswith("_") or a.name in SCENE_DENIED_ATTRS:
                self._bad(node, f"cannot import {a.name!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in SCENE_DENIED_ATTRS:
            self._bad(node, f"attribute .{node.attr} is not available")
        self.generic_visit(node)


def validate_scene_program(source: str, filename: str = "scene.py") -> ast.Module:
    """Parse and check a scene program; raises CodeRejected listing the problems."""
    if len(source) > 60_000:
        raise CodeRejected(f"program too long ({len(source)} characters > 60000)")
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise CodeRejected(f"SyntaxError: {exc.msg} (line {exc.lineno})") from None
    c = _SceneChecker()
    c.visit(tree)
    if c.problems:
        raise CodeRejected("rejected: " + "; ".join(c.problems[:8]) + (" ..." if len(c.problems) > 8 else ""))
    return tree



def scene_program_path(source: str | Path) -> Path:
    p = Path(source)
    return p / "scene.py" if p.is_dir() else p


def program_declarations(source: str) -> tuple[dict[str, Any], list[Problem]]:
    """The literal declarations and the defined functions of a scene program (without running it)."""
    try:
        tree = validate_scene_program(source)
    except CodeRejected as exc:
        return {}, [problem("program", str(exc))]
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                found[node.targets[0].id] = ...
        elif isinstance(node, ast.FunctionDef):
            found[node.name] = "function"
    problems = []
    for name in ("ROBOT", "INSTRUCTION"):
        if name not in found:
            problems.append(problem("program", f"the program must set {name} at the top level"))
    for name in ("build", "success"):
        if found.get(name) != "function":
            problems.append(problem("program", f"the program must define a top-level function {name}(...)"))
    robot = found.get("ROBOT")
    if isinstance(robot, str) and robot.lower() not in ("panda", "aloha", "g1"):
        problems.append(problem("program", f"ROBOT must be \"panda\", \"aloha\" or \"g1\", not {robot!r}"))
    steps = found.get("MAX_STEPS", 900)
    if isinstance(steps, int) and not 100 <= steps <= 4000:
        problems.append(problem("program", f"MAX_STEPS = {steps}: use 100 .. 4000 control steps (20 per second)"))
    return found, problems


class FreeSceneChecker:
    """Checks scene programs; one environment per program (built through the registry, cameras off)."""

    def __init__(self, seeds: Sequence[int] = (0, 1, 2), settle_s: float = 2.0, reach: bool = True,
                 satisfiable: bool = True, timeout_s: float = 600.0, env_type: str = FREE_ENV) -> None:
        self.seeds, self.settle_s, self.reach, self.satisfiable = tuple(seeds), settle_s, reach, satisfiable
        self.timeout_s, self.env_type = timeout_s, env_type
        self._env: Any = None
        self._kin: dict[str, Any] = {}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _guard(self, fn: Any) -> Any:
        """Run fn under a wall-clock limit (a scene program could loop forever); its exception is re-raised."""
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["value"] = fn()
            except Exception as exc:  # noqa: BLE001 - re-raised below, outside the guard
                box["error"] = exc

        res = Sandbox(block_timeout_s=self.timeout_s).call(call)
        if res.timed_out:
            raise TimeoutError(f"the scene program ran longer than {self.timeout_s:g} s and was stopped")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _task(self, path: Path, max_steps: int = 900) -> TaskSpec:
        return TaskSpec(id=f"free/{path.parent.name}", version="check", instruction="(check)",
                        params={"scene": str(path)}, max_steps=max_steps)

    # ------------------------------------------------------------------ entry point
    def check(self, source: str | Path) -> CheckReport:
        path = scene_program_path(source).resolve()
        stages = ["program"]
        if not path.is_file():
            return CheckReport(ok=False, problems=(problem("program", f"no scene program at {path}"),), stages=tuple(stages))
        decl, problems = program_declarations(path.read_text())
        if problems:
            return CheckReport(ok=False, problems=tuple(problems), stages=tuple(stages))
        stages.append("load")
        self.close()
        try:
            self._env = self._guard(lambda: registry.build("envs", {"type": self.env_type, "params": {
                "scene": str(path), "cameras": [], "privileged_obs": True, "demo_hands": DEMO_HANDS}}))
            self._guard(lambda: self._env.reset(TaskInstance(spec=self._task(path), seed=self.seeds[0])))
        except Exception as exc:  # noqa: BLE001 - every failure to load is the author's problem
            return CheckReport(ok=False, problems=(problem("load", _load_message(exc)),), stages=tuple(stages))
        stages.append("physics")
        problems, warnings = [], []
        state = None
        for seed in self.seeds:
            try:
                st, probs, warns = self._guard(lambda: self._physics(path, seed))
            except Exception as exc:  # noqa: BLE001
                probs, warns, st = [problem("load", _load_message(exc) + (f" (seed {seed})" if seed else ""))], [], None
            problems += probs
            warnings += warns
            state = state or st
            if any(p.code == "load" for p in probs):
                break
        problems = dedupe(problems)
        settled = summary(state) if state else {}
        if not problems and self.reach and state is not None:
            stages.append("reach")
            try:
                r_probs, r_warns = self._guard(lambda: self._reach(state))
            except Exception as exc:  # noqa: BLE001
                r_probs, r_warns = [problem("load", _load_message(exc))], []
            problems += r_probs
            warnings += r_warns
        if not problems and self.satisfiable:
            stages.append("satisfiable")
            try:
                s_probs = self._guard(lambda: self._satisfiable(path, decl))
            except Exception as exc:  # noqa: BLE001
                s_probs = [problem("demo_error", _load_message(exc))]
            for p in s_probs:
                (problems if p.severity == "error" else warnings).append(p)
        return CheckReport(ok=not problems, problems=tuple(problems), warnings=tuple(dedupe(warnings)),
                           stages=tuple(stages), settled=settled)

    # ------------------------------------------------------------------ physics
    def _physics(self, path: Path, seed: int) -> tuple[dict | None, list[Problem], list[Problem]]:
        env = self._env
        tag = f" (seed {seed})" if seed else ""
        try:
            obs = env.reset(TaskInstance(spec=self._task(path), seed=seed))
        except Exception as exc:  # noqa: BLE001
            return None, [problem("load", _load_message(exc) + tag)], []
        out: list[Problem] = []
        warns: list[Problem] = []
        start = obs.extra["privileged"]
        for c in start.get("start", {}).get("contacts", []):
            if "robot" in (c["a"], c["b"]):
                other = c["b"] if c["a"] == "robot" else c["a"]
                out.append(problem("robot_overlap", f"at the start {other} overlaps the robot at its home posture by "
                                               f"{1000 * c['depth']:.0f} mm{tag}; move it away from the arm(s)", other))
            else:
                out.append(problem("penetration", f"at the start {c['a']} and {c['b']} overlap by {1000 * c['depth']:.0f} mm"
                                             f"{tag}; move them apart or fix their sizes / heights", c["a"], c["b"]))
        for p, cab in start["cables"].items():
            if cab.get("anchored"):
                warns.append(problem("anchored", f"cable {p}: its first body has no joint, so that end is fixed to the world "
                                            "(composite initial=\"none\"); use initial=\"free\" unless it is meant to "
                                            "be tied down", p, severity="warning"))
        hold = np.asarray(env.hold_action(None), float)
        hz = env.spec().control_hz
        steps = max(2, int(round(self.settle_s * hz)))
        mark = None
        import time

        t0 = time.monotonic()
        for k in range(steps):
            r = env.step(Action(data=hold))
            if r.info.get("success_error"):
                out.append(problem("success_error", f"success(sim) failed{tag}: {r.info['success_error']}"))
                break
            if r.info.get("success_type"):
                out.append(problem("success_type", f"success(sim) returned a {r.info['success_type']}, not a bool: return "
                                              "bool(...)"))
            if r.truncated:
                out.append(problem("unstable", f"the simulation diverged while settling{tag} ({r.info.get('sim_error')}); "
                                          "look for overlapping bodies, very light or thin parts, or stiff "
                                          "cables / cloth with too few, too heavy or too long segments"))
                return None, out, warns
            if r.success:
                when = "at the start" if k == 0 else f"after {(k + 1) / hz:.1f} s of doing nothing"
                out.append(problem("success_true_at_start", f"success(sim) is already True {when}{tag}: the robot would "
                                                       "have nothing to do; arrange the start so the goal is not met "
                                                       "(or fix the success function)"))
                break
            if k == steps - 1 - int(0.5 * hz):
                mark = env.privileged_state(json_ready=True)
        real = (time.monotonic() - t0) / (steps / hz)
        if seed == self.seeds[0] and real > 8.0:
            warns.append(problem("slow", f"the scene simulates {real:.0f}x slower than real time on this machine; fewer "
                                    "cable segments / cloth vertices would make solving it much faster",
                            severity="warning"))
        end = env.privileged_state(json_ready=True)
        out += motion_problems(start, end, mark, tag, self.settle_s)
        return end, out, warns

    # ------------------------------------------------------------------ reach
    def _arms(self) -> list[tuple[Any, Any]]:
        if not self._kin:
            from hylo.manip.kinematics import Kinematics

            robot = self._env.spec().robot
            kin = Kinematics(robot)
            self._kin = {a.name: (kin[a.name], a) for a in robot.arms}
        return list(self._kin.values())

    def reachable_by(self, xyz: Sequence[float], back: float = 0.08) -> list[str]:
        """The arms that reach `xyz` with a top-down or side TCP and the pre-pose `back` metres before it."""
        out = []
        rng = np.random.default_rng(0)
        for arm, spec in self._arms():
            base = np.asarray(spec.base_in_world, float)[:3, 3]
            h = np.asarray(xyz, float)[:2] - base[:2]
            side = np.array([h[0], h[1], 0.0]) / (np.linalg.norm(h) or 1.0)
            for a in (np.array([0.0, 0.0, -1.0]), side, (side + np.array([0, 0, -1.0])) / math.sqrt(2)):
                y = np.cross([0.0, 0.0, 1.0], a)
                y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])
                T = np.eye(4)
                T[:3, :3] = np.stack([a, y, np.cross(a, y)], axis=1)
                T[:3, 3] = xyz
                r = arm.ik(T, q_now=np.asarray(spec.home), free="x", rng=rng, n_random=6, pos_tol=0.006, rot_tol=0.08)
                if not r.ok:
                    continue
                P = T.copy()
                P[:3, 3] = np.asarray(xyz) - a * back
                if arm.ik(P, q_now=r.q, free="x", rng=rng, n_random=3, pos_tol=0.006, rot_tol=0.08).ok:
                    out.append(spec.name)
                    break
        return out

    def _reach(self, state: dict) -> tuple[list[Problem], list[Problem]]:
        out: list[Problem] = []
        warn: list[Problem] = []
        top = float(state["table"]["height"])

        def lift(p: Sequence[float]) -> np.ndarray:
            q = np.asarray(p, float).copy()
            q[2] = max(q[2], top + 0.017)
            return q

        zones = "; ".join(f"{a}: x {z['x']}, y {z['y']}" for a, z in state.get("arm_zones", {}).items())
        for name, pts in movable_points(state).items():
            if not any(self.reachable_by(lift(p)) for p in pts):
                c = np.mean(np.asarray(pts, float), axis=0)
                out.append(problem("unreachable", f"no arm reaches any point of {name} (around ({c[0]:.2f}, {c[1]:.2f}, "
                                             f"{c[2]:.2f})) from above or from the side; move it into an arm's zone "
                                             f"({zones})", name))
        try:
            targets = self._env.program_targets()
        except Exception as exc:  # noqa: BLE001
            return out + [problem("targets_error", _load_message(exc))], warn
        for name, p in targets.items():
            if not self.reachable_by(lift(p)):
                out.append(problem("unreachable", f"targets(): no arm reaches {name} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
                              name))
        return dedupe(out), warn

    # ------------------------------------------------------------------ satisfiable
    def _satisfiable(self, path: Path, decl: dict[str, Any]) -> list[Problem]:
        if decl.get("demo") != "function":
            return [problem("unverified", "no demo(sim): nothing showed that the task can be done; add a demo that moves "
                                     "things with the invisible hands until success(sim) is True", severity="warning")]
        env = self._env
        env.reset(TaskInstance(spec=self._task(path), seed=self.seeds[0]))
        hold = np.asarray(env.hold_action(None), float)
        for _ in range(int(env.spec().control_hz)):              # start from the settled scene, as a robot would
            env.step(Action(data=hold))
        res = env.run_demo()
        out: list[Problem] = []
        if res.get("error"):
            return [problem("demo_error", f"demo(sim) failed: {res['error']}")]
        if not (res.get("success_seen") or res.get("success_end")):
            out.append(problem("unsatisfiable", f"success(sim) never became True during demo(sim) ({res.get('steps', 0)} "
                                           "steps) nor after the hands let go: either the demo does not do the task or "
                                           "the success function asks for something else; print values in success "
                                           "to see why"))
        for g in res.get("grabs", []):
            p = np.asarray(g["point"], float)
            p[2] = max(p[2], float(env.privileged_state()["table"]["height"]) + 0.017)
            if not self.reachable_by(p):
                out.append(problem("unreachable", f"demo grabs {g['grab']} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}), where no "
                                             "arm reaches: a robot could not take hold there", g["grab"]))
        return dedupe(out)


def _load_message(exc: BaseException) -> str:
    msg = str(exc)
    return msg[:900] if msg else type(exc).__name__


def movable_points(state: dict) -> dict[str, list[list[float]]]:
    """Sample points of everything a robot could move: free bodies (centre and top), jointed parts (centre),
    cables (both ends and the middle), cloth / flex (corners of its extent and its centre)."""
    out: dict[str, list[list[float]]] = {}
    jointed = {j["body"] for j in state["joints"].values()}
    for name, b in state["bodies"].items():
        if b["free"] or name in jointed:
            lo, hi = np.asarray(b["bounds"][0]), np.asarray(b["bounds"][1])
            c = (lo + hi) / 2
            out[name] = [c.tolist(), [c[0], c[1], hi[2] - 0.01]]
    for p, cab in state["cables"].items():
        n = cab["nodes"]
        out[p] = [n[0], n[len(n) // 2], n[-1]]
    for f, fx in state["flexes"].items():
        v = np.asarray(fx["vertices"], float)
        if len(v) == 0:
            continue
        idx = {int(np.argmin(v[:, 0] + v[:, 1])), int(np.argmax(v[:, 0] + v[:, 1])), int(np.argmin(v[:, 0] - v[:, 1])),
               int(np.argmax(v[:, 0] - v[:, 1]))}
        out[f] = [v[i].tolist() for i in sorted(idx)] + [v.mean(0).tolist()]
    return out


def motion_problems(start: dict, end: dict, mark: dict | None, tag: str, settle_s: float) -> list[Problem]:
    """What moved, fell or drifted between two privileged states of a free scene."""
    out: list[Problem] = []
    tb = end["table"]
    top = float(tb["height"])

    def off_table(p: Sequence[float]) -> bool:
        return p[2] < top - 0.05 or not (tb["x"][0] < p[0] < tb["x"][1] and tb["y"][0] < p[1] < tb["y"][1])

    for name, b in end["bodies"].items():
        s = start["bodies"].get(name)
        if s is None or not b["free"]:
            continue
        moved = float(np.linalg.norm(np.subtract(b["pos"], s["pos"])))
        tilt = tilt_deg(s["quat"], b["quat"])
        if off_table(b["pos"]):
            out.append(problem("fell_off", f"{name} ends off the table after settling{tag}", name))
        elif moved > RIGID_MOVE or tilt > RIGID_TILT_DEG:
            out.append(problem("unstable", f"{name} does not stay where it starts: it moved {100 * moved:.1f} cm and tilted "
                                      f"{tilt:.0f} degrees in {settle_s:g} s{tag}; rest it flat on its support (z = "
                                      "support top + half height) with free space around it", name))
        elif mark is not None and name in mark["bodies"] and \
                np.linalg.norm(np.subtract(b["pos"], mark["bodies"][name]["pos"])) > STILL:
            out.append(problem("unstable", f"{name} is still moving after {settle_s:g} s{tag}", name))
    for kind, key in (("cables", "nodes"), ("flexes", "vertices")):
        for name, c in end[kind].items():
            s = start[kind].get(name)
            if s is None or not c[key]:
                continue
            a, b = np.asarray(s[key], float), np.asarray(c[key], float)
            if a.shape != b.shape:
                continue
            what = "cable" if kind == "cables" else "flex"
            if np.mean([off_table(p) for p in b]) > 0.5:
                out.append(problem("fell_off", f"{what} {name} ends mostly off the table after settling{tag}", name))
                continue
            moved = float(np.max(np.linalg.norm(b - a, axis=1)))
            if moved > SOFT_MOVE:
                out.append(problem("unstable", f"{what} {name} does not stay where it starts: a point moved {100 * moved:.1f}"
                                          f" cm in {settle_s:g} s{tag}; start it resting on its support (a cable a "
                                          "radius above it, cloth flat), not in the air or inside something", name))
            elif mark is not None and name in mark[kind]:
                late = float(np.max(np.linalg.norm(b - np.asarray(mark[kind][name][key], float), axis=1)))
                if late > 2 * STILL:
                    out.append(problem("unstable", f"{what} {name} is still moving after {settle_s:g} s{tag} (a point moved "
                                              f"{1000 * late:.0f} mm in the last 0.5 s); add damping", name))
    for name, j in end["joints"].items():
        s = start["joints"].get(name)
        if s is None:
            continue
        span = (j["range"][1] - j["range"][0]) if j["range"] else (1.0 if j["type"] == "hinge" else 0.1)
        if abs(j["value"] - s["value"]) > max(0.1 * span, 0.005 if j["type"] == "slide" else 0.05):
            out.append(problem("unstable", f"joint {name} drifts from {s['value']:.3f} to {j['value']:.3f} {j['units']} "
                                      f"without being touched{tag}; check its axis, range, springs and what rests on "
                                      "it", name))
    return out


def summary(state: dict) -> dict[str, Any]:
    """A compact picture of the settled scene for the author: bodies (centre, extent), cable ends, flex extents,
    joint values."""
    r = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    out: dict[str, Any] = {}
    for name, b in state["bodies"].items():
        lo, hi = np.asarray(b["bounds"][0]), np.asarray(b["bounds"][1])
        out[name] = {"pos": r((lo + hi) / 2), "extent": r(hi - lo), "free": b["free"]}
    for p, c in state["cables"].items():
        n = c["nodes"]
        out[p] = {"cable_ends": [r(n[0]), r(n[-1])], "nodes": len(n)}
    for f, fx in state["flexes"].items():
        v = np.asarray(fx["vertices"], float)
        if len(v):
            out[f] = {"flex_min": r(v.min(0)), "flex_max": r(v.max(0)), "vertices": len(v)}
    for name, j in state["joints"].items():
        out[f"joint {name}"] = {"value": round(j["value"], 4), "range": j["range"]}
    return out


def check_free_scene(source: str | Path, **kw: Any) -> CheckReport:
    checker = FreeSceneChecker(**kw)
    try:
        return checker.check(source)
    finally:
        checker.close()


__all__ = ["FREE_ENV", "FreeSceneChecker", "SCENE_IMPORTS", "check_free_scene", "motion_problems", "movable_points",
           "program_declarations", "scene_program_path", "summary", "validate_scene_program"]
