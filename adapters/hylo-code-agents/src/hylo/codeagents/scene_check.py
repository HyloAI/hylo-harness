"""The scene checker (RFC-0006 part 8): is a `hylo.scene/v0.1` file a fair task for the robot?

    report = check_scene("scene.json")           # env "mujoco.panda_scene", seeds (0, 1)
    report.ok, [p.message for p in report.problems]

Checks, in order (later ones need the earlier ones to pass):

1. schema      the file parses and validates against hylo.core.scene.Scene
2. static      names and labels unique, references resolve, no attachment cycles, the predicate can hold at all
               (hylo.core.scene.scene_problems)
3. physics     per seed, through the Environment contract with privileged observations: the scene loads; start
               poses fit (no free position in a region, overlaps deeper than 3 mm); after settling 2 s nothing fell off,
               nothing moved more than 2 cm or tilted more than 10 degrees, joints stayed put, nothing penetrates
               (objects, table, robot); the success predicate is false at the start and stays false while settling
4. reach       on the settled state, with hylo-manip's IK on the env's RobotSpec: every movable object has a top-down
               (or side) grasp the arm reaches, with a pre-grasp 10 cm back, and fits the open hand; every jointed
               part is reachable at its centre and at the end of its travel; every place target (b of on / inside /
               above, inside regions, near points) is reachable 5 cm above its top

Problems are hylo.core.scene.Problem records (code, message, objects, severity); warnings do not fail a scene.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hylo.core.contract import Contract
from hylo.core.registry import registry
from hylo.core.scene import (AllOf, Above, AnyOf, Inside, JointAtLeast, JointAtMost, Lifted, Near, On, Problem, Row,
                             Scene, Sequence as Seq, Upright, referenced, scene_problems, validation_problems, walk)
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Action

DEFAULT_ENV = "mujoco.panda_scene"
MOVE_TOL, TILT_TOL_DEG, JOINT_TOL = 0.02, 10.0, 0.1       # metres, degrees, fraction of the joint range
GRASP_MARGIN = 0.01                                       # clearance inside the open hand (m)


class CheckReport(Contract):
    ok: bool
    problems: tuple[Problem, ...] = ()
    warnings: tuple[Problem, ...] = ()
    stages: tuple[str, ...] = ()                          # the stages that ran
    settled: dict[str, Any] = {}                          # object -> settled pose summary (seed 0)

    def text(self, limit: int = 30) -> str:
        """The problems as a numbered list, for a model or a terminal."""
        lines = [f"{i + 1}. [{p.code}] {p.message}" for i, p in enumerate(self.problems[:limit])]
        lines += [f"- warning [{p.code}] {p.message}" for p in self.warnings[:10]]
        return "\n".join(lines) or "no problems"


def problem(code: str, message: str, *objects: str, severity: str = "error") -> Problem:
    """A checker Problem (shared by the fixed-format and the free checker)."""
    return Problem(code=code, message=message, objects=tuple(objects), severity=severity)  # type: ignore[arg-type]



def parse_scene(source: str | Path | dict | Scene) -> tuple[Scene | None, list[Problem]]:
    """A Scene from a path, JSON text, a mapping or a Scene; or the schema problems."""
    if isinstance(source, Scene):
        return source, []
    try:
        if isinstance(source, dict):
            return Scene.model_validate(source), []
        text = str(source)
        if not text.lstrip().startswith("{"):
            path = Path(text)
            if path.suffix in (".yaml", ".yml"):
                import yaml

                return Scene.model_validate(yaml.safe_load(path.read_text())), []
            text = path.read_text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, [problem("json", f"not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}")]
        return Scene.model_validate(data), []
    except Exception as exc:  # noqa: BLE001 - every failure is a problem for the author
        return None, validation_problems(exc)


def quat_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(q0: Sequence[float], q1: Sequence[float]) -> float:
    """The angle (degrees) between two orientations given as (w, x, y, z) quaternions."""
    d = abs(float(np.dot(q0, q1)))
    return math.degrees(2 * math.acos(min(1.0, d)))


class SceneChecker:
    """Runs the checks; holds one environment (built through the registry, cameras off, privileged on)."""

    def __init__(self, env_type: str = DEFAULT_ENV, seeds: Sequence[int] = (0, 1), settle_s: float = 2.0,
                 reach: bool = True, satisfiable: bool = True) -> None:
        self.env_type, self.seeds, self.settle_s, self.reach = env_type, tuple(seeds), settle_s, reach
        self.satisfiable = satisfiable
        self._env: Any = None
        self._kin: Any = None

    def env(self) -> Any:
        if self._env is None:
            self._env = registry.build("envs", {"type": self.env_type,
                                                "params": {"cameras": [], "privileged_obs": True}})
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------ entry point
    def check(self, source: str | Path | dict | Scene, path: str | Path | None = None) -> CheckReport:
        """Check a scene. The physics stages need it on disk: `path` (default: `source` when it is a path)."""
        scene, problems = parse_scene(source)
        stages = ["schema"]
        if scene is None:
            return CheckReport(ok=False, problems=tuple(problems), stages=tuple(stages))
        stages.append("static")
        problems = scene_problems(scene)
        errors = [p for p in problems if p.severity == "error"]
        if errors:
            return CheckReport(ok=False, problems=tuple(errors), stages=tuple(stages))
        if path is None:
            if isinstance(source, (str, Path)) and not str(source).lstrip().startswith("{"):
                path = source
            else:
                raise ValueError("check: the physics stages need the scene file; pass `path`")
        stages.append("physics")
        settled: dict[str, Any] = {}
        warnings: list[Problem] = []
        state = None
        for seed in self.seeds:
            st, probs, warns = self._physics(scene, Path(path), seed)
            problems += probs
            warnings += warns
            if state is None:
                state = st
            if any(p.code == "load" for p in probs):
                break
        problems = dedupe(problems)
        if state is not None:
            settled = {n: {"pos": [round(v, 3) for v in o["pos"]], "extent": [round(v, 3) for v in o.get("extent", [])]}
                       for n, o in state["objects"].items()}
        if not problems and self.reach and state is not None:
            stages.append("reach")
            r_probs, r_warns = self._reach(scene, state)
            problems += r_probs
            warnings += r_warns
        if not problems and self.satisfiable and state is not None and hasattr(self.env(), "arrange"):
            stages.append("satisfiable")
            for p in self._satisfiable(scene, Path(path), state):
                (problems if p.severity == "error" else warnings).append(p)
        return CheckReport(ok=not problems, problems=tuple(problems), warnings=tuple(dedupe(warnings)),
                           stages=tuple(stages), settled=settled)

    # ------------------------------------------------------------------ physics
    def _physics(self, scene: Scene, path: Path, seed: int) -> tuple[dict | None, list[Problem], list[Problem]]:
        env = self.env()
        task = TaskSpec(id=f"scene/{scene.name}", version="check", instruction=scene.instruction,
                        params={"scene": str(path.resolve())}, max_steps=scene.max_steps)
        try:
            obs = env.reset(TaskInstance(spec=task, seed=seed))
        except Exception as exc:  # noqa: BLE001 - a scene that does not compile is the author's problem
            return None, [problem("load", f"the scene does not load in {self.env_type}: {str(exc)[:300]}")], []
        out: list[Problem] = []
        start = obs.extra["privileged"]
        tag = f" (seed {seed})" if seed else ""
        for msg in start.get("placement_problems", []):
            out.append(problem("placement", msg + tag, msg.split(":")[0]))
        for c in start.get("start_contacts", []):
            out.append(problem("penetration", f"at the start {c['a']} and {c['b']} overlap by {1000 * c['depth']:.0f} mm"
                                         f"{tag}; move them apart or fix their sizes / heights", c["a"], c["b"]))
        hold = np.concatenate([np.asarray(obs.state["q.arm"], float), [1.0]])
        steps = max(1, int(round(self.settle_s * env.spec().control_hz)))
        true_at = None
        fail = None
        warns: list[Problem] = []
        for k in range(steps):
            r = env.step(Action(data=hold))
            if r.success and true_at is None:
                true_at = k
            if k == 0 and not r.success and r.info.get("done", 0) > 0:
                if isinstance(scene.success, Seq):
                    out.append(problem("predicate_true_at_start", f"the first step of the sequence already holds at the "
                                                             f"start{tag}, so it asks the robot nothing; make every "
                                                             "step something the robot has to change"))
                else:
                    warns.append(problem("partly_true", f"{r.info['done']} of {r.info['of']} terms of the predicate already "
                                                   f"hold at the start{tag}", severity="warning"))
            if r.info.get("fail_reason") and fail is None:
                fail = r.info["fail_reason"]
            if r.truncated:
                out.append(problem("unstable", f"the simulation diverged while settling{tag} ({r.info.get('sim_error')}); "
                                          "look for overlapping or very light / thin objects"))
                return None, out, []
            if r.terminated:
                break
        end = env.privileged_state(json_ready=True)
        if true_at is not None:
            out.append(problem("predicate_true_at_start", f"the success predicate already holds {'at the start' if true_at == 0 else f'after {true_at + 1} steps of doing nothing'}{tag}: "
                          "the robot would have nothing to do; place the objects so the goal is not met initially"))
        if fail:
            out.append(problem("fell_off", f"while settling: {fail}{tag}"))
        placed = start.get("placed", {})
        for name, o in end["objects"].items():
            s = {**start["objects"][name], **placed.get(name, {})}       # as placed, before the env's own settling
            lo, tb = o["aabb"][0], end["table"]
            if o["movable"]:
                moved = float(np.linalg.norm(np.subtract(o["pos"], s["pos"])))
                tilt = tilt_deg(s["quat"], o["quat"])
                cx, cy = o["pos"][:2]
                if o["pos"][2] < tb["height"] - 0.03 or not (tb["x"][0] < cx < tb["x"][1] and tb["y"][0] < cy < tb["y"][1]):
                    out.append(problem("fell_off", f"{name} ends off the table after settling{tag}", name))
                elif moved > MOVE_TOL or tilt > TILT_TOL_DEG:
                    out.append(problem("unstable", f"{name} does not stay where it starts: it moved {100 * moved:.1f} cm and "
                                              f"tilted {tilt:.0f} degrees in {self.settle_s:g} s{tag}; give it a stable "
                                              f"resting pose (z omitted, flat side down) and free space", name))
                elif not o.get("at_rest", True):
                    out.append(problem("unstable", f"{name} is still moving after {self.settle_s:g} s{tag}", name))
            if "joint" in o:
                j0, j1 = scene.object(name).joint.initial, o["joint"]["value"]
                span = o["joint"]["range"][1] - o["joint"]["range"][0]
                if abs(j1 - j0) > JOINT_TOL * span:
                    out.append(problem("unstable", f"the joint of {name} drifts from {j0:.3f} to {j1:.3f} {o['joint']['units']} "
                                              f"without being touched{tag}; check its axis, range and spring", name))
            if lo[2] < tb["height"] - 0.01 and not (o["movable"] and o["pos"][2] < tb["height"] - 0.03):
                out.append(problem("penetration", f"{name} reaches {100 * (tb['height'] - lo[2]):.1f} cm into the table{tag}",
                              name))
        for c in end.get("penetration", []):
            out.append(problem("penetration", f"after settling {c['a']} and {c['b']} interpenetrate by "
                                         f"{1000 * c['depth']:.0f} mm{tag}", c["a"], c["b"]))
        return end, out, warns

    # ------------------------------------------------------------------ reach
    def _arm(self) -> tuple[Any, Any]:
        if self._kin is None:
            from hylo.manip.kinematics import Kinematics

            robot = self.env().spec().robot
            if robot is None or not robot.arms:
                raise RuntimeError(f"{self.env_type} publishes no RobotSpec: cannot check reach")
            self._kin = (Kinematics(robot), robot)
        kin, robot = self._kin
        return kin[robot.arms[0].name], robot.arms[0]

    def _reachable(self, xyz: Sequence[float], approaches: Sequence[np.ndarray], back: float = 0.10) -> bool:
        """Some approach direction for which the TCP reaches `xyz` and the point `back` metres before it."""
        arm, spec = self._arm()
        rng = np.random.default_rng(0)
        for a in approaches:
            a = np.asarray(a, float) / np.linalg.norm(a)
            y = np.cross([0.0, 0.0, 1.0], a)
            y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])
            T = np.eye(4)
            T[:3, :3] = np.stack([a, y, np.cross(a, y)], axis=1)
            T[:3, 3] = xyz
            r = arm.ik(T, q_now=np.asarray(spec.home), free="x", rng=rng, n_random=8, pos_tol=0.004, rot_tol=0.05)
            if not r.ok:
                continue
            P = T.copy()
            P[:3, 3] = np.asarray(xyz) - a * back
            if arm.ik(P, q_now=r.q, free="x", rng=rng, n_random=4, pos_tol=0.004, rot_tol=0.05).ok:
                return True
        return False

    def _reach(self, scene: Scene, state: dict) -> tuple[list[Problem], list[Problem]]:
        out: list[Problem] = []
        warn: list[Problem] = []
        objs = state["objects"]
        _, spec = self._arm()
        opening = spec.hand.max_width - GRASP_MARGIN
        base = np.asarray(spec.base_in_world)[:3, 3]
        down = np.array([0.0, 0.0, -1.0])
        must_lift = {n.a for n in walk(scene.success) if isinstance(n, (On, Inside, Above, Lifted))}
        mentioned = {name for name, _ in referenced(scene.success)}

        def side(c: np.ndarray) -> np.ndarray:
            h = c[:2] - base[:2]
            return np.array([h[0], h[1], 0.0]) / (np.linalg.norm(h) or 1.0)

        for o in scene.objects:
            rec = objs[o.name]
            lo, hi = np.asarray(rec["aabb"][0]), np.asarray(rec["aabb"][1])
            c = (lo + hi) / 2
            if o.movable:
                ext = np.asarray(rec.get("extent") or (hi - lo))
                up = int(np.argmax(np.abs(quat_matrix(rec["quat"])[2])))          # the object axis nearest vertical
                width = float(min(e for k, e in enumerate(ext) if k != up))      # narrowest horizontal side
                height = hi[2] - lo[2]
                if width > opening:
                    msg = (f"{o.name} is {width:.3f} m across at its narrowest; the open hand takes at most "
                           f"{opening:.3f} m")
                    if o.name in must_lift:
                        out.append(problem("too_wide", msg + ", but the predicate needs it picked up; make it narrower", o.name))
                    elif o.name in mentioned:
                        warn.append(problem("too_wide", msg + " (it can only be pushed)", o.name, severity="warning"))
                    continue
                grip = np.array([c[0], c[1], hi[2] - min(0.03, height / 2)])
                if not self._reachable(grip, [down, side(c)]):
                    out.append(problem("unreachable", f"{o.name} at ({c[0]:.2f}, {c[1]:.2f}) cannot be grasped: the arm "
                                                 f"reaches neither a top-down nor a side grasp there; move it towards "
                                                 f"x in [-0.27, 0.2], y in [-0.32, 0.32]", o.name))
            elif o.joint is not None:
                targets = [("its centre", c)]
                if o.joint.type == "slide":
                    j = rec["joint"]
                    axis = np.asarray(o.joint.axis, float) / np.linalg.norm(o.joint.axis)
                    parent_q = objs[o.parent]["quat"]
                    axis_w = quat_matrix(parent_q) @ axis
                    targets.append(("the end of its travel", c + axis_w * (j["range"][1] - j["value"])))
                for what, t in targets:
                    if not self._reachable(np.array([t[0], t[1], max(t[2], hi[2] - 0.01)]), [down, side(t), -side(t)],
                                           back=0.06):
                        out.append(problem("unreachable", f"the arm cannot reach {o.name} at {what} "
                                                     f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})", o.name))
        for node in walk(scene.success):
            target = None
            if isinstance(node, (On, Above)) and node.b != "table":
                lo, hi = objs[node.b]["aabb"]
                target, what = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.05]), f"above {node.b}"
            elif isinstance(node, Inside):
                if node.b is not None:
                    lo, hi = objs[node.b]["aabb"]
                    target, what = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.05]), f"above {node.b}"
                else:
                    r = node.region
                    target, what = np.array([sum(r.x) / 2, sum(r.y) / 2, 0.08]), "the region's centre"
            elif isinstance(node, Near) and node.point is not None:
                target, what = np.array([node.point[0], node.point[1], 0.08]), f"the point {list(node.point)}"
            if target is not None and not self._reachable(target, [down], back=0.05):
                out.append(problem("unreachable", f"{node.op}: the arm cannot bring an object to {what} "
                                             f"({target[0]:.2f}, {target[1]:.2f})", *([node.b] if getattr(node, 'b', None) else [])))
        return dedupe(out), dedupe(warn)


    # ------------------------------------------------------------------ satisfiable (privileged arrangement)
    def _satisfiable(self, scene: Scene, path: Path, state: dict) -> list[Problem]:
        """Teleport the movable objects to where the predicate asks (env.arrange, privileged), let them settle and
        test each arranged term (env.holds). Spring-loaded joints undo an arrangement at once: their terms are not
        judged here."""
        sprung = {n.object for n in walk(scene.success) if isinstance(n, (JointAtLeast, JointAtMost))
                  and scene.object(n.object).joint is not None and scene.object(n.object).joint.spring > 0}
        stages = [[t] for t in scene.success.terms] if isinstance(scene.success, Seq) else [[scene.success]]
        unverified = [problem("unverified", "the predicate could not be checked by arranging objects (it has lifted, not "
                                       "or nested sequence terms, or a subject that cannot move)", severity="warning")]
        if any(arrangement(terms, state["objects"], scene, skip=sprung) is None for terms in stages):
            return unverified
        env = self.env()
        if not hasattr(env, "holds"):
            return []
        task = TaskSpec(id=f"scene/{scene.name}", version="check", instruction=scene.instruction,
                        params={"scene": str(path.resolve())}, max_steps=scene.max_steps)
        obs = env.reset(TaskInstance(spec=task, seed=self.seeds[0]))
        hold = np.concatenate([np.asarray(obs.state["q.arm"], float), [1.0]])
        inside = {(n.a, n.b) for n in walk(scene.success) if isinstance(n, Inside) and n.b is not None}
        for terms in stages:                           # each stage planned from where the previous one left things
            plan = arrangement(terms, env.privileged_state(json_ready=True)["objects"], scene, skip=sprung)
            if plan is None:
                return unverified
            poses, joints = plan
            env.arrange(poses, joints)
            fail = None
            for _ in range(int(1.5 * env.spec().control_hz)):
                r = env.step(Action(data=hold))
                fail = fail or r.info.get("fail_reason")
                if r.terminated:
                    break
            end = env.privileged_state(json_ready=True)
            solid = [c for c in end.get("penetration", []) if (c["a"], c["b"]) in inside or (c["b"], c["a"]) in inside]
            if solid:
                c = solid[0]
                a, b = (c["a"], c["b"]) if (c["a"], c["b"]) in inside else (c["b"], c["a"])
                return [problem("unsatisfiable", f"inside({a}, {b}): {b} is solid where {a} has to go (placed there, {a} "
                                            f"penetrates it); make {b} hollow: a tray or cup, or parts with walls and "
                                            f"a floor (a drawer is a tray on a slide joint)", a, b)]
            failed = [t for t in leaves(terms) if not (getattr(t, "object", None) in sprung)
                      and not env.holds(t.model_dump(mode="json"))]
            if not failed:
                continue
            moved = sorted(poses)
            where = "; ".join(f"{n} ended at {[round(v, 3) for v in end['objects'][n]['pos']]}" for n in moved)
            t = failed[0]
            msg = (f"{t.op}({', '.join(str(v) for k, v in t.model_dump().items() if k in ('a', 'b', 'object', 'objects') and v)})"
                   f" does not hold even with the objects placed where the predicate asks (privileged arrangement"
                   f"{': ' + where if where else ''})")
            messy = fail or any(c["depth"] > 0.005 and ({c["a"], c["b"]} & set(moved)) for c in end.get("penetration", []))
            if messy:
                return [problem("unverified", msg + "; the arrangement itself collided, so this is not judged",
                           severity="warning")]
            return [problem("unsatisfiable", msg + ": check sizes, that containers / drawers are hollow and open far "
                                             "enough, and the predicate itself")]
        return []


def leaves(terms: Sequence[Any]) -> list[Any]:
    """The atomic terms an arrangement aims at (all_of flattened, the first option of any_of)."""
    out = []
    for t in terms:
        if isinstance(t, AllOf):
            out += leaves(t.terms)
        elif isinstance(t, AnyOf):
            out += leaves(t.terms[:1])
        else:
            out.append(t)
    return out


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _qmul(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def arrangement(terms: Sequence[Any], objs: dict[str, Any], scene: Scene,
                skip: set[str] = frozenset()) -> tuple[dict, dict] | None:
    """Poses {name: (pos, quat)} and joint values that should make `terms` hold, from the settled state `objs`
    (privileged objects). None when a term cannot be arranged (lifted, not, a non-movable subject)."""
    rel: dict[str, tuple] = {}
    up: dict[str, str] = {}
    rows: list[Row] = []
    joints: dict[str, float] = {}

    def visit(p: Any) -> bool:
        if isinstance(p, AllOf):
            return all(visit(t) for t in p.terms)
        if isinstance(p, AnyOf):
            return visit(p.terms[0])
        if isinstance(p, (On, Above)):
            rel[p.a] = ("on", p.b)
        elif isinstance(p, Inside):
            rel[p.a] = ("inside", p.b, p.region)
        elif isinstance(p, Near):
            rel[p.a] = ("near", p.b, p.point, p.max_dist)
        elif isinstance(p, Upright):
            up[p.a] = p.axis
        elif isinstance(p, Row):
            rows.append(p)
        elif isinstance(p, (JointAtLeast, JointAtMost)):
            if p.object in skip:
                return True
            lo, hi = objs[p.object]["joint"]["range"]
            v = p.value if p.value is not None else lo + p.fraction * (hi - lo)
            inward = 0.05 * (hi - lo) * (1 if isinstance(p, JointAtLeast) and not isinstance(p, JointAtMost) else -1)
            joints[p.object] = float(np.clip(v + inward, lo, hi))
        else:
            return False
        return True

    if not all(visit(t) for t in terms):
        return None
    moved = set(rel) | set(up) | {n for r in rows for n in r.objects}
    if any(not objs[n]["movable"] for n in moved):
        return None
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def orient(n: str) -> np.ndarray:
        q = np.asarray(objs[n]["quat"], float)
        if n not in up:
            return q
        tilt = {"z": [1.0, 0, 0, 0], "x": [math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0],
                "y": [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]}[up[n]]
        return _qmul(_yaw_quat(float(objs[n]["yaw"])), tilt)

    def offsets(n: str, q: np.ndarray) -> tuple[np.ndarray, float, float]:
        """(xy of the box centre, height below the origin, half height) of n at orientation q."""
        lo, hi = np.asarray(objs[n]["local_box"][0]), np.asarray(objs[n]["local_box"][1])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        c = corners @ quat_matrix(q).T
        return c[:, :2].mean(0), float(-c[:, 2].min()), float((c[:, 2].max() - c[:, 2].min()) / 2)

    def shift(n: str) -> np.ndarray:
        """How far n moves when the arranged joint values are applied (slide joints of n and its parents)."""
        out, cur = np.zeros(3), scene.object(n)
        while cur.parent is not None:
            if cur.name in joints and cur.joint is not None:
                if cur.joint.type != "slide":
                    raise LookupError(cur.name)
                axis = np.asarray(cur.joint.axis, float) / np.linalg.norm(cur.joint.axis)
                out += quat_matrix(objs[cur.parent]["quat"]) @ axis * (joints[cur.name] - objs[cur.name]["joint"]["value"])
            cur = scene.object(cur.parent)
        return out

    def box_of(n: str) -> tuple[np.ndarray, np.ndarray]:
        if n not in poses and not objs[n]["movable"]:
            d = shift(n)
            return np.asarray(objs[n]["aabb"][0]) + d, np.asarray(objs[n]["aabb"][1]) + d
        if n in poses:
            pos, q = poses[n]
            lo, hi = np.asarray(objs[n]["local_box"][0]), np.asarray(objs[n]["local_box"][1])
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            c = corners @ quat_matrix(q).T + pos
            return c.min(0), c.max(0)
        return np.asarray(objs[n]["aabb"][0]), np.asarray(objs[n]["aabb"][1])

    def put(n: str, xy: Sequence[float], floor: float) -> None:
        q = orient(n)
        cxy, below, _ = offsets(n, q)
        poses[n] = (np.array([xy[0] - cxy[0], xy[1] - cxy[1], floor + below + 0.002]), q)

    def place(n: str, depth: int = 0) -> None:
        if n in poses or depth > 8:
            return
        r = rel.get(n)
        if r is None:
            put(n, (np.asarray(objs[n]["aabb"][0][:2]) + np.asarray(objs[n]["aabb"][1][:2])) / 2, 0.0)
            return
        if r[0] == "on":
            if r[1] == "table":
                put(n, np.asarray(objs[n]["pos"][:2]), 0.0)
                return
            if r[1] in rel or r[1] in up:
                place(r[1], depth + 1)
            lo, hi = box_of(r[1])
            put(n, (lo[:2] + hi[:2]) / 2, float(hi[2]))
        elif r[0] == "inside":
            if r[2] is not None:
                reg = r[2]
                put(n, (sum(reg.x) / 2, sum(reg.y) / 2), reg.z[0] if reg.z else 0.0)
                return
            b = objs[r[1]]
            if "interior" in b:
                c, h = np.asarray(b["interior"]["centre"]), np.asarray(b["interior"]["half"])
                c = c + (shift(r[1]) if not b["movable"] else 0.0)
                put(n, c[:2], float(c[2] - h[2]))
            else:
                lo, hi = box_of(r[1])
                put(n, (lo[:2] + hi[:2]) / 2, float(lo[2]) + 0.008)
        else:
            _, b, point, dist = r
            if point is not None:
                put(n, point[:2], 0.0)
                return
            lo, hi = box_of(b)
            centre = (lo[:2] + hi[:2]) / 2
            away = np.asarray(objs[n]["pos"][:2]) - centre
            away = away / (np.linalg.norm(away) or 1.0)
            need = float(np.max(hi[:2] - lo[:2]) / 2 + np.max(objs[n]["extent"][:2]) / 2 + 0.005)
            if need <= 0.8 * dist:
                put(n, centre + away * need, 0.0)
            else:
                put(n, centre, float(hi[2]))

    for r in rows:
        k, j = "xy".index(r.axis), 1 - "xy".index(r.axis)
        cs = np.array([objs[n]["pos"] for n in r.objects])
        gap = max(float(np.max(objs[n]["extent"][:2])) for n in r.objects) + max(0.03, r.min_gap)
        start = cs[:, k].mean() - gap * (len(r.objects) - 1) / 2
        for i, n in enumerate(r.objects):
            xy = np.zeros(2)
            xy[k], xy[j] = start + i * gap, cs[:, j].mean()
            put(n, xy, 0.0)
    try:
        for n in list(rel) + list(up):
            place(n)
    except LookupError:                                  # a hinge moves the target: not arranged here
        return None
    return poses, joints


def dedupe(problems: list[Problem]) -> list[Problem]:
    """The problems without repeats (the same code and message on another seed counts as a repeat)."""
    seen, out = set(), []
    for p in problems:
        key = (p.code, re.sub(r" \(seed \d+\)", "", p.message))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def check_scene(source: str | Path | dict | Scene, path: str | Path | None = None, **kw: Any) -> CheckReport:
    checker = SceneChecker(**kw)
    try:
        return checker.check(source, path)
    finally:
        checker.close()


__all__ = ["CheckReport", "SceneChecker", "check_scene", "dedupe", "parse_scene", "problem", "tilt_deg"]
