"""Test doubles for code agents: a toy manipulation environment and a toolkit with the hylo.manip `Tools` signatures.

Not an evaluation environment (the real ones are hylo-bench's mujoco/manip-v0). The toy robot is Cartesian: its
three "joints" are the TCP position, so a plan is a straight line and no kinematics are needed; the hand is a
parallel gripper that stops half closed on a block (so `holding` works as on a real hand). Blocks and a bin lie on a
table at z = 0; the task is "put the <colour> block in the bin". The toy observation carries object positions in
`extra["objects"]` (perception for the fake toolkit), so programs written against the toolkit verbs run unchanged.

    env:       {type: "hylo.codeagents.fakes:ToyManipEnv"}
    benchmark: {type: "hylo.codeagents.fakes:toy_benchmark", params: {protocol: rsi}}
    tools:     {type: "hylo.codeagents.fakes:FakeTools"}          (code.llm / code.program params)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hylo.core.benchmark import Benchmark, rsi_splits, standard_splits
from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation, ObservationSpec,
                             StepResult)
from hylo.manip.tools import Box, CameraView, Detection, Goal, Mask, Plan, Table, Tools, View

LOW, HIGH = (-0.4, -0.4, 0.0), (0.4, 0.4, 0.4)
HOME = (0.0, 0.0, 0.3)
BLOCK = 0.04                       # block edge, metres
BIN = 0.06                         # half width of the bin's square
SPEED = 0.05                       # max TCP travel per control step
IMG = 48
COLOURS = {"red": (220, 40, 40), "blue": (40, 80, 220), "yellow": (230, 200, 40)}


def toy_robot() -> RobotSpec:
    hand = HandSpec(kind="parallel", open=(1.0,), closed=(0.0,), max_width=0.08)
    arm = ArmSpec(name="arm", model="toy://cartesian", joints=("x", "y", "z"), ee_frame="tcp", lower=LOW, upper=HIGH,
                  home=HOME, hand=hand, action_slice=(0, 3), hand_slice=(3, 4))
    return RobotSpec(arms=(arm,), cameras=(CameraSpec(name="top", height=IMG, width=IMG, depth=False),),
                     table_height=0.0, notes="Toy Cartesian robot (a test double): the joints are the TCP position "
                     "x, y, z in metres; the gripper points down. Blocks are 4 cm cubes on the table.")


class ToyManipEnv:
    def __init__(self, terminate_on_success: bool = False) -> None:
        self.terminate_on_success = terminate_on_success
        self._task: TaskInstance | None = None

    def spec(self) -> EnvSpec:
        return EnvSpec(
            name="testing.toy_manip", version="0.1",
            observation=ObservationSpec(state={"q.arm": (3,), "hand.arm": (1,)}, cameras={"top": (IMG, IMG, 3)},
                                        units={"q.arm": "m", "hand.arm": "open fraction"}),
            action=BoxSpace(shape=(4,), low=(*LOW, 0.0), high=(*HIGH, 1.0),
                            semantics=ActionSemantics(control_mode="joint_pos", frame="base",
                                                      dim_labels=("x", "y", "z", "hand"))),
            control_hz=20.0, capabilities=EnvCapabilities(seedable=True, privileged_state=True), robot=toy_robot())

    def reset(self, task: TaskInstance) -> Observation:
        self._task = task
        rng = np.random.default_rng(task.seed)
        spots: list[np.ndarray] = []
        while len(spots) < 1 + len(COLOURS):
            p = rng.uniform(-0.3, 0.3, size=2)
            if all(np.linalg.norm(p - q) > 0.14 for q in spots):
                spots.append(p)
        self.bin = spots[0]
        self.blocks = {c: np.array([*spots[i + 1], BLOCK / 2]) for i, c in enumerate(COLOURS)}
        self.q, self.hand, self.held = np.array(HOME), 1.0, None
        self.t = 0
        return self._observe()

    def _render(self) -> np.ndarray:
        img = np.full((IMG, IMG, 3), 90, np.uint8)

        def px(xy: np.ndarray) -> tuple[int, int]:
            u = int((xy[0] - LOW[0]) / (HIGH[0] - LOW[0]) * (IMG - 1))
            v = int((HIGH[1] - xy[1]) / (HIGH[1] - LOW[1]) * (IMG - 1))
            return v, u

        v, u = px(self.bin)
        img[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4] = (40, 160, 60)
        for c, p in self.blocks.items():
            v, u = px(p)
            img[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = COLOURS[c]
        v, u = px(self.q)
        img[v, u] = (255, 255, 255)
        return img

    def _observe(self) -> Observation:
        objects = {f"{c} block": p.round(4).tolist() for c, p in self.blocks.items()}
        objects["bin"] = [*self.bin.round(4).tolist(), 0.0]
        return Observation(state={"q.arm": self.q.copy(), "hand.arm": np.array([self.hand])}, images={"top": self._render()},
                           instruction=self._task.spec.instruction, time=self.t / 20.0,  # type: ignore[union-attr]
                           extra={"objects": objects})

    def _success(self) -> bool:
        c = self._task.spec.params["object"]  # type: ignore[union-attr]
        p = self.blocks[c]
        return self.held != c and p[2] < BLOCK and bool(np.all(np.abs(p[:2] - self.bin) <= BIN))

    def step(self, action: Action) -> StepResult:
        a = np.asarray(action.data, dtype=float)
        self.q = self.q + np.clip(a[:3] - self.q, -SPEED, SPEED)
        cmd = float(a[3])
        if self.held is None and cmd < self.hand:                       # closing: grab a block between the pads
            for c, p in self.blocks.items():
                if np.linalg.norm(p[:2] - self.q[:2]) < 0.02 and abs(p[2] - self.q[2]) < 0.02:
                    self.held = c
                    break
        floor = 0.5 if self.held is not None else 0.0                   # pads stop on a held block
        self.hand = float(np.clip(self.hand + np.clip(cmd - self.hand, -0.5, 0.5), floor, 1.0))
        if self.held is not None and cmd > 0.7:                         # opening: the block drops to the table
            p = self.blocks[self.held]
            self.blocks[self.held] = np.array([p[0], p[1], BLOCK / 2])
            self.held, self.hand = None, max(self.hand, 0.75)
        if self.held is not None:
            self.blocks[self.held] = self.q.copy()
        self.t += 1
        ok = self._success()
        return StepResult(observation=self._observe(), reward=float(ok), terminated=ok and self.terminate_on_success,
                          truncated=False, success=ok)

    def privileged_state(self) -> dict[str, Any]:
        return {"blocks": {c: p.tolist() for c, p in self.blocks.items()}, "bin": self.bin.tolist(), "held": self.held}

    def close(self) -> None:
        pass


def toy_tasks(max_steps: int = 300) -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(id=f"testing/toy_put_{c}", version="1", instruction=f"put the {c} block in the bin",
                          params={"object": c}, max_steps=max_steps, tags=("toy", "pick_place"))
                 for c in ("red", "blue"))


def toy_benchmark(protocol: str = "rsi", n_states: int = 40, bounds: Sequence[int] = (6, 12, 26),
                  max_steps: int = 300) -> Benchmark:
    splits = rsi_splits(n_states, tuple(bounds)) if protocol == "rsi" else standard_splits(4, 8, 8)  # type: ignore[arg-type]
    return Benchmark(id="testing/toy-manip-v0", version="0.1", tasks=toy_tasks(max_steps), splits=splits)


def _pose(p: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = (0, 0, -1), (0, 1, 0), (1, 0, 0)    # approach (+x of the TCP) points down
    T[:3, 3] = p
    return T


class FakeTools(Tools):
    """The toolkit's verbs over the toy environment (perception from `extra["objects"]`, straight-line plans)."""

    def _obs(self) -> Observation:
        return self.executor.observation()

    def status(self) -> dict[str, Any]:
        obs = self._obs()
        q = np.asarray(obs.state["q.arm"])
        return {**self.executor.status(), "q": {"arm": q}, "hand": {"arm": np.asarray(obs.state["hand.arm"])},
                "tcp": {"arm": _pose(q)}}

    def look(self) -> View:
        obs = self._obs()
        return View(cams={"top": CameraView(rgb=obs.images["top"], depth=None, K=None, T_world_cam=None)},
                    points=np.zeros((0, 3)), step=int(self.executor.status()["step"]))

    def find(self, view: View, labels: str | Sequence[str]) -> list[Mask]:
        labels = [labels] if isinstance(labels, str) else list(labels)
        out = []
        for name in self._obs().extra["objects"]:
            for lab in labels:
                if lab.lower() in name:
                    out.append(Mask(label=name, camera="top", mask=np.zeros((IMG, IMG), bool)))
                    break
        return out

    def table(self, view: View) -> Table:
        return Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]), extent=np.array([LOW[:2], HIGH[:2]]).T)

    def objects(self, view: View, masks: Sequence[Mask], table: Table) -> list[Detection]:
        objs = self._obs().extra["objects"]
        out = []
        for m in masks:
            c = np.asarray(objs[m.label], dtype=float)
            corners = np.array([[dx, dy, dz] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]) * BLOCK / 2
            out.append(Detection(label=m.label, score=1.0, points=c + corners if m.label != "bin" else c[None]))
        return out

    def box(self, points: np.ndarray, table: Table | None = None) -> Box:
        p = np.asarray(points, dtype=float)
        return Box(center=p.mean(axis=0), size=np.ptp(p, axis=0), yaw=0.0, bottom=float(p[:, 2].min()))

    def grasps(self, arm: str, points: np.ndarray, table: Table, scene: np.ndarray | None = None) -> list[Goal]:
        return [self.grasp_at(arm, _pose(np.asarray(points, dtype=float).mean(axis=0)))]

    def grasp_at(self, arm: str, tcp: np.ndarray, hand: Sequence[float] | None = None) -> Goal:
        pre = np.array(tcp, dtype=float)
        pre[2, 3] += 0.1
        return Goal(verb="grasp", arm=arm, tcp=np.asarray(tcp, dtype=float), pre=pre, hand_before=(1.0,),
                    hand=tuple(hand) if hand is not None else (0.0,))

    def place_at(self, arm: str, obj_in_tcp: np.ndarray, obj_points: np.ndarray, xy: Sequence[float],
                 yaw: float | None = None, table: Table | None = None, hover: float = 0.03) -> Goal:
        z = (table.height if table is not None else 0.0) + BLOCK / 2 + hover
        tcp = _pose([float(xy[0]), float(xy[1]), z])
        pre = tcp.copy()
        pre[2, 3] += 0.1
        return Goal(verb="place", arm=arm, tcp=tcp, pre=pre, hand=(1.0,))

    def plan(self, goals: Goal | Sequence[Goal], table: Table | None = None,
             scene: np.ndarray | None = None) -> Plan | None:
        goals = [goals] if isinstance(goals, Goal) else sorted(goals, key=lambda g: g.rank)
        obs = self._obs()
        q0, h0 = np.asarray(obs.state["q.arm"], dtype=float), float(obs.state["hand.arm"][0])
        for g in goals:
            ends = [np.asarray(g.joints, dtype=float)] if g.joints is not None else \
                [np.asarray(T, dtype=float)[:3, 3] for T in (g.pre, g.tcp) if T is not None]
            if not ends or any(np.any(e < np.array(LOW) - 1e-9) or np.any(e > np.array(HIGH) + 1e-9) for e in ends):
                continue
            keep = 0.0 if 0.2 < h0 < 0.8 else h0                         # holding: keep the hand closed
            rows, q, h = [], q0, (g.hand_before[0] if g.hand_before is not None else keep)
            rows += [[*q, h]] * (4 if g.hand_before is not None else 0)
            for e in ends:
                n = max(1, int(np.ceil(np.max(np.abs(e - q)) / (SPEED * 0.9))))
                rows += [[*(q + (e - q) * k / n), h] for k in range(1, n + 1)]
                q = e
            if g.hand is not None:
                rows += [[*q, g.hand[0]]] * 4
            return Plan(goal=g, actions=np.asarray(rows, dtype=float), start_q={"arm": q0},
                        info={"steps": len(rows), "planner": "toy straight line"})
        return None

    def run(self, plan: Plan) -> dict[str, Any]:
        if plan is None:
            raise ValueError("run(None): the planner found no reachable goal")
        res = dict(self.executor.run(plan.actions))
        obs = self._obs()
        q = np.asarray(obs.state["q.arm"])
        target = plan.actions[-1, :3]
        return {**res, "q": {"arm": q}, "hand": {"arm": np.asarray(obs.state["hand.arm"])},
                "tcp": {"arm": _pose(q)}, "tcp_error": float(np.linalg.norm(q - target))}

    def holding(self, arm: str) -> bool | None:
        h = float(self._obs().state["hand.arm"][0])
        return 0.2 < h < 0.8

    def carried(self, view: View, points: np.ndarray, tcp_before: np.ndarray, tcp_now: np.ndarray,
                table: Table) -> dict[str, Any]:
        return {"lifted": bool(self.holding("arm")), "offset": None}

    def show(self, image: View | np.ndarray, note: str = "") -> None:
        arr = image.cams[next(iter(image.cams))].rgb if isinstance(image, View) else np.asarray(image)
        if self._show is not None:
            self._show(arr, note)


TOY_SOLUTION = '''
def solve(robot):
    status = robot.status()
    words = status["instruction"].split()
    colour = words[words.index("block") - 1]
    view = robot.look()
    table = robot.table(view)
    block = robot.objects(view, robot.find(view, [colour + " block"]), table)[0]
    target = robot.objects(view, robot.find(view, ["bin"]), table)[0]
    grasp = robot.plan(robot.grasps("arm", block.points, table), table=table)
    robot.run(grasp)
    xy = target.points.mean(axis=0)[:2]
    robot.run(robot.plan(robot.place_at("arm", np.eye(4), block.points, xy, table=table), table=table))
    home = robot.Goal("home", "arm", joints=np.array([0.0, 0.0, 0.3]))
    robot.run(robot.plan(home, table=table))
    print("placed", colour, "at", np.round(xy, 3))
'''.strip("\n")


__all__ = ["FakeTools", "TOY_SOLUTION", "ToyManipEnv", "toy_benchmark", "toy_robot", "toy_tasks"]
