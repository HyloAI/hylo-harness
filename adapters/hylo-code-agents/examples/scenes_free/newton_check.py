"""Cross-check a free scene's demo and success function in a second physics engine: Newton (Warp), on the CPU.

    uv run --with newton --with warp-lang python harness/adapters/hylo-code-agents/examples/scenes_free/newton_check.py \
        harness/adapters/hylo-code-agents/examples/scenes_free/cloth_drape [--solver xpbd|vbd] [--out DIR]

What it does: loads the scene program in `mujoco.free_scene`, lets it settle 1 s and exports the settled world
(table, the program's rigid bodies as boxes, cloth / flex surfaces as triangle meshes, cables as particle chains);
rebuilds that world in Newton; then runs the program's own demo(sim) and success(sim) against a Newton-backed `sim`
(the same read and demo verbs: pos, bounds, flex, nodes, grab, move, move_together, release, wait). A grabbed vertex
or node becomes a kinematic particle that follows the invisible hand.

What it is not: a second backend for the robot. The robot, its controllers, cameras and the toolkit stay MuJoCo's;
this checks only that the task's physics (here: does the cloth end up over the box?) is not an artefact of one
engine. Rigid bodies are approximated by their bounding boxes and are static here (free bodies are reported and
skipped); hinge / slide parts are not rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def export_world(scene_dir: str) -> tuple[dict, object]:
    from hylo.core.registry import registry
    from hylo.core.task import TaskInstance, TaskSpec
    from hylo.core.types import Action

    env = registry.build("envs", {"type": "mujoco.free_scene", "params": {"scene": scene_dir, "cameras": [],
                                                                         "privileged_obs": True}})
    env.reset(TaskInstance(spec=TaskSpec(id="x", version="1", instruction="x", params={}, max_steps=100), seed=0))
    hold = env.hold_action(None)
    for _ in range(20):
        env.step(Action(data=hold))
    m = env.model
    p = env.privileged_state(json_ready=True)
    flexes = {}
    for name, f in p["flexes"].items():
        tris = env.world.flex_topology(name)
        mass = sum(float(m.body_mass[m.body(b).id]) for b in (f["bodies"] or []))
        flexes[name] = {"vertices": f["vertices"], "tris": tris.tolist() if f["dim"] == 2 else [],
                        "bodies": f["bodies"], "radius": f["radius"], "mass": mass}
    world = {"table": p["table"], "bodies": p["bodies"], "cables": p["cables"], "flexes": flexes}
    return world, env.program


class NewtonSim:
    """The free scene's `sim` verbs over a Newton model (read verbs, and the demo's invisible hands)."""

    def __init__(self, world: dict, solver: str = "xpbd", substeps: int = 10, fps: float = 20.0) -> None:
        import newton
        import warp as wp

        wp.config.quiet = True
        self.newton, self.wp = newton, wp
        self.world, self.fps, self.substeps = world, fps, substeps
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        top = float(world["table"]["height"])
        tx, ty = world["table"]["x"], world["table"]["y"]
        b.add_shape_box(-1, xform=wp.transform(((tx[0] + tx[1]) / 2, (ty[0] + ty[1]) / 2, top - 0.05), wp.quat_identity()),
                        hx=(tx[1] - tx[0]) / 2, hy=(ty[1] - ty[0]) / 2, hz=0.05)
        self.boxes, self.skipped = {}, []
        for name, body in world["bodies"].items():
            lo, hi = np.asarray(body["bounds"][0]), np.asarray(body["bounds"][1])
            if body["free"]:
                self.skipped.append(name)
                continue
            c, h = (lo + hi) / 2, (hi - lo) / 2
            if np.all(h > 1e-4):
                b.add_shape_box(-1, xform=wp.transform(tuple(map(float, c)), wp.quat_identity()), hx=float(h[0]),
                                hy=float(h[1]), hz=float(h[2]))
            self.boxes[name] = (lo, hi)
        self.index: dict[str, int] = {}               # MuJoCo body name of a vertex / node -> particle index
        self.flex_range: dict[str, tuple[int, int]] = {}
        for name, f in world["flexes"].items():
            v = np.asarray(f["vertices"], float)
            start = len(b.particle_q)
            mass = f["mass"] / len(v) if f["mass"] > 0 else 0.001
            for k, x in enumerate(v):
                b.add_particle(pos=tuple(map(float, x)), vel=(0.0, 0.0, 0.0), mass=mass, radius=max(f["radius"], 0.003))
                if f["bodies"]:
                    self.index[f["bodies"][k]] = start + k
            for t in f["tris"]:
                b.add_triangle(start + t[0], start + t[1], start + t[2])
            edges = {tuple(sorted((t[i], t[(i + 1) % 3]))) for t in f["tris"] for i in range(3)}
            for i, j in edges:
                b.add_spring(start + i, start + j, ke=2.0e3, kd=2.0, control=0.0)
            self.flex_range[name] = (start, start + len(v))
        self.cable_range: dict[str, tuple[int, int]] = {}
        for name, c in world["cables"].items():
            n = np.asarray(c["nodes"], float)
            start = len(b.particle_q)
            for k, x in enumerate(n):
                b.add_particle(pos=tuple(map(float, x)), vel=(0.0, 0.0, 0.0), mass=0.005, radius=max(c["radius"], 0.004))
                self.index[c["bodies"][k]] = start + k
                if k:
                    b.add_spring(start + k - 1, start + k, ke=5.0e3, kd=5.0, control=0.0)
            self.cable_range[name] = (start, start + len(n))
        self.model = b.finalize(device="cpu")
        self.model.soft_contact_ke, self.model.soft_contact_mu = 2.0e3, 0.6
        self.solver = (newton.solvers.SolverVBD(self.model) if solver == "vbd" else
                       newton.solvers.SolverXPBD(self.model, iterations=10))
        self.s0, self.s1, self.ctrl = self.model.state(), self.model.state(), self.model.control()
        self.inv_mass0 = self.model.particle_inv_mass.numpy().copy()
        self.hands: dict[int, int] = {}               # hand -> particle
        self.hand_pos: dict[int, np.ndarray] = {}
        self.time = 0.0
        self.steps = 0
        self.success_at: int | None = None
        self.success_fn = None
        self.table_top = top

    # read verbs
    def _q(self) -> np.ndarray:
        return self.s0.particle_q.numpy()

    def pos(self, body: str) -> np.ndarray:
        if body in self.index:
            return self._q()[self.index[body]].astype(float)
        lo, hi = self.boxes[body]
        return (lo + hi) / 2

    def bounds(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        return self.boxes[body]

    def flex(self, name: str) -> np.ndarray:
        a, b = self.flex_range[name]
        return self._q()[a:b].astype(float)

    def nodes(self, prefix: str) -> np.ndarray:
        a, b = self.cable_range[prefix]
        return self._q()[a:b].astype(float)

    # demo verbs
    def _set_inv_mass(self) -> None:
        inv = self.inv_mass0.copy()
        for i in self.hands.values():
            inv[i] = 0.0
        self.model.particle_inv_mass.assign(inv)

    def grab(self, body: str, hand: int = 0, point=None) -> None:
        if body not in self.index:
            raise KeyError(f"newton check: only cable nodes and flex vertices can be grabbed, not {body!r}")
        self.hands[hand] = self.index[body]
        self.hand_pos[hand] = self.pos(body)
        self._set_inv_mass()

    def release(self, hand: int = 0) -> None:
        self.hands.pop(hand, None)
        self.hand_pos.pop(hand, None)
        self._set_inv_mass()

    def hand(self, hand: int = 0) -> np.ndarray:
        return self.hand_pos[hand].copy()

    def move(self, hand: int, xyz, seconds: float = 1.0) -> None:
        self.move_together({hand: xyz}, seconds)

    def wait(self, seconds: float = 0.5) -> None:
        self.move_together({}, seconds)

    def move_together(self, targets: dict, seconds: float = 1.0) -> None:
        start = {h: self.hand_pos[h].copy() for h in targets}
        n = max(1, int(round(seconds * self.fps)))
        dt = 1.0 / (self.fps * self.substeps)
        for k in range(1, n + 1):
            for h, xyz in targets.items():
                self.hand_pos[h] = start[h] + (np.asarray(xyz, float) - start[h]) * k / n
            for _ in range(self.substeps):
                q, qd = self.s0.particle_q.numpy(), self.s0.particle_qd.numpy()
                for h, i in self.hands.items():
                    q[i], qd[i] = self.hand_pos[h], 0.0
                self.s0.particle_q.assign(q)
                self.s0.particle_qd.assign(qd)
                self.s0.clear_forces()
                contacts = self.model.collide(self.s0)
                self.solver.step(self.s0, self.s1, self.ctrl, contacts, dt)
                self.s0, self.s1 = self.s1, self.s0
            self.time += 1.0 / self.fps
            self.steps += 1
            if self.success_fn is not None and self.success_at is None and bool(self.success_fn(self)):
                self.success_at = self.steps


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene")
    ap.add_argument("--solver", default="xpbd", choices=["xpbd", "vbd"])
    ap.add_argument("--out", help="write newton_check.json here (default: the scene directory)")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    world, program = export_world(a.scene)
    if program.demo is None:
        raise SystemExit("the scene program has no demo(sim) to replay")
    t1 = time.monotonic()
    sim = NewtonSim(world, a.solver)
    sim.success_fn = program.success
    start = bool(program.success(sim))
    t2 = time.monotonic()
    error = None
    try:
        program.demo(sim)
        for h in list(sim.hands):
            sim.release(h)
        sim.wait(1.0)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    end = bool(program.success(sim))
    res = {"scene": str(Path(a.scene).resolve()), "engine": "newton", "solver": a.solver, "device": "cpu",
           "success_at_start": start, "success_seen": sim.success_at is not None, "success_at_step": sim.success_at,
           "success_end": end, "demo_error": error, "steps": sim.steps, "particles": int(sim.model.particle_count),
           "static_boxes": sorted(sim.boxes), "skipped_free_bodies": sim.skipped,
           "export_s": round(t1 - t0, 1), "build_s": round(t2 - t1, 1), "simulate_s": round(time.monotonic() - t2, 1),
           "newton": __import__("newton").__version__, "warp": __import__("warp").__version__}
    out = Path(a.out or a.scene)
    (out / "newton_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if (res["success_seen"] or end) and not start and error is None else 1


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.exit(main(sys.argv[1:]))
