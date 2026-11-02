"""What the free-mode scene builder tells the model: the scene-program format, the robots, two examples, and how
the checker's report reads back.

    free_builder_system() -> str      the system message (robots described by the free-scene environment itself)
    feedback_text(report) -> str      a CheckReport as the next user message
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from hylo.core.registry import registry

from .scene_free_check import FREE_ENV

SYSTEM = """You write simulation tasks for robot manipulation as Python programs. A program builds a MuJoCo world
around a robot at a table, states the task, and judges success from the simulation state. A checker loads it, lets
it settle on several seeds, checks reach and runs your demo; then a separate agent tries to do the task with the robot.

## The program (format hylo.freescene/v0.1)
Top level (plain constants and functions only):
    ROBOT = "panda" | "aloha" | "g1"        the robot that fits the task (see Robots)
    INSTRUCTION = "..."                     what the robot must do, naming each object as it looks
    MAX_STEPS = 900                         control steps at 20 Hz (100 .. 4000)
    LABELS = {{"rope": "red rope"}}           optional: words for bodies, cable prefixes and flexes
    def build(scene): ...                   required: add the world's things
    def success(sim): ...                   required: return a plain bool, False at the start
    def demo(sim): ...                      strongly recommended: do the task with invisible hands (checker only)
    def randomize(sim, rng): ...            optional: vary the start per seed (sim.set_body / set_joint)
    def targets(sim): ...                   optional: {{name: xyz}} points a robot hand must be able to reach
Imports allowed: math, numpy (as np), mujoco. No files, no classes, no names starting with "_".

build(scene):
    scene.add_mjcf(xml, pos=None)  add an MJCF document (<mujoco> with worldbody, asset, default, tendon, equality,
                                   contact, extension sections); names stay as written. Angles in this MJCF are
                                   DEGREES (the MJCF default) unless you add <compiler angle="radian"/>.
    scene.spec                     the mujoco.MjSpec being built (you may also use the model editor directly)
    scene.table_top                z of the table surface; scene.table = {{"x": (lo, hi), "y": (lo, hi), "top"}}
    scene.arms                     {{arm: {{"base": xyz, "x": (lo, hi), "y": (lo, hi)}}}}: where each arm works best
    scene.opening                  the gripper's max opening (m); scene.rgba("red") -> [r, g, b, 1]
success(sim) / randomize(sim, rng) / demo(sim) / targets(sim) read the state through sim:
    sim.pos(body) (3,)   sim.quat(body) (w,x,y,z)   sim.xmat(body) (3,3)   sim.speed(body) m/s
    sim.bounds(body) -> (min (3,), max (3,)) world box of the body and its children
    sim.joint(name) value (radians for hinges, metres for slides)   sim.joint_range(name)
    sim.nodes(prefix) (N,3) a composite cable's bodies in order (prefix+"B_first" ... prefix+"B_last")
    sim.flex(name) (N,3) flex vertices (cloth)   sim.site(name)   sim.geom(name)   sim.bodies(prefix="")
    sim.touching(a, b) contact between two things: body, cable prefix, flex name, "table", "robot" or "hand"
    sim.held(thing) touched by the robot's fingers   sim.table_top   sim.time
randomize and demo may also edit: sim.set_body(body, pos=None, quat=None) (free-joint bodies), sim.set_joint(name, v).
demo(sim) moves things with invisible hands while the physics runs (the robot stays still):
    sim.grab(body, hand=0)  hold a body at its frame origin   sim.move(hand, xyz, seconds=1.0) straight line
    sim.move_together({{0: xyz, 1: xyz}}, seconds)   sim.release(hand)   sim.wait(seconds)   sim.hand(hand) -> xyz
  success must become True during the demo or within 1 s after the hands let go. Grab only where a real gripper
  could (an end of a cable, a corner or edge of cloth, a handle, a small object): those points must be reachable.

## The world
- Metres, kilograms, +z up. Every object is a named body; a body with <freejoint/> can move freely, a body without
  joints is fixed to the world, a child body with a hinge or slide joint is an articulated part (give it a range,
  damping, and armature="0.001").
- Objects rest ON their support at the start: z = support top + half height (+0.001). Nothing may overlap anything
  else or the robot; leave 3 cm between objects.
- Put everything the robot must touch inside an arm's zone (scene.arms), and things that must be picked up at most
  scene.opening - 0.01 m wide where they are grasped, at most 0.3 kg.
- Deformables:
  * cable / rope: <composite prefix="rope" type="cable" curve="s" count="N 1 1" size="L" offset="x y z"
    initial="free"> with <plugin plugin="mujoco.elasticity.cable"> (declare <extension><plugin
    plugin="mujoco.elasticity.cable"/></extension>), <joint kind="main" damping="0.02"/> and <geom type="capsule"
    size="R" density="..."/>. It runs along +x from offset; bodies are rope+"B_first", rope+"B_1", ..., rope+"B_last".
    Use initial="free" (initial="none" fixes the first end to the world). N 15-30, R 0.008-0.012, z = top + R.
  * cloth: <flexcomp name="cloth" type="grid" count="8 8 1" spacing="0.025 0.025 0.025" pos="x y z" radius="0.004"
    dim="2" mass="0.04"> <edge equality="true" damping="0.01"/> <contact selfcollide="none" condim="3"/>
    </flexcomp>: centred at pos, flat in xy; vertex bodies cloth_0, cloth_1, ... (row by row); at most 10 x 10.
  * a gripper's fingertips keep its centre (TCP) about 1.5 cm above whatever it reaches down to, so it cannot pinch
    a thin cable or cloth lying flat on the table: give a cable radius >= 0.012, or start the part to be grasped on
    a raised support (a low block or pad, 2-4 cm) with an end or edge overhanging it.
  * also allowed: <tendon> (spatial / fixed), <equality> (connect, weld, joint), springs (stiffness / springref).
- No <actuator> and no <keyframe>: the robot is the only actor.

## Robots
{robots}

Reply with one or two sentences of plan, then the complete program in ONE ```python block.

## Example 1: "slide the green gate to the other end of the rail"
```python
{example1}
```

## Example 2: "put the end of the rope in the bowl"
```python
{example2}
```"""

EXAMPLE_1 = '''ROBOT = "panda"
INSTRUCTION = "slide the green gate to the other end of the rail"
MAX_STEPS = 600
LABELS = {"rail": "grey rail", "gate": "green gate"}

def build(scene):
    t = scene.table_top
    scene.add_mjcf(f"""
<mujoco>
  <worldbody>
    <body name="rail" pos="-0.02 0.0 {t + 0.01}">
      <geom type="box" size="0.03 0.14 0.01" rgba="0.5 0.5 0.5 1"/>
      <body name="gate" pos="0 -0.07 0.05">
        <joint name="gate_slide" type="slide" axis="0 1 0" range="0 0.12" damping="3" armature="0.001"/>
        <geom type="box" size="0.006 0.05 0.04" rgba="0.2 0.7 0.3 1" mass="0.1"/>
        <geom name="knob" type="box" size="0.012 0.012 0.012" pos="0 0 0.052" rgba="0.1 0.1 0.1 1" mass="0.01"/>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="rail" body2="gate"/></contact>
</mujoco>""")

def success(sim):
    return bool(sim.joint("gate_slide") > 0.1)

def demo(sim):
    sim.grab("gate", hand=0)
    sim.move(0, sim.pos("gate") + np.array([0.0, 0.115, 0.0]), seconds=1.5)
    sim.release(0)

def targets(sim):
    return {"knob": sim.geom("knob")}
'''

EXAMPLE_2 = '''ROBOT = "panda"
INSTRUCTION = "put the end of the red rope in the bowl"
MAX_STEPS = 900
LABELS = {"rope": "red rope", "bowl": "bowl", "step": "grey step"}

def build(scene):
    t = scene.table_top
    walls = "".join(
        f'<geom type="box" size="{sx} {sy} 0.025" pos="{x} {y} 0.03" rgba="0.3 0.4 0.8 1"/>'
        for x, y, sx, sy in ((0.055, 0, 0.005, 0.06), (-0.055, 0, 0.005, 0.06), (0, 0.055, 0.06, 0.005), (0, -0.055, 0.06, 0.005)))
    scene.add_mjcf(f"""
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <body name="bowl" pos="0.05 0.18 {t}">
      <geom type="box" size="0.06 0.06 0.005" pos="0 0 0.005" rgba="0.3 0.4 0.8 1"/>{walls}
    </body>
    <body name="step" pos="0.0 -0.2 {t + 0.015}"><geom type="box" size="0.05 0.05 0.015" rgba="0.6 0.6 0.6 1"/></body>
    <composite prefix="rope" type="cable" curve="s" count="20 1 1" size="0.38" offset="-0.33 -0.2 {t + 0.04}" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="5e4"/><config key="bend" value="5e4"/><config key="vmax" value="0"/>
      </plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size="0.01" rgba="0.8 0.15 0.1 1" density="300" condim="4"/>
    </composite>
  </worldbody>
</mujoco>""")

def success(sim):
    lo, hi = sim.bounds("bowl")
    end = sim.pos("ropeB_last")
    return bool(lo[0] < end[0] < hi[0] and lo[1] < end[1] < hi[1] and end[2] < hi[2])

def demo(sim):
    end = sim.pos("ropeB_last")
    lo, hi = sim.bounds("bowl")
    above = np.array([(lo[0] + hi[0]) / 2, hi[1] - 0.035, hi[2] + 0.06])
    sim.grab("ropeB_last", hand=0)
    sim.move(0, end + np.array([0.0, 0.0, 0.1]), seconds=1.0)
    sim.move(0, above, seconds=2.0)
    sim.move(0, above - np.array([0.0, 0.0, 0.09]), seconds=1.0)
    sim.release(0)
'''


def robots_text() -> str:
    """What each robot offers, from the free-scene environment itself (one robot-only program per robot)."""
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for robot in ("panda", "aloha", "g1"):
            p = Path(tmp) / robot / "scene.py"
            p.parent.mkdir()
            p.write_text(f'ROBOT = "{robot}"\nINSTRUCTION = "x"\ndef build(scene):\n    pass\n'
                         "def success(sim):\n    return False\n")
            env = registry.build("envs", {"type": FREE_ENV, "params": {"scene": str(p), "cameras": []}})
            spec = env.spec().robot
            facts = env.robot_facts
            arms = ", ".join(f"{a.name} ({a.hand.kind} hand, opens {a.hand.max_width:.3f} m)" for a in spec.arms)
            zones = "; ".join(f"{k}: x {list(v['x'])}, y {list(v['y'])}" for k, v in facts["arms"].items())
            lines.append(f"- {robot}: arms {arms}. Table top z = {facts['table_top']}, table x {list(facts['table_x'])}, "
                         f"y {list(facts['table_y'])}. Arm zones: {zones}.\n  {spec.notes.split(' Objects come from')[0]}")
            env.close()
    return "\n".join(lines)


def free_builder_system() -> str:
    return SYSTEM.format(robots=robots_text(), example1=EXAMPLE_1.strip(), example2=EXAMPLE_2.strip())


def feedback_text(report: Any) -> str:
    if report.ok:
        head = "The checker passed every stage: " + ", ".join(report.stages) + "."
    else:
        head = f"The checker rejected the program (stages run: {', '.join(report.stages)}):\n" + report.text()
    if report.ok and report.warnings:
        head += "\n" + "\n".join(f"- warning [{p.code}] {p.message}" for p in report.warnings[:6])
    where = ""
    if report.settled:
        where = "\nThe settled scene (seed 0):\n" + "\n".join(f"- {k}: {json.dumps(v)}" for k, v in
                                                               list(report.settled.items())[:30])
    return head + where + ("" if report.ok else "\nFix every problem and reply with the complete corrected program.")



__all__ = ["EXAMPLE_1", "EXAMPLE_2", "SYSTEM", "feedback_text", "free_builder_system", "robots_text"]
