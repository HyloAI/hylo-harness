"""Table-top scene contract (hylo.scene/v0.1): a manipulation task written as data, so a model can author it, a
checker can validate it and an environment (bench: `mujoco.panda_scene`) can load it.

It lives in hylo-core because it crosses the harness / bench boundary: the harness writes and checks scenes (the
scene builder of RFC-0006 part 8), bench environments load them, and neither side may import the other.

Frames and units: metres; ANGLES IN DEGREES (unlike hylo.robot, so hand-written and model-written files stay
readable); world frame of the environment, +z up, table top at z = 0 unless the environment says otherwise.

A scene is a list of objects and a success predicate:

```text
  object   one rigid body: a primitive (box, cylinder, sphere, capsule, mesh) or a ready-made container
           (tray: open box; cup: open cylinder), or a list of primitive `parts`; name (identifier), label (the
           words an instruction uses), colour, mass, friction
  pose     fixed start pose (pos [x, y] or [x, y, z]; z omitted: resting on the table or on `on`), or `region`:
           a box of start positions and a yaw range sampled from the episode seed (seeded variation)
  fixed    a fixture: does not move (no joint to the world)
  parent   attached to another object, rigidly or through one `joint` (slide or hinge): drawers, doors, lids,
           buttons; its pose is then in the parent's frame
  success  a declarative predicate over object poses and joint states, evaluated after every control step
```

Predicates (field `op`): on, inside, near, above, upright, lifted, row, joint_at_least, joint_at_most, not,
all_of, any_of, sequence. Object predicates require the object at rest and released (not touching the hand)
unless `released: false`. `sequence` latches: each term must become true after the previous ones did.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import AfterValidator, Field, model_validator

from .contract import Contract

SCENE_SCHEMA = "hylo.scene/v0.1"

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$", description="identifier: lower case, digits, _")]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Range = Annotated[tuple[float, float], Field(description="[low, high]")]
RGB = tuple[float, float, float]

COLOURS: dict[str, RGB] = {
    "red": (0.85, 0.12, 0.10), "green": (0.15, 0.65, 0.20), "blue": (0.15, 0.35, 0.90), "yellow": (0.95, 0.82, 0.10),
    "purple": (0.55, 0.22, 0.75), "orange": (0.97, 0.50, 0.08), "white": (0.93, 0.93, 0.91), "black": (0.08, 0.08, 0.09),
    "grey": (0.55, 0.56, 0.58), "brown": (0.50, 0.33, 0.18), "pink": (0.95, 0.62, 0.72), "cyan": (0.35, 0.80, 0.85),
    "wood": (0.70, 0.55, 0.38),
}
def _colour(value: Any) -> Any:
    if isinstance(value, str) and value.lower().strip() not in COLOURS:
        raise ValueError(f"unknown colour {value!r}; use one of {', '.join(COLOURS)} or [r, g, b] in 0..1")
    if not isinstance(value, str) and not all(0.0 <= c <= 1.0 for c in value):
        raise ValueError(f"colour components must be in 0..1, got {list(value)}")
    return value


Colour = Annotated[Union[str, RGB], AfterValidator(_colour),
                   Field(description=f"a name ({', '.join(COLOURS)}) or [r, g, b] in 0..1")]

Shape = Literal["box", "cylinder", "sphere", "capsule", "mesh", "tray", "cup"]
PRIMITIVES = ("box", "cylinder", "sphere", "capsule", "mesh")
CONTAINERS = ("tray", "cup")
SIZE_HELP = ("box: [x, y, z] full extents; cylinder and capsule: [radius, height] (capsule height includes the caps); "
             "sphere: [radius]; tray: [x, y, z] outer extents (open top); cup: [radius, height] outer (open top); "
             "mesh: [scale] or [sx, sy, sz] (default 1)")
_SIZE_LEN = {"box": (3,), "cylinder": (2,), "capsule": (2,), "sphere": (1,), "tray": (3,), "cup": (2,), "mesh": (0, 1, 3)}


def _check_size(shape: str, size: tuple[float, ...], mesh: str | None, wall: float, where: str) -> None:
    if len(size) not in _SIZE_LEN[shape]:
        raise ValueError(f"{where}: a {shape} needs size {SIZE_HELP.split(shape + ': ')[1].split(';')[0]} "
                         f"({_SIZE_LEN[shape]} values), got {list(size)}")
    if any(not (0.0 < s <= 2.0) for s in size):
        raise ValueError(f"{where}: sizes must be in (0, 2] metres, got {list(size)}")
    if shape == "mesh" and not mesh:
        raise ValueError(f"{where}: shape mesh needs `mesh` (a path to an .obj or .stl file)")
    if shape != "mesh" and mesh:
        raise ValueError(f"{where}: `mesh` is only for shape mesh")
    if shape == "capsule" and size[1] < 2 * size[0]:
        raise ValueError(f"{where}: a capsule's height includes both caps, so it must be at least 2 x radius")
    if shape in CONTAINERS:
        inner = min(size[:2]) if shape == "tray" else size[0]
        if 2 * wall >= inner * (1 if shape == "tray" else 2) or wall >= size[-1]:
            raise ValueError(f"{where}: wall {wall} m is too thick for a {shape} of size {list(size)}")


class Part(Contract):
    """One primitive of a compound object, in the object's frame."""
    shape: Literal["box", "cylinder", "sphere", "capsule", "mesh"]
    size: tuple[float, ...] = Field(default=(), description=SIZE_HELP)
    mesh: str | None = None
    pos: Vec3 = (0.0, 0.0, 0.0)
    euler_deg: Vec3 = Field(default=(0.0, 0.0, 0.0), description="roll, pitch, yaw (degrees, extrinsic x-y-z)")
    colour: Colour | None = None       # default: the object's colour

    @model_validator(mode="after")
    def _check(self) -> "Part":
        _check_size(self.shape, self.size, self.mesh, 0.0, "part")
        return self


class Joint(Contract):
    """How an object with a `parent` moves relative to it. Slide values in metres, hinge values in degrees; 0 is
    the pose written in `pose`."""
    type: Literal["slide", "hinge"]
    axis: Vec3 = Field(default=(1.0, 0.0, 0.0), description="in the parent's frame")
    range: Range
    anchor: Vec3 = Field(default=(0.0, 0.0, 0.0), description="hinge: a point on the axis, in the object's frame")
    initial: float = 0.0               # start value (inside range)
    damping: float = Field(default=1.0, ge=0.0)
    spring: float = Field(default=0.0, ge=0.0, description="stiffness pulling back to spring_ref (e.g. a button)")
    spring_ref: float | None = None    # default: initial
    friction: float = Field(default=0.0, ge=0.0, description="dry friction of the joint")

    @model_validator(mode="after")
    def _check(self) -> "Joint":
        lo, hi = self.range
        if not lo < hi:
            raise ValueError(f"joint range must be [low, high] with low < high, got {list(self.range)}")
        if not lo - 1e-9 <= self.initial <= hi + 1e-9:
            raise ValueError(f"joint initial {self.initial} is outside its range {list(self.range)}")
        if math.hypot(*self.axis) < 1e-6:
            raise ValueError("joint axis must not be zero")
        if self.type == "slide" and max(abs(lo), abs(hi)) > 1.0:
            raise ValueError("slide joint range is in metres (at most 1 m)")
        return self


class Pose(Contract):
    pos: Union[Vec2, Vec3] = Field(description="[x, y] (resting on the table or on `on`) or [x, y, z] (centre)")
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


class Region(Contract):
    """Start positions sampled per episode seed: x, y uniform in the ranges, yaw uniform in yaw_deg."""
    x: Range
    y: Range
    yaw_deg: Range = (0.0, 0.0)

    @model_validator(mode="after")
    def _check(self) -> "Region":
        if self.x[0] > self.x[1] or self.y[0] > self.y[1] or self.yaw_deg[0] > self.yaw_deg[1]:
            raise ValueError("region ranges must be [low, high]")
        return self


class SceneObject(Contract):
    name: Name
    label: str = Field(min_length=1, max_length=60, description="words an instruction uses, e.g. 'red block'")
    shape: Shape | None = None         # None: `parts` describe the object
    size: tuple[float, ...] = Field(default=(), description=SIZE_HELP)
    mesh: str | None = Field(default=None, description="shape mesh: path to an .obj or .stl file")
    parts: tuple[Part, ...] = ()
    wall: float = Field(default=0.006, gt=0.0, le=0.05, description="tray / cup wall and floor thickness")
    colour: Colour = "grey"
    mass: float = Field(default=0.1, gt=0.0, le=20.0, description="kg (whole object)")
    friction: float = Field(default=0.8, gt=0.0, le=3.0)
    fixed: bool = False                # fixture: does not move
    parent: Name | None = None
    joint: Joint | None = None
    pose: Pose | None = None
    region: Region | None = None
    on: Name | None = Field(default=None, description="start resting on top of this object")

    @model_validator(mode="after")
    def _check(self) -> "SceneObject":
        w = f"object {self.name}"
        if (self.shape is None) == (not self.parts):
            raise ValueError(f"{w}: give either `shape` (with `size`) or `parts`, not both")
        if self.shape is not None:
            _check_size(self.shape, self.size, self.mesh, self.wall, w)
        if self.pose is not None and self.region is not None:
            raise ValueError(f"{w}: give `pose` or `region`, not both")
        if self.parent is not None:
            if self.region is not None or self.on is not None:
                raise ValueError(f"{w}: an object with a parent is placed by `pose` (in the parent's frame), "
                                 "not by `region` or `on`")
            if self.pose is None:
                raise ValueError(f"{w}: an object with a parent needs `pose` (its position in the parent's frame)")
            if self.pose is not None and len(self.pose.pos) != 3:
                raise ValueError(f"{w}: with a parent, pose.pos must be [x, y, z] in the parent's frame")
            if self.fixed:
                raise ValueError(f"{w}: `fixed` is for objects without a parent (a child is held by its parent)")
        elif self.joint is not None:
            raise ValueError(f"{w}: a joint needs a `parent` (the object it moves relative to; make the parent "
                             "`fixed` for a cabinet or a frame)")
        if self.pose is None and self.region is None and self.on is None and self.parent is None:
            raise ValueError(f"{w}: needs a `pose`, a `region` or `on`")
        return self

    @property
    def movable(self) -> bool:
        """Free to move as a whole (no parent, not fixed)."""
        return self.parent is None and not self.fixed


class Box3(Contract):
    """An axis-aligned box in the world (z optional: any height)."""
    x: Range
    y: Range
    z: Range | None = None


# -- predicates --------------------------------------------------------------------------------------------------
class On(Contract):
    """`a` rests on top of `b` ("table" for the table): touching it, bottom at or above b's top, centre over b."""
    op: Literal["on"]
    a: Name
    b: Name
    released: bool = True


class Inside(Contract):
    """`a`'s centre inside container `b` (its interior; any object: its bounding box) or inside `region`."""
    op: Literal["inside"]
    a: Name
    b: Name | None = None
    region: Box3 | None = None
    released: bool = True

    @model_validator(mode="after")
    def _check(self) -> "Inside":
        if (self.b is None) == (self.region is None):
            raise ValueError("inside needs exactly one of `b` (an object) or `region`")
        return self


class Near(Contract):
    """`a`'s centre within max_dist of `b`'s centre or of `point` (horizontal distance unless xy_only is false)."""
    op: Literal["near"]
    a: Name
    b: Name | None = None
    point: Union[Vec2, Vec3, None] = None
    max_dist: float = Field(default=0.05, gt=0.0)
    xy_only: bool = True
    released: bool = True

    @model_validator(mode="after")
    def _check(self) -> "Near":
        if (self.b is None) == (self.point is None):
            raise ValueError("near needs exactly one of `b` or `point`")
        return self


class Above(Contract):
    """`a`'s bottom at or above `b`'s top and its centre over b's footprint (touching or not)."""
    op: Literal["above"]
    a: Name
    b: Name
    released: bool = True


class Upright(Contract):
    """`a`'s own `axis` points up (either way along it) within max_tilt_deg."""
    op: Literal["upright"]
    a: Name
    axis: Literal["x", "y", "z"] = "z"
    max_tilt_deg: float = Field(default=15.0, gt=0.0, le=90.0)
    released: bool = True


class Lifted(Contract):
    """`a`'s bottom at least `height` above the table (held or resting on something)."""
    op: Literal["lifted"]
    a: Name
    height: float = Field(default=0.05, gt=0.0)


class Row(Contract):
    """The objects in a line along `axis`, in the listed order of increasing coordinate, each at most max_offset
    from the line and at least min_gap apart (centre to centre)."""
    op: Literal["row"]
    objects: tuple[Name, ...] = Field(min_length=2)
    axis: Literal["x", "y"] = "y"
    max_offset: float = Field(default=0.03, gt=0.0)
    min_gap: float = Field(default=0.0, ge=0.0)
    released: bool = True


class JointAtLeast(Contract):
    """The joint of `object` (its joint to its parent) at least `value` (metres or degrees) or at least `fraction`
    of the way from its range's low end to its high end."""
    op: Literal["joint_at_least"]
    object: Name
    value: float | None = None
    fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> "JointAtLeast":
        if (self.value is None) == (self.fraction is None):
            raise ValueError(f"{self.op} needs exactly one of `value` or `fraction`")
        return self


class JointAtMost(JointAtLeast):
    """The joint of `object` at most `value` or `fraction` of its range."""
    op: Literal["joint_at_most"]  # type: ignore[assignment]


class Not(Contract):
    op: Literal["not"]
    term: "Predicate"


class AllOf(Contract):
    op: Literal["all_of"]
    terms: tuple["Predicate", ...] = Field(min_length=1)


class AnyOf(Contract):
    op: Literal["any_of"]
    terms: tuple["Predicate", ...] = Field(min_length=1)


class Sequence(Contract):
    """Each term must become true after all earlier terms did (latched); succeeds when the last one holds."""
    op: Literal["sequence"]
    terms: tuple["Predicate", ...] = Field(min_length=1)


Predicate = Annotated[Union[On, Inside, Near, Above, Upright, Lifted, Row, JointAtLeast, JointAtMost, Not, AllOf,
                            AnyOf, Sequence], Field(discriminator="op")]
for _m in (Not, AllOf, AnyOf, Sequence):
    _m.model_rebuild()


class Scene(Contract):
    schema_version: Literal["hylo.scene/v0.1"] = SCENE_SCHEMA
    name: Name
    instruction: str = Field(min_length=3, max_length=400)
    objects: tuple[SceneObject, ...] = Field(min_length=1, max_length=40)
    success: Predicate
    max_steps: int = Field(default=600, ge=20, le=5000, description="control steps (20 per second on the Panda)")
    notes: str = ""

    def object(self, name: str) -> SceneObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"no object {name!r}")


# -- helpers ------------------------------------------------------------------------------------------------------
def walk(p: Predicate):
    """Every predicate node, depth first."""
    yield p
    if isinstance(p, Not):
        yield from walk(p.term)
    elif isinstance(p, (AllOf, AnyOf, Sequence)):
        for t in p.terms:
            yield from walk(t)


def referenced(p: Predicate) -> list[tuple[str, str]]:
    """(object name, role) pairs a predicate mentions (role: "a", "b", "object", "objects")."""
    out = []
    for n in walk(p):
        for role in ("a", "b", "object"):
            v = getattr(n, role, None)
            if isinstance(v, str):
                out.append((v, role))
        for v in getattr(n, "objects", ()) or ():
            out.append((v, "objects"))
    return out


def nominal_extent(o: SceneObject) -> Vec3 | None:
    """Full extents of an object's own shape in its frame (None for meshes and compound objects)."""
    if o.shape in ("box", "tray"):
        return tuple(o.size)  # type: ignore[return-value]
    if o.shape in ("cylinder", "capsule", "cup"):
        return (2 * o.size[0], 2 * o.size[0], o.size[1])
    if o.shape == "sphere":
        return (2 * o.size[0],) * 3  # type: ignore[return-value]
    return None


def interior(o: SceneObject) -> Vec3 | None:
    """Inner extents of a tray or cup (x, y, depth)."""
    if o.shape == "tray":
        return (o.size[0] - 2 * o.wall, o.size[1] - 2 * o.wall, o.size[2] - o.wall)
    if o.shape == "cup":
        d = 2 * (o.size[0] - o.wall)
        return (d, d, o.size[1] - o.wall)
    return None


class Problem(Contract):
    """One thing wrong with a scene, as a checker reports it."""
    code: str                          # schema, reference, duplicate, unsatisfiable, load, penetration, ...
    message: str
    objects: tuple[str, ...] = ()
    severity: Literal["error", "warning"] = "error"


def _p(code: str, message: str, *objects: str, severity: str = "error") -> Problem:
    return Problem(code=code, message=message, objects=tuple(objects), severity=severity)  # type: ignore[arg-type]


def scene_problems(scene: Scene) -> list[Problem]:
    """What can be found wrong from the file alone: names, labels, references, attachment cycles, predicates that
    can never hold. The environment's checker adds what needs physics (loads, settles, reachable, false at start)."""
    out: list[Problem] = []
    names: dict[str, SceneObject] = {}
    for o in scene.objects:
        if o.name in names or o.name == "table":
            out.append(_p("duplicate", f"object name {o.name!r} is used twice (or is the reserved name 'table')", o.name))
        names[o.name] = o
    labels: dict[str, str] = {}
    for o in scene.objects:
        key = " ".join(o.label.lower().split())
        if key in labels:
            out.append(_p("duplicate_label", f"objects {labels[key]!r} and {o.name!r} share the label {o.label!r}; "
                                             "labels must tell objects apart", labels[key], o.name))
        labels[key] = o.name
    for o in scene.objects:
        for ref, what in ((o.parent, "parent"), (o.on, "on")):
            if ref is not None and ref not in names:
                out.append(_p("reference", f"object {o.name!r}: {what} {ref!r} is not an object", o.name))
            elif ref == o.name:
                out.append(_p("reference", f"object {o.name!r} cannot be its own {what}", o.name))
    for what in ("parent", "on"):
        for o in scene.objects:
            seen, cur = {o.name}, getattr(o, what)
            while cur in names:
                if cur in seen:
                    out.append(_p("reference", f"object {o.name!r}: `{what}` links form a cycle", o.name))
                    break
                seen.add(cur)
                cur = getattr(names[cur], what)
    for name, role in referenced(scene.success):
        if name == "table" and role == "b":
            continue
        if name not in names:
            out.append(_p("reference", f"the success predicate mentions {name!r}, which is not an object "
                                       f"(objects: {sorted(names)})", name))
    if out:
        return out

    def moves(n: str) -> bool:
        o = names[n]
        while True:
            if o.movable or o.joint is not None:
                return True
            if o.parent is None:
                return False
            o = names[o.parent]

    for node in walk(scene.success):
        if isinstance(node, (On, Inside, Above, Near, Upright, Lifted)):
            a = node.a
            other = getattr(node, "b", None)
            if other == a:
                out.append(_p("unsatisfiable", f"{node.op}({a}, {a}): an object cannot relate to itself", a))
            if not moves(a) and (other is None or other == "table" or not moves(other)):
                out.append(_p("unsatisfiable", f"{node.op}: {a!r} cannot move (fixed, or attached rigidly to a "
                                               "fixture), so the robot cannot change whether this holds", a))
            if isinstance(node, Inside) and node.b is not None:
                b = names[node.b]
                ia, eb = nominal_extent(names[a]), interior(b) or nominal_extent(b)
                if ia is not None and eb is not None and sorted(ia[:2])[0] >= sorted(eb[:2])[0]:
                    out.append(_p("unsatisfiable", f"inside({a}, {node.b}): {a!r} (footprint {ia[0]:.3f} x "
                                                   f"{ia[1]:.3f} m) does not fit in {node.b!r} (opening {eb[0]:.3f} x "
                                                   f"{eb[1]:.3f} m)", a, node.b))
        if isinstance(node, Row):
            if len(set(node.objects)) != len(node.objects):
                out.append(_p("unsatisfiable", "row lists an object twice", *node.objects))
        if isinstance(node, (JointAtLeast, JointAtMost)):
            o = names[node.object]
            if o.joint is None:
                out.append(_p("unsatisfiable", f"{node.op}: {node.object!r} has no joint", node.object))
            elif node.value is not None:
                lo, hi = o.joint.range
                if not lo <= node.value <= hi:
                    out.append(_p("unsatisfiable", f"{node.op}({node.object}, {node.value}) is outside the joint's "
                                                   f"range {list(o.joint.range)}", node.object))
    for node in walk(scene.success):                   # contradictions inside one all_of
        if isinstance(node, AllOf):
            rest: dict[str, str] = {}
            for t in node.terms:
                if isinstance(t, On):
                    if t.a in rest and rest[t.a] != t.b:
                        out.append(_p("unsatisfiable", f"all_of asks {t.a!r} to rest on both {rest[t.a]!r} and "
                                                       f"{t.b!r}", t.a))
                    rest[t.a] = t.b
            for a, b in rest.items():
                if rest.get(b) == a:
                    out.append(_p("unsatisfiable", f"all_of asks {a!r} on {b!r} and {b!r} on {a!r}", a, b))
                    break
    return out


def validation_problems(error: Exception) -> list[Problem]:
    """A pydantic ValidationError (or any exception) as problems, one per error, with the field path."""
    errs = getattr(error, "errors", None)
    if not callable(errs):
        return [_p("schema", str(error)[:500])]
    out = []
    for e in errs()[:20]:
        loc = ".".join(str(x) for x in e.get("loc", ()) if not (isinstance(x, str) and x[:1].isupper()))
        out.append(_p("schema", f"{loc or '(top level)'}: {e.get('msg', '')}"[:400]))
    return out


__all__ = ["AllOf", "Above", "AnyOf", "Box3", "COLOURS", "CONTAINERS", "Inside", "Joint", "JointAtLeast",
           "JointAtMost", "Lifted", "Near", "Not", "On", "PRIMITIVES", "Part", "Pose", "Predicate", "Problem", "Region",
           "Row", "SCENE_SCHEMA", "Scene", "SceneObject", "Sequence", "Upright", "interior", "nominal_extent",
           "referenced", "scene_problems", "validation_problems", "walk"]
