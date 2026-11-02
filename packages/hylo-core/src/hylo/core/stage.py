"""Stage contract (hylo.stage/v0.1): an open world as an entity tree that a person, a model or a program edits through
patches, with every revision kept. A world engine (bench: `hylo.envs.mujoco.world`) compiles a stage into a running
simulation; tools (hylo-agents `hylo.agents.worldtools`) read and change it.

It differs from the table-top `hylo.scene/v0.1` on purpose: a scene is one manipulation task written as data (objects
on a table and a success predicate); a stage is a general workspace (robots of any kind from packs, mounted on each
other or on fixtures, meshes, cameras, people) that changes while someone works in it.

Frames and units: metres, radians in quaternions; world frame +z up; `quat` is (w, x, y, z) like MuJoCo.

```text
  entity      id, parent (an entity id or none: the world), label, pose in the parent's frame, locked
              and components, each optional:
    shape     a primitive (box, sphere, cylinder, capsule, plane): size, rgba
    mesh      a mesh file (obj, stl) with a scale; its collision geometry comes from `collision`
    body      physics: dynamic (has a free joint) or static, mass or density, friction
    collision how the visual geometry becomes contact geometry: the primitive itself, the convex hull, convex
              parts (a decomposition file) or voxel boxes (hylo-tools `physicalize` writes them)
    robot     a robot from a pack (hylo.pack/v0.1): pack name, base fixed or free, `mount` onto a body of
              another entity, initial joint values
    camera    a camera: vertical field of view, image size, depth on/off; `body` names the body it rides on
    light     a directional or spot light
    human     a walking person (hylo.envs.mujoco.humans): route, speed, appearance
```

A patch is a list of operations with the revision it was written against (compare and swap): `add` an entity,
`remove` one (with its children when `cascade`), `update` some of its fields, `reparent` it. The store applies a
patch to a copy, refuses it if the stage changed in between or an operation touches a locked entity, validates the
result and writes it as the next revision; old revisions are never changed, and `restore(n)` commits revision n again
as the newest.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Union

from pydantic import Field, model_validator

from .contract import Contract

STAGE_SCHEMA = "hylo.stage/v0.1"
Id = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$", description="identifier: lower case, digits, _")]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
RGBA = tuple[float, float, float, float]


def _finite(values: Any, where: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{where}: values must be finite, got {list(values)}")


class Pose(Contract):
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = Field(default=(1.0, 0.0, 0.0, 0.0), description="(w, x, y, z); normalised if within 1e-3 of unit")

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("quat") is not None:
            q = [float(c) for c in data["quat"]]
            n = math.sqrt(sum(c * c for c in q))
            if len(q) == 4 and math.isfinite(n) and abs(n - 1.0) <= 1e-3 and n > 0:
                data = {**data, "quat": tuple(c / n for c in q)}
        return data

    @model_validator(mode="after")
    def _check(self) -> "Pose":
        _finite(self.pos + self.quat, "pose")
        if abs(math.sqrt(sum(c * c for c in self.quat)) - 1.0) > 1e-6:
            raise ValueError(f"pose.quat must be a unit quaternion (w, x, y, z), got {list(self.quat)}")
        return self

    @classmethod
    def yaw(cls, pos: Vec3, yaw: float = 0.0) -> "Pose":
        return cls(pos=pos, quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


_SIZES = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2, "plane": 2}


class Shape(Contract):
    type: Literal["box", "sphere", "cylinder", "capsule", "plane"]
    size: tuple[float, ...] = Field(description="box: half extents [x, y, z]; sphere: [radius]; cylinder, capsule: "
                                                "[radius, half length]; plane: half extents [x, y]")
    rgba: RGBA = (0.6, 0.6, 0.62, 1.0)

    @model_validator(mode="after")
    def _check(self) -> "Shape":
        if len(self.size) != _SIZES[self.type] or any(not (0 < s <= 1000) for s in self.size):
            raise ValueError(f"shape {self.type} needs {_SIZES[self.type]} positive sizes, got {list(self.size)}")
        return self


class Mesh(Contract):
    path: str = Field(description="obj or stl file; relative paths are relative to the stage's folder")
    scale: Vec3 = (1.0, 1.0, 1.0)
    rgba: RGBA | None = None

    @model_validator(mode="after")
    def _check(self) -> "Mesh":
        if any(s == 0 or not math.isfinite(s) for s in self.scale):
            raise ValueError(f"mesh scale must be finite and non-zero, got {list(self.scale)}")
        return self


class Body(Contract):
    dynamic: bool = True                     # a free joint: falls, can be pushed and carried
    mass: float | None = Field(default=None, gt=0)
    density: float = Field(default=500.0, gt=0, description="kg/m^3 when mass is not given")
    friction: float = Field(default=1.0, ge=0)


class Collision(Contract):
    strategy: Literal["none", "primitive", "hull", "parts", "voxels"] = "primitive"
    parts: str | None = Field(default=None, description="parts / voxels: the file `physicalize` wrote (json)")

    @model_validator(mode="after")
    def _check(self) -> "Collision":
        if self.strategy in ("parts", "voxels") and not self.parts:
            raise ValueError(f"collision strategy {self.strategy} needs `parts` (run `hylo tool physicalize`)")
        return self


class Mount(Contract):
    entity: Id                               # the entity carrying this one
    body: str | None = None                  # a body of that entity's model (robots); none: its root


class Robot(Contract):
    pack: str = Field(description="a robot pack name (hylo.pack/v0.1), see `hylo packs list`")
    base: Literal["fixed", "free"] = "fixed"
    mount: Mount | None = None               # fixed base attached to another entity instead of the world
    joints: dict[str, float] = Field(default_factory=dict, description="initial joint values; default: the "
                                                                      "pack's home keyframe")


class Camera(Contract):
    fovy_deg: float = Field(default=58.0, gt=1, lt=179)
    width: int = Field(default=320, ge=8, le=4096)
    height: int = Field(default=240, ge=8, le=4096)
    depth: bool = True
    body: str | None = Field(default=None, description="ride on this body of the parent entity's model")


class Light(Contract):
    kind: Literal["directional", "spot"] = "directional"
    direction: Vec3 = (0.0, 0.0, -1.0)
    intensity: float = Field(default=0.7, ge=0, le=1)


class Human(Contract):
    route: tuple[tuple[float, float], ...] = Field(default=(), description="waypoints [x, y] in the world frame")
    loop: bool = True
    speed: float = Field(default=1.0, gt=0, le=3, description="walking speed, m/s")
    height: float = Field(default=1.75, gt=0.5, le=2.4)
    rgba: RGBA = (0.35, 0.45, 0.65, 1.0)
    dwell: float = Field(default=0.0, ge=0, description="seconds to stand at each waypoint")


class Entity(Contract):
    id: Id
    parent: Id | None = None
    label: str = ""
    pose: Pose = Field(default_factory=Pose)
    locked: bool = False
    shape: Shape | None = None
    mesh: Mesh | None = None
    body: Body | None = None
    collision: Collision | None = None
    robot: Robot | None = None
    camera: Camera | None = None
    light: Light | None = None
    human: Human | None = None

    @model_validator(mode="after")
    def _check(self) -> "Entity":
        kinds = [k for k in ("shape", "mesh", "robot", "camera", "light", "human") if getattr(self, k) is not None]
        if len(kinds) > 1:
            raise ValueError(f"entity {self.id}: one of shape, mesh, robot, camera, light, human per entity "
                             f"(got {kinds}); group several under a parent entity")
        if self.body is not None and not (self.shape or self.mesh):
            raise ValueError(f"entity {self.id}: `body` needs a shape or a mesh")
        if self.shape and self.shape.type == "plane" and self.body and self.body.dynamic:
            raise ValueError(f"entity {self.id}: a plane cannot be dynamic")
        if self.robot and self.robot.mount and self.robot.base != "fixed":
            raise ValueError(f"entity {self.id}: a mounted robot needs base: fixed")
        return self

    @property
    def kind(self) -> str:
        for k in ("robot", "shape", "mesh", "camera", "light", "human"):
            if getattr(self, k) is not None:
                return k
        return "group"


class Stage(Contract):
    schema_version: Literal["hylo.stage/v0.1"] = STAGE_SCHEMA
    name: Id
    revision: int = Field(default=0, ge=0)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    timestep: float = Field(default=0.002, gt=0, le=0.05)
    ground: bool = True                      # an infinite floor at z = 0
    notes: str = ""
    entities: tuple[Entity, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Stage":
        problems = stage_problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def get(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(f"no entity {entity_id!r}; entities: {[e.id for e in self.entities]}")

    def children(self, entity_id: str | None) -> list[Entity]:
        return [e for e in self.entities if e.parent == entity_id]

    def ordered(self) -> list[Entity]:
        """Parents before children (the order an engine builds them in)."""
        out: list[Entity] = []

        def visit(pid: str | None) -> None:
            for e in self.children(pid):
                out.append(e)
                visit(e.id)
        visit(None)
        return out

    def digest(self) -> str:
        body = self.model_dump(mode="json", exclude={"revision"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def stage_problems(stage: Stage) -> list[str]:
    ids = [e.id for e in stage.entities]
    problems = [f"duplicate entity id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    known = set(ids)
    for e in stage.entities:
        if e.parent is not None and e.parent not in known:
            problems.append(f"entity {e.id}: parent {e.parent!r} does not exist")
        if e.robot and e.robot.mount and e.robot.mount.entity not in known:
            problems.append(f"entity {e.id}: mount target {e.robot.mount.entity!r} does not exist")
        if e.robot and e.robot.mount and e.robot.mount.entity == e.id:
            problems.append(f"entity {e.id}: cannot be mounted on itself")
    parent = {e.id: e.parent for e in stage.entities}
    for e in stage.entities:
        seen, cur = {e.id}, parent.get(e.id)
        while cur is not None and cur in parent:
            if cur in seen:
                problems.append(f"entity {e.id}: parent chain has a cycle")
                break
            seen.add(cur)
            cur = parent[cur]
    mounts = {e.id: e.robot.mount.entity for e in stage.entities if e.robot and e.robot.mount}
    for start in mounts:
        seen, cur = {start}, mounts.get(start)
        while cur in mounts:
            if cur in seen:
                problems.append(f"entity {start}: mount chain has a cycle")
                break
            seen.add(cur)
            cur = mounts[cur]
    return problems


# --------------------------------------------------------------------------------------------------- patches


class Add(Contract):
    op: Literal["add"] = "add"
    entity: Entity


class Remove(Contract):
    op: Literal["remove"] = "remove"
    id: Id
    cascade: bool = False


class Update(Contract):
    op: Literal["update"] = "update"
    id: Id
    set: dict[str, Any] = Field(description="fields to replace, e.g. {\"pose\": {...}, \"label\": \"...\"}; a "
                                            "component given as null is removed")


class Reparent(Contract):
    op: Literal["reparent"] = "reparent"
    id: Id
    parent: Id | None = None
    keep_world_pose: bool = False            # engines may refuse when the pose cannot be kept exactly


Op = Annotated[Union[Add, Remove, Update, Reparent], Field(discriminator="op")]


class Patch(Contract):
    expected_revision: int = Field(ge=0, description="the revision this patch was written against")
    ops: tuple[Op, ...] = Field(min_length=1)
    note: str = ""
    author: str = ""


class StageConflict(RuntimeError):
    """The stage changed since the patch's expected revision (compare and swap failed)."""


class StageRefused(ValueError):
    """The patch cannot be applied: unknown entity, locked entity or an invalid result."""


def apply(stage: Stage, patch: Patch) -> Stage:
    """The stage after `patch` (a new revision); raises StageConflict or StageRefused."""
    if patch.expected_revision != stage.revision:
        raise StageConflict(f"stage {stage.name} is at revision {stage.revision}, the patch was written against "
                            f"{patch.expected_revision}; read it again and rewrite the patch")
    ents: dict[str, dict[str, Any]] = {e.id: e.model_dump(mode="json") for e in stage.entities}
    order = [e.id for e in stage.entities]
    locked = {e.id for e in stage.entities if e.locked}

    def need(eid: str) -> dict[str, Any]:
        if eid not in ents:
            raise StageRefused(f"no entity {eid!r}")
        return ents[eid]

    for op in patch.ops:
        if isinstance(op, Add):
            if op.entity.id in ents:
                raise StageRefused(f"entity {op.entity.id!r} already exists")
            ents[op.entity.id] = op.entity.model_dump(mode="json")
            order.append(op.entity.id)
        elif isinstance(op, Remove):
            need(op.id)
            doomed = {op.id}
            grew = True
            while grew:
                more = {i for i, e in ents.items() if e.get("parent") in doomed} - doomed
                grew = bool(more)
                doomed |= more
            if len(doomed) > 1 and not op.cascade:
                raise StageRefused(f"entity {op.id} has children {sorted(doomed - {op.id})}; use cascade: true")
            if doomed & locked:
                raise StageRefused(f"locked entities cannot be removed: {sorted(doomed & locked)}")
            for i in doomed:
                ents.pop(i)
            order = [i for i in order if i not in doomed]
        elif isinstance(op, Update):
            cur = need(op.id)
            if op.id in locked and set(op.set) != {"locked"}:
                raise StageRefused(f"entity {op.id} is locked; unlock it first ({{\"locked\": false}})")
            if "id" in op.set and op.set["id"] != op.id:
                raise StageRefused("an entity's id cannot change; remove it and add a new one")
            ents[op.id] = {**cur, **op.set}
            if "locked" in op.set:
                (locked.add if op.set["locked"] else locked.discard)(op.id)
        elif isinstance(op, Reparent):
            cur = need(op.id)
            if op.id in locked:
                raise StageRefused(f"entity {op.id} is locked")
            if op.parent is not None:
                need(op.parent)
            ents[op.id] = {**cur, "parent": op.parent}
    data = {**stage.model_dump(mode="json"), "revision": stage.revision + 1,
            "entities": [ents[i] for i in order]}
    try:
        return Stage.model_validate(data)
    except ValueError as exc:
        raise StageRefused(f"the patched stage is not valid: {exc}") from exc


# --------------------------------------------------------------------------------------------------- store


class StageStore:
    """Stages on disk: `<root>/<name>/current.json`, every revision in `history/rev-<n>.json` and one line per
    commit in `log.jsonl`. Commits hold an exclusive lock, so several processes may write the same stage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def names(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/current.json"))

    def exists(self, name: str) -> bool:
        return (self._dir(name) / "current.json").is_file()

    @contextmanager
    def _lock(self, name: str) -> Iterator[None]:
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def _write(self, stage: Stage, note: str, author: str) -> None:
        d = self._dir(stage.name)
        (d / "history").mkdir(parents=True, exist_ok=True)
        text = stage.model_dump_json(indent=1)
        hist = d / "history" / f"rev-{stage.revision:05d}.json"
        if hist.exists():
            raise StageConflict(f"revision {stage.revision} of {stage.name} already exists")
        hist.write_text(text)
        tmp = d / f".current.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / "current.json")
        with open(d / "log.jsonl", "a") as f:
            f.write(json.dumps({"revision": stage.revision, "digest": stage.digest(), "note": note,
                                "author": author, "time": round(time.time(), 3),
                                "entities": len(stage.entities)}) + "\n")

    def create(self, stage: Stage, *, note: str = "created", author: str = "") -> Stage:
        with self._lock(stage.name):
            if self.exists(stage.name):
                raise StageConflict(f"stage {stage.name} already exists")
            stage = stage.model_copy(update={"revision": 0})
            self._write(stage, note, author)
            return stage

    def load(self, name: str, revision: int | None = None) -> Stage:
        d = self._dir(name)
        path = d / "current.json" if revision is None else d / "history" / f"rev-{revision:05d}.json"
        if not path.is_file():
            raise KeyError(f"no stage {name!r}" + ("" if revision is None else f" revision {revision}"))
        return Stage.model_validate_json(path.read_text())

    def commit(self, name: str, patch: Patch) -> Stage:
        with self._lock(name):
            new = apply(self.load(name), patch)
            self._write(new, patch.note, patch.author)
            return new

    def restore(self, name: str, revision: int, *, author: str = "") -> Stage:
        """Commit revision `revision` again as the newest revision."""
        with self._lock(name):
            cur, old = self.load(name), self.load(name, revision)
            new = old.model_copy(update={"revision": cur.revision + 1})
            self._write(Stage.model_validate(new.model_dump(mode="json")), f"restored revision {revision}", author)
            return new

    def history(self, name: str) -> list[dict[str, Any]]:
        path = self._dir(name) / "log.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


__all__ = ["Add", "Body", "Camera", "Collision", "Entity", "Human", "Light", "Mesh", "Mount", "Patch", "Pose",
           "Remove", "Reparent", "Robot", "STAGE_SCHEMA", "Shape", "Stage", "StageConflict", "StageRefused",
           "StageStore", "Update", "apply", "stage_problems"]
