"""hylo.tools/v0.1 surfaces, hylo.stage/v0.1 stages and their store, the trace hash chain and run seals."""

import json
import os
import stat

import numpy as np
import pytest

from hylo.core.events import Event
from hylo.core.seal import SealRefused, chains, scan_visible, seal, verify_chain, verify_seal
from hylo.core.stage import (Body, Entity, Mount, Patch, Pose, Robot, Shape, Stage, StageConflict, StageRefused,
                             StageStore, apply)
from hylo.core.stage_cli import main as stage_main
from hylo.core.tooling import Help, Surface, ToolDef, ToolError, ToolOutput, check, obj
from hylo.core.trace import TraceStore

# ------------------------------------------------------------------------------------------------ tooling


def surface():
    def add(a):
        if a["x"] < 0:
            raise ToolError("negative", "x must not be negative", field="x")
        return {"sum": a["x"] + a.get("y", 0)}

    def boom(a):
        raise RuntimeError("kaput")

    return Surface([
        ToolDef("add", "read", Help("add numbers", returns="sum"),
                obj({"x": {"type": "number", "description": "x"},
                     "y": {"type": "integer", "minimum": 0, "maximum": 5, "description": "y"},
                     "mode": {"type": "string", "enum": ["a", "b"], "description": "mode"}}, ["x"]), add),
        ToolDef("pic", "act", Help("an image"), obj(), lambda a: ToolOutput({"n": 1}, [np.zeros((2, 2, 3), np.uint8)],
                                                                             ["blank"])),
        ToolDef("boom", "end", Help("fails"), obj(), boom),
    ])


def test_surface_checks_arguments_and_reports_errors_as_data():
    s = surface()
    assert s.call("add", {"x": 1, "y": 2}).result == {"sum": 3}
    assert s.call("add", {"x": 1.5, "y": None, "mode": None}).result == {"sum": 1.5}    # null = not given
    for args, code, field in [({}, "bad_arguments", "x"), ({"x": "1"}, "bad_arguments", "x"),
                              ({"x": 1, "y": 9}, "bad_arguments", "y"), ({"x": 1, "z": 1}, "bad_arguments", "z"),
                              ({"x": 1, "mode": "c"}, "bad_arguments", "mode"), ({"x": True}, "bad_arguments", "x"),
                              ({"x": -1}, "negative", "x")]:
        r = s.call("add", args)
        assert not r.ok and r.error["code"] == code and r.error["field"] == field, (args, r.error)
    assert s.call("nope", {}).error["code"] == "unknown_tool"
    r = s.call("boom", {})
    assert r.error["code"] == "tool_failed" and "kaput" in r.error["message"]
    r = s.call("pic", {})
    assert r.ok and r.images[0].shape == (2, 2, 3) and json.loads(r.text())["images"] == ["blank"]
    assert s.kind("pic") == "act" and s.kind("boom") == "end"


def test_surface_declarations_are_closed_described_and_digested():
    s = surface()
    spec = s.specs()[0]
    p = spec["parameters"]
    assert p["additionalProperties"] is False and p["properties"]["y"]["type"] == ["integer", "null"]
    assert p["properties"]["mode"]["enum"] == ["a", "b", None] and p["properties"]["x"]["type"] == "number"
    assert "Returns: sum" in spec["description"]
    d = s.digest()
    s.remove("boom")
    assert s.digest() != d and len(d) == 16
    with pytest.raises(ValueError):
        ToolDef("x", "read", Help("h"), obj({"a": {"type": "number"}}), lambda a: {})   # undescribed field
    assert check({"type": "array", "items": {"type": "number"}, "minItems": 2}, [1]) != []


# ------------------------------------------------------------------------------------------------ stage


def lab():
    return Stage(name="lab", entities=[
        Entity(id="table", shape=Shape(type="box", size=(0.4, 0.3, 0.02)), body=Body(dynamic=False),
               pose=Pose(pos=(0.5, 0, 0.4)), locked=True),
        Entity(id="arm", robot=Robot(pack="panda", mount=Mount(entity="table"))),
        Entity(id="cam", camera={"fovy_deg": 60}, pose=Pose.yaw((1.0, 0.0, 1.0), 3.14159)),
    ])


def test_stage_validation():
    with pytest.raises(ValueError, match="parent"):
        Stage(name="s", entities=[Entity(id="a", parent="ghost")])
    with pytest.raises(ValueError, match="cycle"):
        Stage(name="s", entities=[Entity(id="a", parent="b"), Entity(id="b", parent="a")])
    with pytest.raises(ValueError, match="duplicate"):
        Stage(name="s", entities=[Entity(id="a"), Entity(id="a")])
    with pytest.raises(ValueError, match="one of shape"):
        Entity(id="a", shape=Shape(type="sphere", size=(0.1,)), camera={})
    with pytest.raises(ValueError, match="unit quaternion"):
        Pose(quat=(1, 1, 0, 0))
    assert Pose(quat=(1.0005, 0, 0, 0)).quat == (1.0, 0, 0, 0)               # near unit: normalised
    st = lab()
    assert [e.id for e in st.ordered()] == ["table", "arm", "cam"] and st.get("arm").kind == "robot"


def test_patches_compare_and_swap_and_respect_locks(tmp_path):
    store = StageStore(tmp_path)
    store.create(lab())
    st = store.commit("lab", Patch(expected_revision=0, ops=[
        {"op": "add", "entity": {"id": "cube", "parent": "table", "shape": {"type": "box", "size": [0.02] * 3},
                                 "body": {}}},
        {"op": "update", "id": "cam", "set": {"label": "overview"}}], note="add a cube"))
    assert st.revision == 1 and st.get("cube").parent == "table" and st.get("cam").label == "overview"
    with pytest.raises(StageConflict):
        store.commit("lab", Patch(expected_revision=0, ops=[{"op": "remove", "id": "cube"}]))
    for op, msg in [({"op": "update", "id": "table", "set": {"label": "x"}}, "locked"),
                    ({"op": "remove", "id": "cube", "cascade": False}, None),
                    ({"op": "remove", "id": "ghost"}, "no entity"),
                    ({"op": "add", "entity": {"id": "arm"}}, "already exists"),
                    ({"op": "update", "id": "cam", "set": {"shape": {"type": "box", "size": [1, 1, 1]}}}, "not valid")]:
        if msg is None:
            continue
        with pytest.raises(StageRefused, match=msg):
            apply(store.load("lab"), Patch(expected_revision=1, ops=[op]))
    with pytest.raises(StageRefused, match="mount target"):    # removing the table would strand the mounted arm
        apply(store.load("lab"), Patch(expected_revision=1, ops=[{"op": "update", "id": "table",
                                                                   "set": {"locked": False}},
                                                                  {"op": "remove", "id": "table", "cascade": True}]))
    st = store.restore("lab", 0)
    assert st.revision == 2 and "cube" not in [e.id for e in st.entities]
    assert [h["revision"] for h in store.history("lab")] == [0, 1, 2]
    assert store.load("lab", 1).get("cube") and store.history("lab")[2]["digest"] == store.history("lab")[0]["digest"]


def test_stage_cli(tmp_path, capsys):
    root = str(tmp_path)
    src = tmp_path / "s.json"
    src.write_text(lab().model_dump_json())
    assert stage_main(["--root", root, "new", "demo", "--from", str(src)]) == 0
    patch = tmp_path / "p.json"
    patch.write_text(json.dumps({"expected_revision": 0, "ops": [{"op": "remove", "id": "cam"}]}))
    assert stage_main(["--root", root, "patch", "demo", str(patch)]) == 0
    assert stage_main(["--root", root, "patch", "demo", str(patch)]) == 1          # stale revision
    assert stage_main(["--root", root, "list"]) == 0
    assert "demo  revision 1  2 entities" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ chain and seal


def write_run(tmp_path):
    store = TraceStore(tmp_path / "run")
    for i in range(5):
        store.write(Event(type="agent.tool.turn" if i == 2 else "step", episode_id="ep1", seq=i, source="test",
                          payload={"i": i, "img": np.ones((200, 200, 3), np.uint8) * i}))
    store.close()
    return tmp_path / "run"


def test_hash_chain_detects_changes_and_survives_reopening(tmp_path):
    run = write_run(tmp_path)
    store = TraceStore(run)                              # appending later continues the chain
    store.write(Event(type="late", episode_id="ep1", seq=5, source="test", payload={}))
    store.close()
    path = run / "episodes" / "ep1.jsonl"
    r = verify_chain(path)
    assert r["ok"] and r["lines"] == r["chained"] == 6
    assert store.read("ep1").events[3]["payload"]["img"].shape == (200, 200, 3)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + [lines[2].replace('"i":2', '"i":7')] + lines[3:]) + "\n")
    r = verify_chain(path)
    assert not r["ok"] and r["broken_at"] == 3 and "changed" in r["why"]
    path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert verify_chain(path)["broken_at"] == 3                # a removed line


def test_seal_and_verify(tmp_path):
    run = write_run(tmp_path)
    doc = seal(run)
    assert doc["chains"]["ep1"]["lines"] == 5 and len(doc["files"]) >= 2
    assert not os.stat(run / "episodes" / "ep1.jsonl").st_mode & stat.S_IWUSR
    assert verify_seal(run)["ok"]
    with pytest.raises(SealRefused):
        seal(run)
    blob = next((run / "blobs").glob("*.npy"))
    blob.chmod(0o644)
    blob.write_bytes(b"x")
    (run / "extra.txt").write_text("x")
    r = verify_seal(run)
    assert not r["ok"] and r["changed"] == [f"blobs/{blob.name}"] and r["added"] == ["extra.txt"]


def test_seal_refuses_privileged_keys_in_model_visible_events(tmp_path):
    store = TraceStore(tmp_path / "run")
    store.write(Event(type="agent.tool.result", episode_id="e", seq=0, source="t",
                      payload={"result": {"object_poses": [1, 2]}}))
    store.write(Event(type="step", episode_id="e", seq=1, source="t", payload={"object_poses": [1]}))
    store.close()
    assert [x["line"] for x in scan_visible(tmp_path / "run")] == [1]
    with pytest.raises(SealRefused, match="privileged"):
        seal(tmp_path / "run")
    assert chains(tmp_path / "run")["e"]["ok"]
