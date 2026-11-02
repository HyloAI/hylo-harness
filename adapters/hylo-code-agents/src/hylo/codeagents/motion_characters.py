"""Target characters of the motion studio: humanoid robots (MJCF) and rigged human characters (Blender), each
described by a short mapping file in motion_characters/<name>.json.

A mapping names, for every role of the studio's skeleton (pelvis, chest, head and, per side, hip, knee, ankle, toe,
shoulder, elbow, wrist), where that point is on the character:

- kind "mjcf" (a robot or a simulated humanoid): `bodies` maps a role to a body, or to [body, [x, y, z]] (an
  offset in the body frame, e.g. the toe in front of the ankle); `feet` are the two foot bodies (contact, sole
  flat); `clearance` keeps hands and elbows that far from the body's midline (metres); the asset is a MuJoCo
  Menagerie folder at a pinned commit (`menagerie`: a manifest beside this module with a digest per file) or a
  builtin model.
- kind "rig" (a skinned character, driven in Blender): `joints` maps a role to a bone end ("Bone.head" or
  "Bone.tail"); `drive` says which bones move: `pelvis` (placed and turned with the hips), `spine` (bones that
  share the turn from the hips to the chest), `neck`, and per side upper_arm, forearm, thigh, shin, foot (turned
  so the segment between their joints points like the source's); `attach` pins a bone's head to another bone's
  end (an IK foot of the original rig); `colours` overrides material base colours (sRGB, by material name).
  `height_m` scales the character to that height. The asset is an archive
  fetched from its original page (`archive`: url, sha256, members to extract).

Every mapping records `licence` and `credit`. Robots are fetched from MuJoCo Menagerie and characters from their
authors' uploads, verified by digest, into the asset cache; nothing is vendored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

CHAR_DIR = Path(__file__).with_name("motion_characters")
SIDE_ROLES = ("hip", "knee", "ankle", "toe", "shoulder", "elbow", "wrist")
ROLES = ("pelvis", "chest", "head", *[f"{s}_{r}" for s in ("left", "right") for r in SIDE_ROLES])
ROLE_KP = {"pelvis": "Hips", "chest": "Chest", "head": "Head",
           **{f"{s}_{r}": f"{S}{k}" for s, S in (("left", "Left"), ("right", "Right")) for r, k in (
               ("hip", "Leg"), ("knee", "Shin"), ("ankle", "Foot"), ("toe", "ToeBase"), ("shoulder", "Arm"),
               ("elbow", "ForeArm"), ("wrist", "Hand"))}}
WEIGHTS = {"pelvis": 3.0, "chest": 1.5, "head": 1.0, "hip": 1.0, "knee": 1.5, "ankle": 2.0, "toe": 1.0,
           "shoulder": 1.0, "elbow": 1.5, "wrist": 2.0}
DRIVE_SIDE = {"upper_arm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist"), "thigh": ("hip", "knee"),
              "shin": ("knee", "ankle"), "foot": ("ankle", "toe")}
BONE_END = re.compile(r"^.+\.(head|tail)$")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) hylo-motion-studio"


# ------------------------------------------------------------------------------------------------- mappings
def names(kind: str | None = None) -> list[str]:
    """The characters with a mapping file (optionally only one kind: "mjcf" or "rig")."""
    out = []
    for p in sorted(CHAR_DIR.glob("*.json")):
        if kind is None or json.loads(p.read_text()).get("kind") == kind:
            out.append(p.stem)
    return out


ROBOTS_DEFAULT = ("g1", "h1", "t1", "apollo", "n1")


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    p = CHAR_DIR / f"{name}.json"
    if not p.is_file():
        raise ValueError(f"unknown character {name!r}; characters: {', '.join(names())}")
    return p.read_text()


def load(name: str) -> dict[str, Any]:
    """A character's mapping (validated; raises ValueError listing the problems)."""
    data = json.loads(_load(name))
    problems = validate(data)
    if problems:
        raise ValueError(f"character {name!r}: " + "; ".join(problems))
    return data


def _point(v: Any) -> bool:
    if isinstance(v, str):
        return bool(v)
    return (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], list)
            and len(v[1]) == 3 and all(isinstance(x, (int, float)) for x in v[1]))


def validate(data: dict[str, Any]) -> list[str]:
    """Problems with a mapping's structure (an empty list when it is fine). Whether the bodies or bones exist in
    the model is checked when the model is loaded (motion_sim `validate`, the Blender script)."""
    p: list[str] = []
    if not isinstance(data, dict):
        return ["a mapping is a JSON object"]
    for k in ("name", "title", "kind", "licence", "credit", "asset"):
        if not data.get(k):
            p.append(f"missing {k!r}")
    kind = data.get("kind")
    if kind not in ("mjcf", "rig"):
        p.append(f"kind must be 'mjcf' or 'rig', not {kind!r}")
        return p
    h = data.get("height_m", 1.7)
    if not isinstance(h, (int, float)) or not 0.3 < h < 3.0:
        p.append(f"height_m {h!r} is not a plausible height in metres")
    asset = data.get("asset") or {}
    if kind == "mjcf":
        bodies = data.get("bodies") or {}
        missing = [r for r in ROLES if r not in bodies]
        if missing:
            p.append(f"bodies: no entry for {missing}")
        bad = [r for r, v in bodies.items() if r in ROLES and not _point(v)]
        if bad:
            p.append(f"bodies: {bad} must be a body name or [body, [x, y, z]]")
        extra = sorted(set(bodies) - set(ROLES))
        if extra:
            p.append(f"bodies: unknown roles {extra}")
        feet = data.get("feet")
        if not (isinstance(feet, list) and len(feet) == 2 and all(isinstance(f, str) and f for f in feet)):
            p.append("feet must be [left foot body, right foot body]")
        if not (asset.get("menagerie") or asset.get("builtin")):
            p.append("asset needs 'menagerie' (a manifest file) or 'builtin'")
        if asset.get("menagerie") and not (Path(__file__).with_name(asset["menagerie"])).is_file():
            p.append(f"asset manifest {asset['menagerie']} not found beside the module")
        if asset.get("menagerie") and not asset.get("xml"):
            p.append("asset needs 'xml' (the model file in the folder)")
        for k, v in (data.get("clearance") or {}).items():
            if k not in ("wrist", "elbow") or not isinstance(v, (int, float)) or not 0 <= v < 1:
                p.append(f"clearance {k!r}: {v!r} (wrist / elbow, metres)")
    else:
        joints = data.get("joints") or {}
        missing = [r for r in ROLES if r not in joints]
        if missing:
            p.append(f"joints: no entry for {missing}")
        bad = [r for r, v in joints.items() if not (isinstance(v, str) and BONE_END.match(v))]
        if bad:
            p.append(f"joints: {bad} must be 'Bone.head' or 'Bone.tail'")
        drive = data.get("drive") or {}
        need = ["pelvis", "spine", "neck", *[f"{s}_{r}" for s in ("left", "right") for r in DRIVE_SIDE]]
        miss = [r for r in need if r not in drive]
        if miss:
            p.append(f"drive: no entry for {miss}")
        if "spine" in drive and not (isinstance(drive["spine"], list) and drive["spine"]
                                     and all(isinstance(b, str) for b in drive["spine"])):
            p.append("drive.spine must be a non-empty list of bone names")
        bones = [b for r, b in drive.items() if r != "spine"]
        if any(not isinstance(b, str) or not b for b in bones):
            p.append("drive: every role except spine names one bone")
        allb = bones + list(drive.get("spine") or [])
        if len(set(allb)) != len(allb):
            p.append("drive: a bone is driven twice")
        for b, end in (data.get("attach") or {}).items():
            if not (isinstance(end, str) and BONE_END.match(end)):
                p.append(f"attach {b!r}: {end!r} must be 'Bone.head' or 'Bone.tail'")
        arc = asset.get("archive") or {}
        if not (str(arc.get("url", "")).startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", str(arc.get("sha256")))
                and isinstance(arc.get("members"), dict) and arc["members"]):
            p.append("asset.archive needs an https url, a sha256 and members")
        if asset.get("file") not in (arc.get("members") or {}):
            p.append("asset.file must be one of the archive members")
        if asset.get("texture") and asset["texture"] not in (arc.get("members") or {}):
            p.append("asset.texture must be one of the archive members")
    return p


# ------------------------------------------------------------------------------------------------- assets
_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{folder}/{path}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_root() -> Path:
    """The asset cache ($HYLO_ASSETS, default ~/.cache/hylo/assets)."""
    return Path(os.environ.get("HYLO_ASSETS") or Path.home() / ".cache" / "hylo" / "assets").expanduser()


def menagerie_dir(manifest: str | Path) -> Path:
    """Where a Menagerie folder of `manifest` lives in the asset cache."""
    m = json.loads(Path(manifest).read_text())
    return asset_root() / "mujoco_menagerie" / m["commit"][:12] / m["folder"]


def menagerie_assets(manifest: str | Path) -> Path:
    """A MuJoCo Menagerie model folder (its LICENSE kept) at the commit pinned in `manifest` (repo, commit, folder,
    sha256 per file), fetched on first use from the original repository into the asset cache and verified file by
    file; never vendored. Offline: copy that folder (the files listed in the manifest) to menagerie_dir() by hand."""
    manifest = Path(manifest)
    m = json.loads(manifest.read_text())
    final = menagerie_dir(manifest)
    marker = final / ".hylo-verified"
    if marker.is_file():
        return final
    if final.is_dir():
        bad = [r for r, d in m["sha256"].items() if not (final / r).is_file() or _sha256(final / r) != d]
        if bad:
            raise RuntimeError(f"{final}: {len(bad)} files differ from the recorded digests (e.g. {bad[:3]}); "
                               f"delete the folder to fetch it again")
        marker.write_text(m["commit"] + "\n")
        return final
    tmp = final.with_name(f"{final.name}.partial-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)

    def get(item: tuple[str, str]) -> None:
        rel, digest = item
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _RAW.format(repo=m["repo"], commit=m["commit"], folder=m["folder"], path=rel)
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        if _sha256(dest) != digest:
            raise RuntimeError(f"digest mismatch for {url}")

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(get, m["sha256"].items()))
    except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"fetching {m['folder']} from MuJoCo Menagerie failed ({exc}); offline, copy "
                           f"{m['folder']} of mujoco_menagerie at {m['commit']} to {final}") from exc
    (tmp / ".hylo-verified").write_text(m["commit"] + "\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.rename(final)
    except OSError:                                   # another process finished first
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def archive_dir(asset: dict[str, Any]) -> Path:
    arc = asset["archive"]
    return asset_root() / "characters" / arc["sha256"][:12]


def archive_assets(asset: dict[str, Any]) -> Path:
    """Fetch a character archive from its original page (verified by sha256), extract the listed members into
    the asset cache and return that folder."""
    arc = asset["archive"]
    final = archive_dir(asset)
    if (final / ".hylo-verified").is_file() and all((final / k).is_file() for k in arc["members"]):
        return final
    final.mkdir(parents=True, exist_ok=True)
    zpath = final / "archive.zip"
    if not zpath.is_file() or hashlib.sha256(zpath.read_bytes()).hexdigest() != arc["sha256"]:
        req = urllib.request.Request(arc["url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
        except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
            raise RuntimeError(f"fetching {arc['url']} failed ({exc}); offline, put the archive at {zpath}") from exc
        if hashlib.sha256(data).hexdigest() != arc["sha256"]:
            raise RuntimeError(f"digest mismatch for {arc['url']}")
        zpath.write_bytes(data)
    with zipfile.ZipFile(zpath) as z:
        for dest, member in arc["members"].items():
            if "/" in dest or dest.startswith("."):
                raise ValueError(f"member target {dest!r} must be a plain file name")
            with z.open(member) as src, open(final / dest, "wb") as out:
                shutil.copyfileobj(src, out)
    zpath.unlink()                                    # only the extracted members are kept
    (final / ".hylo-verified").write_text(arc["sha256"] + "\n")
    return final


# ------------------------------------------------------------------------------------------------- specs
def mjcf_spec(name: str, work: Path) -> dict[str, Any]:
    """The robot description motion_sim needs (see its docstring) for an MJCF character."""
    c = load(name)
    if c["kind"] != "mjcf":
        raise ValueError(f"{name} is a rigged character (kind {c['kind']}), not an MJCF model")
    asset = c["asset"]
    if asset.get("builtin") == "hylo_humanoid":
        from hylo.tools.text_motion import HUMANOID_XML
        work.mkdir(parents=True, exist_ok=True)
        xml = work / "hylo_humanoid.xml"
        xml.write_text(HUMANOID_XML)
        floor = False
    else:
        xml = menagerie_assets(Path(__file__).with_name(asset["menagerie"])) / asset["xml"]
        floor = True
    targets = {}
    for role in ROLES:
        v = c["bodies"][role]
        body, off = (v, None) if isinstance(v, str) else (v[0], v[1])
        targets[ROLE_KP[role]] = [body, off, WEIGHTS[role.split("_", 1)[-1]]]
    clear = c.get("clearance") or {}
    clearance = {f"{S}{k}": clear[r] for S in ("Left", "Right") for r, k in (("wrist", "Hand"), ("elbow", "ForeArm"))
                 if r in clear}
    return {"name": name, "xml": str(xml), "floor": floor, "targets": targets, "feet": list(c["feet"]),
            "clearance": clearance, "scaling": "bones", "height_m": c.get("height_m", 1.7)}


def rig_spec(name: str) -> dict[str, Any]:
    """What the Blender script needs to drive a rigged character."""
    c = load(name)
    if c["kind"] != "rig":
        raise ValueError(f"{name} is an MJCF model, not a rigged character")
    folder = archive_assets(c["asset"])
    return {"name": name, "title": c["title"], "file": str(folder / c["asset"]["file"]),
            "texture": str(folder / c["asset"]["texture"]) if c["asset"].get("texture") else None,
            "height_m": float(c.get("height_m", 1.7)),
            "joints": {ROLE_KP[r]: v for r, v in c["joints"].items()}, "drive": c["drive"],
            "attach": c.get("attach") or {}, "colours": c.get("colours") or {}}


def kind(name: str) -> str:
    return load(name)["kind"]


def title(name: str) -> str:
    return load(name)["title"]


def camera_distance(name: str) -> float:
    return float(load(name).get("camera_distance", 3.0))


def credits(chars: list[str]) -> dict[str, dict[str, Any]]:
    """Source and licence of each character (for sources.json)."""
    out = {}
    for n in chars:
        c = load(n)
        a = c["asset"]
        if a.get("menagerie"):
            m = json.loads(Path(__file__).with_name(a["menagerie"]).read_text())
            src = {"repo": f"https://github.com/{m['repo']}", "commit": m["commit"], "folder": m["folder"],
                   "files_verified": len(m["sha256"])}
        elif a.get("archive"):
            src = {"page": a["archive"].get("page"), "url": a["archive"]["url"], "sha256": a["archive"]["sha256"],
                   "members": list(a["archive"]["members"].values())}
        else:
            src = {"builtin": a.get("builtin")}
        out[n] = {"title": c["title"], "kind": c["kind"], "licence": c["licence"], "credit": c["credit"], **src}
    return out


__all__ = ["CHAR_DIR", "ROBOTS_DEFAULT", "ROLES", "ROLE_KP", "archive_assets", "asset_root",
           "camera_distance", "credits", "kind", "load", "menagerie_assets", "menagerie_dir", "mjcf_spec", "names",
           "rig_spec", "title", "validate"]
