"""Conformance, sealed artifacts and the store: contract checks, tamper detection and deletion protection.

What it shows
    1. Conformance: `hylo conformance CONFIG --agent` on a MuJoCo planar-reach config, in-process and with the
       simulator behind the EnvServer (stdio), then the same kit (hylo.testing.conformance) on a deliberately broken
       environment: it must find the faults.
    2. Sealed artifacts: a run recorded with `hylo run`, a dataset of (state, action) pairs derived from its traces
       and sealed with an `artifact.json` manifest (content digest + provenance). `hylo artifact` verifies it; one
       changed byte makes it INVALID; restoring the byte makes it valid again. Sealing anything built from gate or
       held-out data is refused (the split guard).
    3. The store: `hylo store ls` gives the run and the artifact global ids; a config can name the artifact as
       `artifact:<name>@<digest>` and the id resolves to this machine's path only while the content still matches;
       `hylo store rm` refuses to delete the run while the artifact built from it refers to it.

Command (from the harness repository root)
    uv run python examples/core/05_conformance_artifacts.py [--quick] [--out DIR]

Typical runtime: about 3 s (--quick: about 3 s) on a 128-core CPU, no GPU.
Requires: hylo-env-mujoco (hylo-bench).
Output: <out>/result.json, <out>/store (a store root holding the run and the artifact).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from hylo.core.artifact import ArtifactError, Provenance, SourceRef, SplitLeak, referenced_artifacts, verify, write_manifest
from hylo.core.registry import registry
from hylo.core.trace import TraceStore
from hylo.testing.conformance import check_environment

SLUG = "conformance_artifacts"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"schema_version": "hylo.experiment/v0.1", "name": "core-conformance",
          "benchmark": {"type": "mujoco/reach-v0", "split": "validation", "params": {"validation": 5}},
          "env": {"type": "mujoco.planar_reach"}, "agent": {"type": "baseline.planar_ik", "params": {"kp": 0.3}},
          "workers": 4}


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def hylo(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """The `hylo` command of this venv."""
    return subprocess.run([sys.executable, "-m", "hylo.core.cli", *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def short(msg: str, out: Path, n: int = 95) -> str:
    """A message without the output directory's absolute path, cut to n characters."""
    return msg.replace(str(out) + "/", "").strip()[:n]


def parse_checks(stdout: str) -> dict:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("[")]
    tags = [ln[1:5].strip() for ln in lines]
    return {"ok": tags.count("ok"), "skip": tags.count("skip"), "fail": tags.count("FAIL"), "lines": lines}


class BrokenEnv:
    """Wraps the real simulator and breaks two promises of its spec: reset ignores the seed, restore does nothing."""

    def __init__(self) -> None:
        self.inner = registry.build("envs", {"type": "mujoco.planar_reach"})
        self.rng = np.random.default_rng()                           # not the task's seed

    def spec(self):
        return self.inner.spec()

    def reset(self, task):
        obs = self.inner.reset(task)
        self.inner.data.qpos[:] += self.rng.normal(0, 0.01, 2)
        return obs.model_copy(update={"state": {**obs.state, "qpos": self.inner.data.qpos.copy()}})

    def step(self, action):
        return self.inner.step(action)

    def snapshot(self) -> bytes:
        return self.inner.snapshot()

    def restore(self, blob: bytes) -> None:
        pass

    def close(self) -> None:
        self.inner.close()


def extract_pairs(run_dir: Path, dest: Path) -> int:
    """A derived artifact: every (state, executed action) pair of a run's successful episodes."""
    log = json.loads((run_dir / "evallog.json").read_text())
    store = TraceStore(run_dir)
    X, Y = [], []
    for s in log["samples"]:
        if s["status"] != "success":
            continue
        tr = store.read(s["episode_id"])
        obs = [e["payload"]["state"] for e in tr.of_type("observation")]
        acts = [e["payload"]["data"] for e in tr.of_type("action")]
        for o, act in zip(obs, acts):
            X.append(np.concatenate([o["qpos"], o["qvel"], o["tip_pos"], o["target_pos"]]))
            Y.append(act)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "pairs.npz", x=np.array(X), y=np.array(Y))
    (dest / "README.txt").write_text("x: qpos, qvel, tip_pos, target_pos; y: executed action (joint deltas)\n")
    return len(X)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    a = ap.parse_args()
    try:
        registry.reference("envs", "mujoco.planar_reach")
    except KeyError:
        sys.exit("'mujoco.planar_reach' is not registered: install hylo-env-mujoco (hylo-bench) into this venv")
    out = a.out.resolve()
    if (out / "store").exists():
        shutil.rmtree(out / "store")                                   # a fresh store every run
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**CONFIG, "benchmark": {**CONFIG["benchmark"], "params": {"validation": 5 if a.quick else 25}}}
    cfg_path = out / "experiment.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    remote_path = out / "experiment_remote.yaml"
    remote_path.write_text(yaml.safe_dump({**cfg, "env": {**cfg["env"], "remote": {}}}, sort_keys=False))
    result: dict = {"example": SLUG, "quick": a.quick}

    print("== 1. conformance")
    rows, conf = [], {}
    for name, path in (("in-process", cfg_path), ("EnvServer (stdio)", remote_path)):
        t0 = time.perf_counter()
        p = hylo("conformance", str(path), "--agent", "--spec-budget", "10")
        c = parse_checks(p.stdout)
        conf[name] = {"exit_code": p.returncode, "wall_s": time.perf_counter() - t0, **c}
        rows.append([f"hylo conformance ({name})", p.returncode, c["ok"], c["skip"], c["fail"]])
    task = registry.build("benchmarks", {"type": "mujoco/reach-v0"}).instances("validation")[0]
    broken = check_environment(BrokenEnv, task)
    fails = [c for c in broken.checks if not c.passed]
    conf["broken env"] = {"ok": broken.ok, "failed": [f"{c.name}: {c.detail}" for c in fails]}
    rows.append(["conformance kit on BrokenEnv", "-", sum(c.passed and not c.skipped for c in broken.checks),
                 sum(c.skipped for c in broken.checks), len(fails)])
    table(rows, ["subject", "exit", "ok", "skip", "FAIL"])
    for line in conf["in-process"]["lines"]:
        print("    " + line)
    for c in fails:
        print(f"    BrokenEnv FAIL: {c.name}: {c.detail[:90]}")
    result["conformance"] = conf

    print("\n== 2. a run, a sealed artifact derived from it, tamper detection")
    store_root = out / "store"
    p = hylo("run", str(cfg_path), "--output", str(store_root / "runs"))
    if p.returncode != 0:
        print(p.stderr)
        return 1
    run = json.loads(p.stdout)
    run_dir = Path(run["run_dir"])
    run_id = run_dir.name
    art = store_root / "artifacts" / "reach-pairs"
    n_pairs = extract_pairs(run_dir, art)
    t0 = time.perf_counter()
    m = write_manifest(art, kind="dataset", role="data", name="reach-pairs",
                       provenance=Provenance(builder="examples.core.extract_pairs@0.1",
                                             sources=[SourceRef(kind="run", id=run_id, split="validation",
                                                                path=str(run_dir), n_episodes=run["n_episodes"])]),
                       metrics={"pairs": n_pairs, "source_success_mean": run["success_mean"]})
    seal_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    verify(art)
    verify_ms = (time.perf_counter() - t0) * 1e3
    ok1 = hylo("artifact", str(art))
    blob = art / "pairs.npz"
    raw = bytearray(blob.read_bytes())
    pos = len(raw) // 2
    raw[pos] ^= 0x01                                                   # flip one bit in the middle of the data
    blob.write_bytes(bytes(raw))
    bad = hylo("artifact", str(art))
    raw[pos] ^= 0x01
    blob.write_bytes(bytes(raw))
    ok2 = hylo("artifact", str(art))
    try:
        leak_dir = out / "leaky"
        leak_dir.mkdir(exist_ok=True)
        (leak_dir / "x.txt").write_text("built from gate data\n")
        write_manifest(leak_dir, kind="dataset", role="data", provenance=Provenance(
            builder="examples.core.leaky@0.1", sources=[SourceRef(kind="run", id="r", split="gate")]))
        leak = "sealed (the guard did not fire)"
    except SplitLeak as exc:
        leak = f"refused: {exc}"
    table([
        ["seal (write artifact.json)", f"digest {m.digest}, {n_pairs} pairs from {run['n_episodes']} episodes, {seal_ms:.1f} ms"],
        ["hylo artifact (as sealed)", f"exit {ok1.returncode}, verify {verify_ms:.1f} ms"],
        ["hylo artifact (1 bit flipped)", f"exit {bad.returncode}: {short(bad.stdout, out)}"],
        ["hylo artifact (bit restored)", f"exit {ok2.returncode}"],
        ["seal a gate-split build", short(leak, out)],
    ], ["step", "measured"])
    result["artifact"] = {"digest": m.digest, "pairs": n_pairs, "seal_ms": seal_ms, "verify_ms": verify_ms,
                          "exit_sealed": ok1.returncode, "exit_tampered": bad.returncode, "tampered_message": bad.stdout.strip(),
                          "exit_restored": ok2.returncode, "gate_split_seal": leak}

    print("\n== 3. the store: ids, references, deletion protection")
    ls = hylo("store", "--root", str(store_root), "ls", "--json")
    entries = json.loads(ls.stdout)
    for e in entries:
        print(f"    {e['id']}")
    art_id = next(e["id"] for e in entries if e["kind"] == "artifact")
    run_ref = next(e["id"] for e in entries if e["kind"] == "run")
    show = json.loads(hylo("store", "--root", str(store_root), "show", run_ref).stdout)
    os.environ["HYLO_STORE"] = str(store_root)                         # what `hylo store init` would record
    resolved = referenced_artifacts({"agent": {"params": {"data": art_id}}})
    raw[pos] ^= 0x01
    blob.write_bytes(bytes(raw))
    try:
        referenced_artifacts({"agent": {"params": {"data": art_id}}})
        ref_tampered = "resolved (not detected)"
    except ArtifactError as exc:
        ref_tampered = f"refused: {short(str(exc), out, 90)}"
    raw[pos] ^= 0x01
    blob.write_bytes(bytes(raw))
    rm = hylo("store", "--root", str(store_root), "rm", run_ref)
    table([
        ["entries", f"{sum(e['kind'] == 'run' for e in entries)} run, {sum(e['kind'] == 'artifact' for e in entries)} artifact"],
        ["referrers of the run", ", ".join(show["referrers"]) or "-"],
        ["config names the artifact id", f"resolves to {Path(next(iter(resolved))).relative_to(out)}"],
        ["same id after a 1-bit change", ref_tampered],
        ["hylo store rm <run>", f"exit {rm.returncode}: {short(rm.stderr, out)}"],
    ], ["check", "measured"])
    result["store"] = {"entries": [e["id"] for e in entries], "run_referrers": show["referrers"],
                       "artifact_id_resolves": bool(resolved), "artifact_id_after_tamper": ref_tampered,
                       "rm_run_exit": rm.returncode, "rm_run_message": rm.stderr.strip()}

    good = (conf["in-process"]["exit_code"] == 0 and conf["EnvServer (stdio)"]["exit_code"] == 0 and not broken.ok
            and ok1.returncode == 0 and bad.returncode == 1 and ok2.returncode == 0 and leak.startswith("refused")
            and ref_tampered.startswith("refused") and rm.returncode == 1)
    result["all_expected"] = good
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nall checks behaved as expected: {good}\nresult: {out / 'result.json'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
