"""Read-only catalog over the runs directory: runs, campaigns, artifacts, skill libraries, lesson books, memories.

The trace store is the only source of truth; nothing is copied into a database. Scans are cached by file mtime, so
new runs (from the live sandbox or a job) appear on the next request.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from worldcurrent.eval.scoring import paired_compare
from worldcurrent.core.artifact import MANIFEST

SKIP_DIRS = {"blobs", "episodes", "__pycache__", "vlm-cache", "registry"}


def _walk(root: Path, filename: str, max_depth: int = 8) -> list[Path]:
    """Directories under root containing `filename` (does not descend into trace internals)."""
    out = []
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and len(Path(dirpath).parts) - base < max_depth]
        if filename in filenames:
            out.append(Path(dirpath))
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


class Index:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._runs: dict[str, tuple[float, dict[str, Any]]] = {}     # evallog path -> (mtime, summary)
        self._paths: dict[str, Path] = {}                            # run_id -> run dir
        self._lock = threading.Lock()

    def rel(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    # ---------------------------------------------------------------- runs
    def _summary(self, rd: Path) -> dict[str, Any] | None:
        log = _read_json(rd / "evallog.json")
        if not isinstance(log, dict) or "samples" not in log:
            return None
        man = _read_json(rd / "manifest.json") or {}
        samples = log["samples"]
        done = [s for s in samples if s.get("status") in ("success", "failure")]
        results = log.get("results") or {}
        bench = man.get("benchmark") or log.get("benchmark") or {}
        run_id = log.get("run_id") or rd.name
        return {"run_id": run_id, "id": f"run:{run_id}", "path": self.rel(rd), "group": self.rel(rd.parent),
                "split": bench.get("split"), "benchmark": bench.get("id"), "agent": (log.get("agent") or {}).get("name"),
                "name": (man.get("config") or {}).get("name"), "created_at": man.get("created_at") or rd.stat().st_mtime,
                "n": len(samples), "success": (sum(s["status"] == "success" for s in done) / len(done)) if done else None,
                "safety_mean": results.get("safety_interventions_mean"),
                "tasks": len({s.get("task_id") for s in samples})}

    def runs(self) -> list[dict[str, Any]]:
        with self._lock:
            found = _walk(self.root, "evallog.json")
            out = []
            for rd in found:
                f = rd / "evallog.json"
                m = f.stat().st_mtime
                cached = self._runs.get(str(f))
                if cached is None or cached[0] != m:
                    s = self._summary(rd)
                    if s is None:
                        continue
                    cached = (m, s)
                    self._runs[str(f)] = cached
                self._paths[cached[1]["run_id"]] = rd
                out.append(cached[1])
        out.sort(key=lambda r: str(r["created_at"]), reverse=True)
        return out

    def run_dir(self, run_id: str) -> Path:
        if run_id not in self._paths:
            self.runs()
        if run_id not in self._paths:
            raise KeyError(run_id)
        return self._paths[run_id]

    def run(self, run_id: str) -> dict[str, Any]:
        rd = self.run_dir(run_id)
        log = _read_json(rd / "evallog.json")
        man = _read_json(rd / "manifest.json") or {}
        return {"summary": self._summary(rd), "results": log.get("results"), "agent": log.get("agent"),
                "config": man.get("config"), "samples": [
                    {k: s.get(k) for k in ("key", "task_id", "seed", "episode_id", "status", "steps")} for s in log["samples"]]}

    def compare(self, a: str, b: str) -> dict[str, Any]:
        """Two runs paired on (task, seed); b is the candidate, a the baseline (exact one-sided McNemar)."""
        la = json.loads((self.run_dir(a) / "evallog.json").read_text())
        lb = json.loads((self.run_dir(b) / "evallog.json").read_text())

        def per_key(log):
            return {s["key"]: (s.get("scores", {}).get("success", {}) or {}).get("value") for s in log["samples"]}

        ka, kb = per_key(la), per_key(lb)
        pc = paired_compare(kb, ka)
        eps_a = {s["key"]: s for s in la["samples"]}
        eps_b = {s["key"]: s for s in lb["samples"]}
        tasks: dict[str, dict[str, list[float]]] = {}
        pairs = []
        for k in ka:
            if k in kb and ka[k] is not None and kb[k] is not None:
                t = eps_a[k]["task_id"]
                tasks.setdefault(t, {"a": [], "b": []})
                tasks[t]["a"].append(ka[k])
                tasks[t]["b"].append(kb[k])
                pairs.append({"key": k, "task_id": t, "seed": eps_a[k]["seed"], "a": ka[k], "b": kb[k],
                              "episode_a": eps_a[k]["episode_id"], "episode_b": eps_b[k]["episode_id"]})
        return {"a": self._summary(self.run_dir(a)), "b": self._summary(self.run_dir(b)), "paired": pc.model_dump(),
                "per_task": [{"task_id": t, "n": len(v["a"]), "a": sum(v["a"]) / len(v["a"]), "b": sum(v["b"]) / len(v["b"])}
                             for t, v in sorted(tasks.items())], "pairs": pairs}

    # ---------------------------------------------------------------- campaigns
    def campaigns(self) -> list[dict[str, Any]]:
        out = []
        for d in _walk(self.root, "campaign_state.json"):
            st = _read_json(d / "campaign_state.json") or {}
            cfg = _read_json(d / "evolve_config.json") or {}
            n_arch = sum(1 for _ in open(d / "archive.jsonl")) if (d / "archive.jsonl").exists() else 0
            out.append({"id": self.rel(d), "name": st.get("name"), "cycles_done": st.get("cycles_done"),
                        "stopped": st.get("stopped"), "elapsed_s": st.get("elapsed_s"), "candidates": n_arch,
                        "proposer": (cfg.get("proposer") or {}).get("type"),
                        "pending": (d / "pending_approval.json").exists()})
        return sorted(out, key=lambda c: c["id"])

    def campaign(self, cid: str) -> dict[str, Any]:
        d = (self.root / cid).resolve()
        if self.root not in d.parents or not (d / "campaign_state.json").exists():
            raise KeyError(cid)
        archive = [json.loads(l) for l in (d / "archive.jsonl").read_text().splitlines() if l.strip()] \
            if (d / "archive.jsonl").exists() else []
        cycles = []
        for c in sorted(d.glob("cycle-*")):
            rep = _read_json(c / "cycle_report.json")
            if rep:
                cycles.append({"cycle": c.name, "parent_id": rep.get("parent_id"), "promoted": rep.get("promoted"),
                               "heldout": rep.get("heldout"), "elapsed_s": rep.get("elapsed_s"),
                               "candidates": [e["candidate"]["id"] for e in rep.get("entries", [])],
                               "diagnosis": {k: rep.get("diagnosis", {}).get(k) for k in ("success_rate", "bucket_rate", "n")}})
        reg = d / "registry"
        ledger = [json.loads(l) for l in (reg / "ledger.jsonl").read_text().splitlines() if l.strip()] \
            if (reg / "ledger.jsonl").exists() else []
        active = ((reg / "ACTIVE").read_text().strip() or None) if (reg / "ACTIVE").exists() else None
        return {"id": cid, "state": _read_json(d / "campaign_state.json"), "config": _read_json(d / "evolve_config.json"),
                "archive": archive, "cycles": cycles, "ledger": ledger, "active": active,
                "pending": _read_json(d / "pending_approval.json")}

    def loop_summary(self) -> dict[str, Any]:
        """Counts along the improvement cascade over every campaign, and each campaign's strongest gate result."""
        stages = {"proposed": 0, "screened": 0, "gated": 0, "passed": 0, "promoted": 0}
        bests: list[dict[str, Any]] = []
        for c in self.campaigns():
            d = self.root / c["id"]
            if not (d / "archive.jsonl").exists():
                continue
            best = None
            for line in (d / "archive.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                stages["proposed"] += 1
                stages["screened"] += e.get("stage_reached", 0) >= 2
                stages["gated"] += e.get("stage_reached", 0) >= 3
                stages["passed"] += e.get("verdict") in ("gate_passed", "promoted")
                stages["promoted"] += e.get("verdict") == "promoted"
                looks = [s for s in e.get("stages", []) if s.get("stage") == 3 and s.get("success") is not None]
                if looks:
                    last = looks[-1]
                    eff = last["success"] - last["parent_success"]
                    if best is None or eff > best["effect"]:
                        best = {"campaign": c["id"], "candidate": e["candidate"]["id"], "hypothesis": e.get("hypothesis"),
                                "effect": eff, "success": last["success"], "parent_success": last["parent_success"],
                                "wins": last.get("wins"), "losses": last.get("losses"), "p_value": last.get("p_value"),
                                "notes": last.get("notes"), "verdict": e.get("verdict"), "patch": e["candidate"].get("patch")}
            if best:
                bests.append(best)
        return {"stages": stages, "best": bests}

    def latest_live(self) -> dict[str, Any] | None:
        live = [r for r in self.runs() if r["split"] == "playground"]
        if not live:
            return None
        r = live[0]
        log = _read_json(self.run_dir(r["run_id"]) / "evallog.json") or {}
        s = (log.get("samples") or [None])[-1]
        return {"run_id": r["run_id"], "name": r["name"], **({"episode_id": s["episode_id"], "status": s["status"],
                "steps": s["steps"], "task_id": s["task_id"], "seed": s["seed"], "instruction": (s.get("playground") or {}).get("instruction")} if s else {})}

    # ---------------------------------------------------------------- experience and artifacts
    def artifacts(self) -> list[dict[str, Any]]:
        out = []
        for d in _walk(self.root, MANIFEST):
            m = _read_json(d / MANIFEST) or {}
            prov = m.get("provenance") or {}
            ident = f"artifact:{m.get('name') or d.name}@{m.get('digest')}" if m.get("digest") else None
            out.append({"id": ident, "path": self.rel(d), "kind": m.get("kind"), "role": m.get("role"), "digest": m.get("digest"),
                        "builder": prov.get("builder"), "created_at": prov.get("created_at"),
                        "splits": sorted({s.get("split") for s in prov.get("sources", []) if s.get("split")}),
                        "sources": prov.get("sources", []), "metrics": m.get("metrics"),
                        "calibration": m.get("calibration")})
        return sorted(out, key=lambda a: a.get("created_at") or 0, reverse=True)

    def skill_libraries(self) -> list[dict[str, Any]]:
        libs: dict[str, list[dict[str, Any]]] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if "@" in fn and fn.endswith(".json"):
                    s = _read_json(Path(dirpath) / fn)
                    if isinstance(s, dict) and "trigger" in s and "status" in s:
                        libs.setdefault(self.rel(Path(dirpath)), []).append(s)
        return [{"path": k, "skills": v} for k, v in sorted(libs.items())]

    def lesson_books(self) -> list[dict[str, Any]]:
        out = []
        for d in _walk(self.root, "lessons.json"):
            lessons = _read_json(d / "lessons.json") or []
            ops = (d / "ops.jsonl").read_text().splitlines() if (d / "ops.jsonl").exists() else []
            out.append({"path": self.rel(d), "lessons": lessons, "ops": [json.loads(o) for o in ops[-50:] if o.strip()]})
        return out

    def memories(self) -> list[dict[str, Any]]:
        out = []
        for d in _walk(self.root, "bank.json"):
            rep = _read_json(d / "build_report.json") or {}
            meta = _read_json(d / "bank.json") or {}
            out.append({"path": self.rel(d), "segments": rep.get("segments") or meta.get("n") or len(meta.get("segments", [])),
                        "episodes": rep.get("episodes"), "tasks": rep.get("tasks"), "digest": rep.get("digest"),
                        "nn_same_task_other_episode": rep.get("nn_same_task_other_episode"),
                        "nn_other_task": rep.get("nn_other_task"), "splits": rep.get("splits")})
        return out


Catalog = Index
