"""Conformance for learning-side plugins: offline learners, online learners, data sources, success judges.

They run against small runs made on the spot with `testing.setup.check_experiment()` (MuJoCo planar reach with the
IK controller when installed, the test double otherwise; or any experiment you pass), so a plugin can be checked
without any dataset. A learner that needs specific inputs (images, a policy venv) is checked on its own
data with the same functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.artifact import SplitLeak, verify
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.datasource import ExternalEpisode
from worldcurrent.core.learning import TrainJob
from worldcurrent.testing.conformance import ConformanceReport, _run
from worldcurrent.testing.setup import check_experiment

def sample_runs(root: str | Path, splits: tuple[str, ...] = ("search", "search_val", "gate"), *,
                experiment: ExperimentConfig | None = None) -> Path:
    cfg = check_experiment(experiment)
    for sp in splits:
        run_experiment(cfg, split=sp, run_root=Path(root), label=sp)
    return Path(root)


def check_learner(name: str, params: dict[str, Any] | None = None, *, inputs: list[str] | None = None,
                  experiment: ExperimentConfig | None = None) -> ConformanceReport:
    from .jobs import run_job

    rep = ConformanceReport(subject=f"learner {name}")
    tmp = Path(tempfile.mkdtemp(prefix="wc-learner-"))
    runs = inputs or [str(sample_runs(tmp / "runs", experiment=experiment))]
    params = dict(params or {})
    from .online import is_online
    from worldcurrent.core.registry import registry

    online = is_online(registry.build("learners", {"type": name, "params": {}}))
    if online:
        params.setdefault("experiment", check_experiment(experiment).model_dump(mode="json"))
        params.setdefault("iterations", 1)
    state: dict[str, Any] = {}

    def seals() -> str:
        m = run_job(TrainJob(learner=name, inputs=runs, out=str(tmp / "out"), params=params))
        verify(tmp / "out")
        state["m"] = m
        return f"{m.kind} / {m.role}, digest {m.digest}"

    def provenance() -> str:
        m = state["m"]
        if not m.provenance.sources:
            raise AssertionError("the manifest lists no sources")
        bad = [s for s in m.provenance.splits() if s.startswith(("gate", "heldout"))]
        if bad:
            raise AssertionError(f"read protected splits {bad}")
        return f"splits {sorted(m.provenance.splits())}"

    def refuses() -> None:
        try:
            run_job(TrainJob(learner=name, inputs=runs, out=str(tmp / "leak"), splits=("gate",), params=params))
        except SplitLeak:
            return
        raise AssertionError("a job asking for the gate split was not refused")

    def resumes() -> None:
        before = state["m"].digest
        if run_job(TrainJob(learner=name, inputs=runs, out=str(tmp / "out"), params=params)).digest != before:
            raise AssertionError("re-running the same job did not reuse the sealed artifact")

    _run(rep, ("online: " if online else "") + "runs and seals a verifiable artifact", seals)
    _run(rep, "provenance lists development data only", provenance, skip="m" not in state)
    _run(rep, "refuses gate / held-out splits", refuses)
    _run(rep, "an identical job reuses the artifact", resumes, skip="m" not in state)
    return rep


def check_datasource(source: Any, *, limit: int = 3) -> ConformanceReport:
    from .data import ingest

    rep = ConformanceReport(subject=f"datasource {getattr(source, 'name', type(source).__name__)}")
    eps: list[Any] = []

    def meta() -> str:
        for f in ("name", "origin", "licence"):
            if not getattr(source, f, None):
                raise AssertionError(f"source.{f} is required (provenance and licence travel with the data)")
        return f"licence {source.licence}"

    def episodes() -> str:
        for i, ep in enumerate(source.episodes()):
            if not isinstance(ep, ExternalEpisode) or not ep.frames:
                raise AssertionError(f"episode {i} is not a non-empty ExternalEpisode")
            for fr in ep.frames:
                for cam, img in fr.images.items():
                    if img.dtype.name != "uint8" or img.ndim != 3:
                        raise AssertionError(f"camera {cam}: images are HxWxC uint8")
            eps.append(ep)
            if len(eps) >= limit:
                break
        if not eps:
            raise AssertionError("no episodes")
        return f"{len(eps)} episodes"

    def ingests() -> str:
        r = ingest(eps, tempfile.mkdtemp(prefix="wc-ingest-"), name="conformance", origin=source.origin, licence=source.licence)
        return f"run {r['split']}, {r['frames']} frames"

    _run(rep, "declares name, origin and licence", meta)
    _run(rep, "yields ExternalEpisodes with HxWxC uint8 images", episodes)
    _run(rep, "ingests into a sealed external run", ingests, skip=not eps)
    return rep


def check_judge(judge: Any, *, runs: str | None = None, experiment: ExperimentConfig | None = None) -> ConformanceReport:
    from worldcurrent.core.trace import TraceStore

    rep = ConformanceReport(subject=f"judge {getattr(judge, 'name', type(judge).__name__)}")

    def judges() -> str:
        root = Path(runs) if runs else sample_runs(tempfile.mkdtemp(prefix="wc-judge-"), ("search",), experiment=experiment)
        rd = next(p.parent for p in root.rglob("evallog.json"))
        store = TraceStore(rd)
        ps = [judge.judge(store.read(e)).p_success for e in store.episode_ids()[:5]]
        if not all(0.0 <= p <= 1.0 for p in ps):
            raise AssertionError(f"p_success outside [0, 1]: {ps}")
        return f"p_success {[round(p, 2) for p in ps]}"

    _run(rep, "judges recorded episodes with p_success in [0, 1]", judges)
    return rep
