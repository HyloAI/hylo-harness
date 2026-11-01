"""A dependency-free learner for tests and conformance: summarises development runs into a sealed artifact."""

from __future__ import annotations

import json
from pathlib import Path

from hylo.core.artifact import ArtifactManifest, Provenance, development_runs, write_manifest


class SummaryLearner:
    name = "hylo.testing.summary@0.1"

    def fit(self, job) -> ArtifactManifest:
        sources = development_runs(job.inputs, job.splits)
        status = [s["status"] for rd, _ in sources for s in json.loads((rd / "evallog.json").read_text())["samples"]]
        out = Path(job.out)
        out.mkdir(parents=True, exist_ok=True)
        rate = sum(s == "success" for s in status) / max(1, len(status))
        (out / "summary.json").write_text(json.dumps({"episodes": len(status), "success_rate": rate}))
        return write_manifest(out, kind="other", role="behaviour", metrics={"episodes": len(status), "success_rate": rate,
                                                                          "kp": float(job.params.get("kp", 1.0))},
                              provenance=Provenance(builder=self.name, sources=[r for _, r in sources],
                                                    params=dict(job.params)))
