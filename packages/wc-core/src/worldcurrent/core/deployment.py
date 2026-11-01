"""The active version of an improvement campaign, as a file (wc.active/v0.1).

The campaign (wc-evolve) writes `<campaign>/active.json` whenever its active version changes: at start, on a
promotion, on an approval, on a rollback. Anything that deploys the campaign's current behaviour (the policy server
in wc-serve) reads only this file, so the deploying side needs neither the improvement code nor the candidate
registry, and the file can be copied to another machine.

    {"schema_version": "wc.active/v0.1", "campaign": "libero10-steering", "version": "cand-1a2b3c4d5e",
     "agent": {"type": "lerobot.steered", "params": {...}}, "patch": {"agent.params.kp": 2.0},
     "lineage": ["cand-1a2b3c4d5e"], "updated_at": 1793400000.0}

`version` is None while the base configuration is active. `agent` is the fully resolved agent spec (base config
with the lineage's patches applied), so a reader builds it with the registry and nothing else.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .contract import Contract, JsonAny

ACTIVE_SCHEMA = "wc.active/v0.1"
ACTIVE_FILE = "active.json"


class ActiveVersion(Contract):
    schema_version: Literal["wc.active/v0.1"] = ACTIVE_SCHEMA
    campaign: str
    version: str | None                       # candidate id; None = the base configuration
    agent: dict[str, JsonAny]                 # resolved agent spec {type, params}
    patch: dict[str, JsonAny] = Field(default_factory=dict)   # the lineage's composed config patch
    lineage: tuple[str, ...] = ()             # active first, back to the first promotion
    updated_at: float = Field(default_factory=time.time)


def write_active(campaign_dir: str | Path, active: ActiveVersion) -> Path:
    """Atomic write (temp file + rename): a reader never sees a half-written file."""
    path = Path(campaign_dir) / ACTIVE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(active.model_dump_json(indent=2))
    os.replace(tmp, path)
    return path


def read_active(campaign_dir: str | Path) -> ActiveVersion:
    path = Path(campaign_dir) / ACTIVE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: not a campaign directory, or written before {ACTIVE_SCHEMA} "
                                f"(run `wc evolve status {campaign_dir} --publish` to write it)")
    data: dict[str, Any] = json.loads(path.read_text())
    return ActiveVersion.model_validate(data)


__all__ = ["ACTIVE_FILE", "ACTIVE_SCHEMA", "ActiveVersion", "read_active", "write_active"]
