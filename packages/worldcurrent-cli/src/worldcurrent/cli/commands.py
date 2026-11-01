"""The harness's subcommands of `wc`, one entry point each in the "worldcurrent.commands" group (core.cli).

Each forwards to main.py's parser, so `wc run ...` and `python -m worldcurrent.cli.main run ...` are the same.
"""

from __future__ import annotations

from typing import Callable

from .main import main

HELP = {
    "run": "run an experiment config",
    "plugins": "list plugins",
    "schema": "print a contract JSON Schema",
    "trace": "summarise a run or an episode trace",
    "evolve": "improvement campaigns: run, status, approve, rollback, shadow",
    "skills": "archive -> skill library",
    "learn": "run a learner job (TrainJob JSON) and seal its artifact",
    "calibrate": "calibrate a success judge against simulator outcomes",
    "ingest": "ingest an external data source as a run",
    "label": "label an ingested run with a calibrated success judge",
    "contamination": "check external runs against gate / held-out init states",
    "artifact": "verify a sealed artifact and print its manifest",
    "plugin": "scaffold a plugin package (new) or check one (check)",
    "conformance": "check a config's environment (and agent) against the contracts",
}


def _command(name: str) -> Callable[[list[str]], int]:
    def command(argv: list[str]) -> int:
        return main([name, *argv])

    command.__name__, command.__doc__ = name, HELP[name]
    return command


run, plugins, schema, trace, evolve, skills, learn, calibrate, ingest, label, contamination, artifact, plugin, \
    conformance = (_command(n) for n in HELP)
