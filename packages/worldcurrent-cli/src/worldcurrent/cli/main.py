"""The harness's `wc` subcommands (registered through commands.py; dispatched by core.cli).

    wc run CONFIG [--split SPLIT] [--output DIR] [--set key=value ...]
                                                     run an experiment config, print metrics
    wc plugins [--all] [--json]                      list plugins: contract version, maturity, licence
    wc schema [experiment|task|env-spec|candidate|memory-item]   print a contract's JSON Schema
    wc trace RUN_DIR [EPISODE_ID]                    summarise a run or one episode trace
    wc evolve run EVOLVE_CONFIG [--set key=value]    run (or resume) an improvement campaign
    wc evolve status|approve|rollback CONFIG_OR_DIR  inspect, approve a pending promotion, roll back
    wc skills ARCHIVE --out LIB [--tags ...] [--env E]   turn an improvement archive into a skill library
    wc conformance CONFIG [--agent] [--set ...]      run the conformance kit on a config's env (and agent)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from worldcurrent.core.registry import GROUPS

SCHEMAS = {
    "experiment": ("worldcurrent.core.config", "ExperimentConfig"),
    "task": ("worldcurrent.core.task", "TaskSpec"),
    "env-spec": ("worldcurrent.core.types", "EnvSpec"),
    "agent-spec": ("worldcurrent.core.agent", "AgentSpec"),
    "candidate": ("worldcurrent.evolve.candidate", "Candidate"),
    "memory-item": ("worldcurrent.core.memory", "MemoryItem"),
    "benchmark": ("worldcurrent.eval.benchmark", "Benchmark"),
}


def _run(args: argparse.Namespace) -> int:
    from worldcurrent.eval.experiment import run_experiment
    from worldcurrent.core.config import ExperimentConfig

    import yaml

    cfg = ExperimentConfig.load(args.config)
    if args.set:
        cfg = cfg.with_overrides({k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    done = [0]

    def progress(sample: dict) -> None:
        done[0] += 1
        print(f"[{done[0]}] {sample['task_id']} seed={sample['seed']} {sample['status']} ({sample['steps']} steps)",
              file=sys.stderr, flush=True)

    result = run_experiment(cfg, split=args.split, run_root=args.output, progress=progress)
    print(json.dumps({"run_dir": str(result.run_dir), "config_digest": cfg.digest(), **result.metrics}, indent=2))
    return 0


def _conformance(args: argparse.Namespace) -> int:
    from worldcurrent.eval.experiment import build_agent, env_factory
    from worldcurrent.core.config import ExperimentConfig
    from worldcurrent.core.registry import registry
    from worldcurrent.testing.conformance import check_agent, check_environment

    import yaml

    cfg = ExperimentConfig.load(args.config)
    if args.set:
        cfg = cfg.with_overrides({k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    bench = registry.build("benchmarks", cfg.benchmark.model_dump(exclude={"split"}))
    task = bench.instances(args.split or cfg.benchmark.split)[0]
    make_env = env_factory(cfg)
    reports = [check_environment(make_env, task, spec_budget_s=args.spec_budget)]
    if args.agent:
        env, agent = make_env(), build_agent(cfg)
        try:
            reports.append(check_agent(agent, env, task, deterministic=not args.nondeterministic))
        finally:
            env.close()
            close = getattr(agent, "close", None)
            if callable(close):
                close()
    for rep in reports:
        print(rep.summary())
    return 0 if all(r.ok for r in reports) else 1


def _skills(args: argparse.Namespace) -> int:
    from worldcurrent.evolve import experience

    return experience.main(args.rest)


def _learn(args: argparse.Namespace) -> int:
    from worldcurrent.core.learning import TrainJob
    from worldcurrent.learn.jobs import launch, run_job

    job = TrainJob.model_validate_json(Path(args.job).read_text())
    m = launch(job, python=args.python, timeout_s=args.timeout) if args.python else run_job(job)
    print(json.dumps({"out": job.out, "kind": m.kind, "role": m.role, "digest": m.digest,
                      "splits": sorted(m.provenance.splits()), "metrics": m.metrics}, indent=2, default=str))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from worldcurrent.learn import calibrate

    return calibrate.main(args.rest)


def _artifact(args: argparse.Namespace) -> int:
    from worldcurrent.core.artifact import ArtifactError, verify

    try:
        m = verify(args.path)
    except ArtifactError as exc:
        print(f"INVALID: {exc}")
        return 1
    print(json.dumps({"kind": m.kind, "role": m.role, "digest": m.digest, "builder": m.provenance.builder,
                      "splits": sorted(m.provenance.splits()), "sources": len(m.provenance.sources),
                      "metrics": m.metrics, "calibration": m.calibration}, indent=2, default=str))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    import subprocess

    if args.python:                               # the source's dependencies live in another venv (e.g. lerobot)
        return subprocess.run([args.python, "-m", "worldcurrent.learn.data", "ingest", args.spec]).returncode
    from worldcurrent.learn import data

    return data.main(["ingest", args.spec])


def _label(args: argparse.Namespace) -> int:
    from worldcurrent.learn.calibrate import label_run
    from worldcurrent.core.registry import registry

    from worldcurrent.core.store import resolve_refs

    args.judge, args.run = resolve_refs(args.judge), resolve_refs(args.run)    # store ids -> this machine's paths
    judge = registry.build("instruments", {"type": args.judge_type, "params": {"path": args.judge}})
    print(json.dumps(label_run(args.run, judge, args.judge, threshold=args.threshold), indent=2))
    return 0


def _contamination(args: argparse.Namespace) -> int:
    from worldcurrent.learn.data import contamination_report

    rep = contamination_report(args.external, args.protected, tolerance=args.tolerance)
    print(json.dumps(rep, indent=2))
    return 0 if rep["clean"] else 3


def _plugin(args: argparse.Namespace) -> int:
    from .scaffold import check_plugin, declaration_warnings, new_plugin

    if args.action == "new":
        root = new_plugin(args.group, args.name, args.dir)
        print(f"created {root}\n  install: uv pip install -e {root}\n  check:   uv run pytest {root}/tests "
              f"&& wc plugin check {args.group} {args.name}")
        return 0
    for warning in declaration_warnings(args.group, args.name):
        print(f"warning: {warning}")
    try:
        rep = check_plugin(args.group, args.name, json.loads(args.params) if args.params else None, args.experiment)
    except Exception as exc:  # noqa: BLE001 - a plugin that cannot be built fails its check
        print(f"cannot build {args.group} {args.name!r}: {type(exc).__name__}: {exc}")
        return 1
    print(rep.summary())
    return 0 if rep.ok else 1


def _plugins(args: argparse.Namespace) -> int:
    from worldcurrent.core.plugins import catalogue

    metas = [m for m in catalogue(GROUPS) if args.all or m.maturity != "test-double"]   # test doubles: --all only
    if args.json:
        print(json.dumps([m.model_dump(mode="json") for m in metas], indent=2))
        return 0
    print(f"{'group':<12} {'name':<30} {'contract':<20} {'maturity':<15} {'licence':<12} weights / notes")
    for m in metas:
        contract = m.contract or "undeclared"
        if m.compatible is False:
            contract += f" (core: {m.expected})"
        extra = "; ".join(x for x in (m.weights and f"weights: {m.weights}", m.notes) if x)
        print(f"{m.group:<12} {m.name:<30} {contract:<20} {m.maturity:<15} {m.licence or '-':<12} {extra}".rstrip())
    return 0


def _schema(args: argparse.Namespace) -> int:
    import importlib

    module, cls = SCHEMAS[args.name]
    print(json.dumps(getattr(importlib.import_module(module), cls).model_json_schema(), indent=2))
    return 0


def _trace(args: argparse.Namespace) -> int:
    from worldcurrent.core.trace import TraceStore

    store = TraceStore(args.run_dir)
    if args.episode is None:
        log_path = Path(args.run_dir) / "evallog.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
            print(json.dumps(log["results"], indent=2))
            print(Counter(s["status"] for s in log["samples"]))
        for eid in store.episode_ids():
            print(eid)
        return 0
    tr = store.read(args.episode)
    print(Counter(e["type"] for e in tr.events))
    end = tr.last("episode.end")
    print(json.dumps(end["payload"] if end else {}, indent=2))
    return 0


def _evolve_cfg(path: str, sets: list[str] | None = None):
    import yaml

    from worldcurrent.evolve.campaign import EvolveConfig

    p = Path(path)
    cfg = EvolveConfig.model_validate_json((p / "evolve_config.json").read_text()) if p.is_dir() else EvolveConfig.load(p)
    if sets:
        d = cfg.model_dump(mode="json")
        for k, v in (kv.split("=", 1) for kv in sets):
            node = d
            *parents, leaf = k.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = yaml.safe_load(v)
        cfg = EvolveConfig.model_validate(d)
    return cfg


def _evolve(args: argparse.Namespace) -> int:
    import getpass

    from worldcurrent.evolve import campaign

    cfg = _evolve_cfg(args.target, args.set)
    if args.action == "run":
        state = campaign.run_campaign(cfg, log=lambda m: print(m, flush=True))
        print(json.dumps({"cycles_done": state.cycles_done, "stopped": state.stopped, "out_dir": cfg.out_dir}, indent=2))
    elif args.action == "status":
        if args.publish:                                     # (re)write active.json, e.g. for an older campaign
            campaign.publish_active(cfg)
        print(json.dumps(campaign.status(cfg), indent=2, default=str))
    elif args.action == "shadow":
        from worldcurrent.agents.shadow import shadow_report
        from worldcurrent.eval.experiment import run_experiment

        if not args.candidate:
            raise SystemExit("wc evolve shadow CAMPAIGN --candidate ID [--split SPLIT]")
        res = run_experiment(campaign.shadow_config(cfg, args.candidate), split=args.split, label=f"{args.candidate}-shadow",
                             run_root=Path(cfg.out_dir) / "shadow")
        print(json.dumps({"run_dir": str(res.run_dir), "success_mean": res.metrics["success_mean"],
                          "shadow": shadow_report(res.run_dir)}, indent=2, default=str))
    elif args.action == "approve":
        print("promoted", campaign.approve(cfg, by=getpass.getuser()))
    elif args.action == "rollback":
        print("active is now", campaign.rollback(cfg, reason=args.reason or "manual rollback", by=getpass.getuser()))
    return 0


PASSTHROUGH = {"skills": _skills, "calibrate": _calibrate}   # commands with their own argument parser


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] in PASSTHROUGH:          # argparse.REMAINDER cannot start with an option
        return PASSTHROUGH[args_in[0]](argparse.Namespace(rest=args_in[1:]))
    p = argparse.ArgumentParser(prog="wc", description="worldcurrent Physical AI harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config")
    r.add_argument("--split")
    r.add_argument("--output")
    r.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="dotted-path override, e.g. --set benchmark.params.suite=libero_goal --set workers=16")
    r.set_defaults(fn=_run)
    pg = sub.add_parser("plugins", help="list plugins")
    pg.add_argument("--all", action="store_true", help="include the test doubles (testing.*)")
    pg.add_argument("--json", action="store_true")
    pg.set_defaults(fn=_plugins)
    s = sub.add_parser("schema", help="print a contract JSON Schema")
    s.add_argument("name", choices=sorted(SCHEMAS), nargs="?", default="experiment")
    s.set_defaults(fn=_schema)
    t = sub.add_parser("trace", help="summarise a run or an episode trace")
    t.add_argument("run_dir")
    t.add_argument("episode", nargs="?")
    t.set_defaults(fn=_trace)
    ev = sub.add_parser("evolve", help="improvement campaigns")
    ev.add_argument("action", choices=["run", "status", "approve", "rollback", "shadow"])
    ev.add_argument("target", help="evolve config YAML, or a campaign directory")
    ev.add_argument("--set", action="append", metavar="KEY=VALUE")
    ev.add_argument("--reason")
    ev.add_argument("--candidate", help="shadow: the candidate to run alongside the active version")
    ev.add_argument("--split", help="shadow: benchmark split (default: the config's)")
    ev.add_argument("--publish", action="store_true", help="status: also (re)write the campaign's active.json")
    ev.set_defaults(fn=_evolve)
    sk = sub.add_parser("skills", help="archive -> skill library (see worldcurrent.evolve.experience)", add_help=False)
    sk.add_argument("rest", nargs=argparse.REMAINDER)
    sk.set_defaults(fn=_skills)
    le = sub.add_parser("learn", help="run a learner job (TrainJob JSON) and seal its artifact (RFC-0004)")
    le.add_argument("job")
    le.add_argument("--python", help="interpreter to run the learner with (its venv); default: in this process")
    le.add_argument("--timeout", type=float)
    le.set_defaults(fn=_learn)
    ca = sub.add_parser("calibrate", help="calibrate a success judge against simulator outcomes", add_help=False)
    ca.add_argument("rest", nargs=argparse.REMAINDER)
    ca.set_defaults(fn=_calibrate)
    ig = sub.add_parser("ingest", help="ingest an external data source as a run (split external:<name>)")
    ig.add_argument("spec", help='JSON: {"source": {"type", "params"}, "name", "licence", "out", "image_every", "assume_success"}')
    ig.add_argument("--python", help="interpreter whose venv has the source's dependencies")
    ig.set_defaults(fn=_ingest)
    lb = sub.add_parser("label", help="label an ingested run's unlabeled episodes with a calibrated success judge")
    lb.add_argument("run")
    lb.add_argument("--judge", required=True, help="the judge's artifact (directory or store id)")
    lb.add_argument("--judge-type", default="judge.trace")
    lb.add_argument("--threshold", type=float)
    lb.set_defaults(fn=_label)
    co = sub.add_parser("contamination", help="check external runs against gate / held-out init states")
    co.add_argument("external", nargs="+")
    co.add_argument("--protected", nargs="+", default=["runs"])
    co.add_argument("--tolerance", type=float, default=1e-3)
    co.set_defaults(fn=_contamination)
    ar = sub.add_parser("artifact", help="verify a sealed artifact and print its manifest")
    ar.add_argument("path")
    ar.set_defaults(fn=_artifact)
    pl = sub.add_parser("plugin", help="scaffold a plugin package (new) or run its conformance check (check)")
    pl.add_argument("action", choices=["new", "check"])
    pl.add_argument("group", choices=["guard", "gate_test", "learner", "online_learner", "datasource", "judge", "generator", "policy", "agent"])
    pl.add_argument("name", help="plugin name (new) or registered name / module:attr (check)")
    pl.add_argument("--dir", default="plugins", help="new: where to create the package")
    pl.add_argument("--params", help="check: constructor params as JSON")
    pl.add_argument("--experiment", help="check: the experiment config to check on (default: MuJoCo planar reach if "
                                         "installed, else the test double)")
    pl.set_defaults(fn=_plugin)
    c = sub.add_parser("conformance", help="check a config's environment (and agent) against the contracts")
    c.add_argument("config")
    c.add_argument("--split")
    c.add_argument("--agent", action="store_true", help="also check the agent (runs two full episodes)")
    c.add_argument("--nondeterministic", action="store_true", help="skip the same-seed-same-episode check")
    c.add_argument("--spec-budget", type=float, default=2.0, help="seconds spec() may take (remote envs start a process)")
    c.add_argument("--set", action="append", metavar="KEY=VALUE")
    c.set_defaults(fn=_conformance)
    args = p.parse_args(args_in)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
