"""`hylo preflight`: check, before a tool-agent run, everything that can be checked without running it.

    hylo preflight CONFIG [--out DIR] [--no-model] [--no-env]
    hylo preflight --model ollama:qwen3.5:35b-a3b [--vision]

Each check prints PASS / FAIL / SKIP with what it found; the exit code is 1 when any check failed, and with `--out`
the results go to DIR/preflight.json (the run directory, so the evidence of a run includes its preflight):

```text
  config        the experiment file loads and its plugins (env, agent, benchmark) are installed
  source        the git commit and whether the working tree has uncommitted changes (a warning, not a failure)
  storage       free space where the run writes (>= 2 GiB) and in the asset cache (>= 1 GiB)
  model         a live probe of the agent's chat model: a text reply, an image (vision models), a tool call with
                valid JSON arguments, and an answer that uses a tool result
  environment   the env builds and publishes a RobotSpec (tool agents need one), and its first reset works
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

PROBE_TOOL = {"name": "add_numbers", "description": "Add two numbers and return the sum.",
              "parameters": {"type": "object", "properties": {"a": {"type": "number", "description": "first"},
                                                              "b": {"type": "number", "description": "second"}},
                             "required": ["a", "b"], "additionalProperties": False}}


def _check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        out = fn()
        status = out.pop("status", "pass")
    except Exception as exc:  # noqa: BLE001 - a failed check is a result
        out, status = {"error": f"{type(exc).__name__}: {exc}"[:500]}, "fail"
    return {"check": name, "status": status, "seconds": round(time.monotonic() - t0, 2), **out}


def probe_model(spec: Any) -> dict[str, Any]:
    """Text, image, tool call and tool-result round trip against a chat model spec."""
    from hylo.reasoners.llm.chat import build_chat, with_tools

    model = with_tools(build_chat(spec))
    out: dict[str, Any] = {"model": model.name}
    r = model.chat([{"role": "user", "content": "Reply with the single word READY."}], max_tokens=400)
    out["text"] = "READY" in r.text.upper()
    if getattr(model, "vision", False):
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :, 0] = 220                                  # a red square
        r = model.chat([{"role": "user", "content": "What colour fills this image? One word.", "images": [img]}],
                       max_tokens=400)
        out["image"] = "red" in r.text.lower()
    msgs = [{"role": "user", "content": "Use the add_numbers tool to add 17 and 25."}]
    r = model.chat(msgs, tools=[PROBE_TOOL], max_tokens=800)
    call = r.tool_calls[0] if r.tool_calls else None
    out["tool_call"] = bool(call and call.name == "add_numbers" and call.arguments
                            and {call.arguments.get("a"), call.arguments.get("b")} == {17, 25})
    if call is not None:
        msgs += [r.message(), {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": '{"sum": 42}'}]
        r2 = model.chat(msgs, tools=[PROBE_TOOL], max_tokens=800)
        out["tool_result"] = "42" in r2.text
    else:
        out["tool_result"] = False
    out["status"] = "pass" if all(v for k, v in out.items() if k != "model") else "fail"
    return out


def _git() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"status": "skip", "note": "git not available"}
    if not sha:
        return {"status": "skip", "note": "not a git checkout"}
    return {"commit": sha, "dirty": dirty, "status": "warn" if dirty else "pass"}


def _storage(out_dir: Path) -> dict[str, Any]:
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    run_free = shutil.disk_usage(out_dir).free / 2 ** 30
    cache = Path(os.environ.get("HYLO_ASSETS") or Path.home() / ".cache" / "hylo" / "assets").expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    cache_free = shutil.disk_usage(cache).free / 2 ** 30
    ok = run_free >= 2 and cache_free >= 1
    return {"run_free_gib": round(run_free, 1), "cache_free_gib": round(cache_free, 1), "status": "pass" if ok else "fail"}


def preflight(config: str | None, *, out: str | None = None, model: Any = None, check_model: bool = True,
              check_env: bool = True) -> list[dict[str, Any]]:
    from hylo.core.config import ExperimentConfig
    from hylo.core.registry import registry

    results: list[dict[str, Any]] = []
    cfg = None
    if config:
        def load() -> dict[str, Any]:
            nonlocal cfg
            cfg = ExperimentConfig.load(config)
            for group, node in (("envs", cfg.env), ("agents", cfg.agent), ("benchmarks", cfg.benchmark)):
                registry.resolve(group, node.type) if hasattr(registry, "resolve") else registry.get(group, node.type)
            return {"name": cfg.name, "env": cfg.env.type, "agent": cfg.agent.type, "benchmark": cfg.benchmark.type}
        results.append(_check("config", load))
    results.append(_check("source", _git))
    results.append(_check("storage", lambda: _storage(Path(out or (cfg.output_dir if cfg else "runs")))))
    spec = model
    if spec is None and cfg is not None:
        spec = (cfg.agent.params or {}).get("model")
    if check_model and spec is not None:
        results.append(_check("model", lambda: probe_model(spec)))
    elif check_model:
        results.append({"check": "model", "status": "skip", "note": "no model in the config"})
    if check_env and cfg is not None:
        def env() -> dict[str, Any]:
            e = registry.build("envs", {"type": cfg.env.type, "params": cfg.env.params or {}})
            try:
                s = e.spec()
                info = {"env": s.name, "robot": s.robot is not None, "cameras": list(s.observation.cameras)}
                b = registry.build("benchmarks", {"type": cfg.benchmark.type, "params": cfg.benchmark.params or {}})
                inst = b.instances(cfg.benchmark.split)[0]
                obs = e.reset(inst)
                info["first_reset"] = bool(obs.instruction or inst.spec.instruction)
                if cfg.agent.type.startswith(("tools.", "code.")) and s.robot is None:
                    info["status"] = "fail"
                    info["error"] = "tool agents need EnvSpec.robot"
                return info
            finally:
                e.close()
        results.append(_check("environment", env))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "preflight.json").write_text(json.dumps({"schema": "hylo.preflight/v0.1", "time": time.time(),
                                                              "config": config, "checks": results}, indent=1))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """check a config, storage and a live model before a tool-agent run"""
    ap = argparse.ArgumentParser(prog="hylo preflight", description="check a tool-agent run before starting it")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--out", help="write DIR/preflight.json")
    ap.add_argument("--model", help="probe this chat model (e.g. ollama:qwen3.5:35b-a3b) instead of the config's")
    ap.add_argument("--vision", action="store_true", help="with --model: probe images too")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-env", action="store_true")
    a = ap.parse_args(argv)
    if not a.config and not a.model:
        ap.error("give a CONFIG or --model")
    model = None
    if a.model:
        backend, _, name = a.model.partition(":")
        model = {"backend": backend, "model": name, "vision": a.vision}
    res = preflight(a.config, out=a.out, model=model, check_model=not a.no_model, check_env=not a.no_env)
    for r in res:
        detail = {k: v for k, v in r.items() if k not in ("check", "status")}
        print(f"[{r['status'].upper():4}] {r['check']:12} {json.dumps(detail, default=str)[:300]}")
    return 1 if any(r["status"] == "fail" for r in res) else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
