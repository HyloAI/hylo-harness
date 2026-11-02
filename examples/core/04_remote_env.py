"""Remote components: the same experiment in-process, over stdio (EnvServer / AgentServer) and over WebSocket.

What it shows
    1. One MuJoCo planar-reach experiment run five ways:
           in-process             env and agent in this process
           stdio env              `env: {..., remote: {}}`: the simulator in a subprocess (EnvServer protocol)
           stdio env + agent      the agent also in a subprocess (`agent: {type: remote, ...}`, AgentServer)
           ws env                 `env: {type: remote.ws, params: {url}}`: a separate server process on a port
           ws env + agent         both components behind WebSocket servers
       and checks that every episode has the same status, step count and executed actions (bit for bit), and
       whether the config digest changes (where a component runs is not part of what is measured).
    2. Per-call latency: the same 'step' call made directly on the environment, through the stdio proxy and
       through the WebSocket proxy (median and 95th percentile), and the bytes one step puts on the wire.
    3. The two wire codecs: an observation with a 224x224 RGB image encoded with core.codec (JSON + base64, the
       EnvServer protocol) and with core.remote.wire (msgpack-numpy, the policy-server protocol): size and time.

Command (from the harness repository root)
    uv run python examples/core/04_remote_env.py [--quick] [--out DIR]

Typical runtime: about 8 s (--quick: about 3 s) on a 128-core CPU, no GPU.
Requires: hylo-env-mujoco (hylo-bench), websockets>=13, msgpack.
Output: <out>/result.json and one run directory per mode.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hylo.core import codec
from hylo.core.config import ExperimentConfig
from hylo.core.registry import registry
from hylo.core.remote import wire
from hylo.core.remote.client import RemoteEnvironment
from hylo.core.remote.ws import WSEnvironment
from hylo.core.trace import TraceStore
from hylo.core.types import Action, Observation
from hylo.eval.experiment import run_experiment

SLUG = "remote_env"
ROOT = Path(__file__).resolve().parents[2]
ENV, AGENT = "mujoco.planar_reach", {"type": "baseline.planar_ik", "params": {"kp": 0.3}}


def table(rows: list[list], headers: list[str]) -> None:
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


class Server:
    """`python -m hylo.core.remote.ws KIND REF` in its own process, on a free port."""

    def __init__(self, kind: str, ref: str, params: dict | None = None) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "hylo.core.remote.ws", kind, ref,
                                      "--params", json.dumps(params or {}), "--port", "0"],
                                     stderr=subprocess.PIPE, text=True)
        for line in self.proc.stderr:                                   # "[ws] serving on ws://127.0.0.1:PORT"
            m = re.search(r"serving on (ws://\S+)", line)
            if m:
                self.url = m.group(1)
                break
        else:
            raise RuntimeError(f"ws server for {ref} did not start (exit {self.proc.poll()})")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)


def actions_of(run_dir: Path, log: dict) -> tuple[dict[str, np.ndarray], float]:
    """Executed actions per episode, and the seconds spent inside episodes (trace wall clock, start to end):
    without process start-up and connection set-up."""
    store = TraceStore(run_dir)
    acts, inside = {}, 0.0
    for s in log["samples"]:
        tr = store.read(s["episode_id"])
        acts[s["key"]] = np.array([e["payload"]["data"] for e in tr.of_type("action")])
        inside += tr.last("episode.end")["wall_time"] - tr.first("episode.start")["wall_time"]
    return acts, inside


def step_latency(env, task, n: int) -> list[float]:
    """Seconds per env.step, measured from the caller's side, over n steps (episodes restarted as needed)."""
    rng = np.random.default_rng(0)
    out: list[float] = []
    env.reset(task)
    while len(out) < n:
        a = Action(data=rng.uniform(-1, 1, 2))
        t0 = time.perf_counter()
        r = env.step(a)
        out.append(time.perf_counter() - t0)
        if r.terminated or r.truncated or len(out) % 150 == 0:
            env.reset(task)
    return out


def codec_compare(reps: int) -> dict:
    rng = np.random.default_rng(0)
    obs = Observation(state={"qpos": rng.normal(size=7), "eef": rng.normal(size=7)},
                      images={"front": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)}, instruction="pick", time=0.0)
    payload = obs.model_dump(mode="python")
    out = {}
    for name, enc, dec in (("core.codec (JSON + base64)", lambda p: json.dumps(codec.encode(p)).encode(),
                            lambda b: codec.decode(json.loads(b))),
                           ("remote.wire (msgpack-numpy)", wire.packb, wire.unpackb)):
        blob = enc(payload)
        t0 = time.perf_counter()
        for _ in range(reps):
            back = dec(enc(payload))
        dt = (time.perf_counter() - t0) / reps
        assert np.array_equal(back["images"]["front"], obs.images["front"])
        out[name] = {"bytes": len(blob), "roundtrip_ms": dt * 1e3}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer episodes and calls (< 60 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "examples" / "core" / SLUG)
    a = ap.parse_args()
    try:
        registry.reference("envs", ENV)
    except KeyError:
        sys.exit(f"{ENV!r} is not registered: install hylo-env-mujoco (hylo-bench) into this venv")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_val, n_calls = (3, 300) if a.quick else (25, 3000)

    env_srv, agent_srv = Server("env", ENV), Server("agent", AGENT["type"], AGENT["params"])
    try:
        base = dict(benchmark={"type": "mujoco/reach-v0", "params": {"validation": n_val}}, agent=AGENT, workers=1)
        modes = {
            "in-process": ExperimentConfig(name="core-remote", env={"type": ENV}, **base),
            "stdio env": ExperimentConfig(name="core-remote", env={"type": ENV, "remote": {}}, **base),
            "stdio env + agent": ExperimentConfig(name="core-remote", env={"type": ENV, "remote": {}},
                                                  **{**base, "agent": {"type": "remote", "params": {
                                                      "ref": AGENT["type"], "params": AGENT["params"]}}}),
            "ws env": ExperimentConfig(name="core-remote", env={"type": "remote.ws", "params": {"url": env_srv.url}}, **base),
            "ws env + agent": ExperimentConfig(name="core-remote", env={"type": "remote.ws", "params": {"url": env_srv.url}},
                                               **{**base, "agent": {"type": "remote.ws", "params": {"url": agent_srv.url}}}),
        }
        print(f"== 1. one experiment, five placements ({n_val} seeds x 2 tasks, 1 worker)")
        runs, rows, ref_acts, ref_key = {}, [], None, None
        for name, cfg in modes.items():
            t0 = time.perf_counter()
            res = run_experiment(cfg, run_root=out / "runs", label=name.replace(" ", "").replace("+", "-"))
            wall = time.perf_counter() - t0
            acts, inside = actions_of(res.run_dir, res.log)
            key = [(s["key"], s["status"], s["steps"]) for s in res.log["samples"]]
            if ref_acts is None:
                ref_acts, ref_key = acts, key
            dev = max(float(np.max(np.abs(acts[k] - ref_acts[k]))) if acts[k].shape == ref_acts[k].shape else np.inf
                      for k in acts)
            steps = sum(s["steps"] for s in res.log["samples"])
            runs[name] = {"config_digest": cfg.digest(), "success_mean": res.metrics["success_mean"], "steps": steps,
                          "wall_s": wall, "startup_s": wall - inside, "ms_per_step": inside / steps * 1e3,
                          "same_status_and_steps": key == ref_key,
                          "max_action_dev_vs_in_process": dev, "run_dir": str(res.run_dir)}
        base_ms = runs["in-process"]["ms_per_step"]
        for name, r in runs.items():
            r["overhead_ms_per_step"] = r["ms_per_step"] - base_ms
            rows.append([name, r["config_digest"], f"{r['success_mean']:.2f}", r["steps"], r["same_status_and_steps"],
                         f"{r['max_action_dev_vs_in_process']:g}", f"{r['ms_per_step']:.3f}",
                         f"{r['overhead_ms_per_step']:+.3f}", f"{r['startup_s']:.2f}"])
        table(rows, ["placement", "config digest", "success", "steps", "same outcomes", "max |da|", "ms/step",
                     "overhead", "set-up (s)"])
        print("  ms/step: time inside episodes (trace clock) per step, with the agent call and trace writing; "
              "set-up: the rest of the run")

        print(f"\n== 2. latency of one env.step call ({n_calls} calls each)")
        task = registry.build("benchmarks", {"type": "mujoco/reach-v0"}).instances("validation")[0]
        lat, lrows = {}, []
        for name, make in (("direct", lambda: registry.build("envs", {"type": ENV})),
                           ("stdio proxy", lambda: RemoteEnvironment(registry.reference("envs", ENV))),
                           ("ws proxy", lambda: WSEnvironment(env_srv.url))):
            env = make()
            try:
                step_latency(env, task, 50)                              # warm up
                xs = np.array(step_latency(env, task, n_calls)) * 1e6
            finally:
                env.close()
            lat[name] = {"median_us": float(np.median(xs)), "p95_us": float(np.percentile(xs, 95)),
                         "mean_us": float(np.mean(xs))}
        for name, v in lat.items():
            lrows.append([name, f"{v['median_us']:.0f}", f"{v['p95_us']:.0f}",
                          f"{v['median_us'] - lat['direct']['median_us']:+.0f}"])
        table(lrows, ["call path", "median (us)", "p95 (us)", "added vs direct (us)"])
        env = registry.build("envs", {"type": ENV})
        env.reset(task)
        req = json.dumps({"id": 1, "method": "step", "args": codec.encode({"action": Action(data=np.zeros(2))})})
        rep = json.dumps({"id": 1, "ok": True, "result": env.step(Action(data=np.zeros(2))).model_dump(mode="json")})
        env.close()
        wire_bytes = {"request": len(req), "reply": len(rep)}
        print(f"  one step on the wire: request {len(req)} B, reply {len(rep)} B (JSON, arrays base64)")

        print("\n== 3. wire codecs on an observation with a 224x224x3 image")
        cc = codec_compare(20 if a.quick else 200)
        table([[k, f"{v['bytes'] / 1024:.1f}", f"{v['roundtrip_ms']:.3f}"] for k, v in cc.items()],
              ["codec", "KiB", "encode+decode (ms)"])
    finally:
        env_srv.stop()
        agent_srv.stop()

    ok = all(r["same_status_and_steps"] and r["max_action_dev_vs_in_process"] == 0.0 for r in runs.values())
    result = {"example": SLUG, "quick": a.quick, "episodes": 2 * n_val, "placements": runs, "step_latency_us": lat,
              "step_wire_bytes": wire_bytes, "codecs_224px_obs": cc, "identical_outcomes": ok}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nidentical outcomes in every placement: {ok}\nresult: {out / 'result.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
