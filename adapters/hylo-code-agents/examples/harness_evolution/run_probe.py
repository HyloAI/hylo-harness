"""Run one real-model CPU toy campaign and retain source/protocol/outcome evidence.

This probes orchestration and program synthesis. It is not a robot benchmark reproduction.
The model must already be available through the configured local Ollama service.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from hylo.codeagents import harness_evolution
from hylo.evolve.campaign import EvolveConfig, run_campaign, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-oss:20b")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = EvolveConfig.load(Path(__file__).with_name("campaign.yaml"))
    params = {**cfg.proposer.params, "model": {"backend": "ollama", "model": args.model,
              "reasoning_effort": "low", "timeout_s": 90, "retries": 0},
              "budget": {"turns": 16, "submits": 4, "resets": 4, "tokens": 180000},
              "out_root": str(root / "proposals"), "repair_campaigns": [str(root)]}
    cfg = cfg.model_copy(update={"out_dir": str(root), "proposer": cfg.proposer.model_copy(update={"params": params})})
    (root / "protocol.json").write_text(json.dumps({"interpretation": "Real LLM on CPU toy manipulation; not a robot benchmark result.",
                                                  "config": cfg.model_dump(mode="json")}, indent=2) + "\n")
    source = Path(harness_evolution.__file__).parent
    records = []
    for name in ("harness_evolution", "phase_memory", "research", "generator", "skills", "sandbox", "agents", "bridge", "fakes", "prompt"):
        path = source / f"{name}.py"
        target = root / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "source-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    start = time.time()
    try:
        state = run_campaign(cfg, log=lambda message: print(message, flush=True))
        result = {"elapsed_s": time.time() - start, "status": status(cfg)}
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"completed": True, "elapsed_s": result["elapsed_s"], "state": state.model_dump(mode="json")}), flush=True)
    except Exception as exc:
        (root / "error.json").write_text(json.dumps({"error": repr(exc), "elapsed_s": time.time() - start}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
