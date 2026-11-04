"""Summarize attempted episodes and exact paired outcomes without hiding invalid runs."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def wilson(successes, n):
    p, z = successes / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def summarize(rows, *, baseline="memory_off", candidate="memory_on"):
    if baseline == candidate:
        raise ValueError("baseline and candidate must differ")
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        groups[row["task"], row["condition"]].append(row)
        key = row["task"], row["split"], row["episode"]
        if row["condition"] in pairs[key]:
            raise ValueError(f"duplicate condition for {key}")
        pairs[key][row["condition"]] = row
    table = []
    for (task, condition), group in sorted(groups.items()):
        n = len(group)
        successes = sum(r["status"] == "success" for r in group)
        table.append({"task": task, "condition": condition, "attempted": n,
                      "successes": successes, "success_per_attempt": successes / n,
                      "wilson95": wilson(successes, n),
                      "outcomes": dict(Counter(r["status"] for r in group)),
                      "median_seconds": statistics.median(r["seconds"] for r in group)})
    wins = losses = ties = incomplete = invalid = 0
    valid_status = {"success", "fail", "timeout", "agent_error", "model_error", "invalid_output", "budget_exhausted"}
    for key, pair in pairs.items():
        if not {candidate, baseline} <= set(pair):
            incomplete += 1
            continue
        on, off = pair[candidate], pair[baseline]
        if on["status"] not in valid_status or off["status"] not in valid_status:
            invalid += 1
            continue
        if on.get("comparison_fingerprint") != off.get("comparison_fingerprint"):
            raise ValueError(f"different runtime configurations for {key}")
        if on["seed"] != off["seed"]:
            raise ValueError(f"unpaired seeds for {key}")
        if not on.get("reset_observation_sha256") or on.get("reset_observation_sha256") != off.get("reset_observation_sha256"):
            raise ValueError(f"different reset observations for {key}")
        a, b = on["status"] == "success", off["status"] == "success"
        wins += a and not b
        losses += b and not a
        ties += a == b
    discordant = wins + losses
    one_sided = sum(math.comb(discordant, i) for i in range(losses + 1)) / 2 ** discordant if discordant else 1.0
    two_sided = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1.0
    overall = {}
    for condition in sorted({r["condition"] for r in rows}):
        selected = [r for r in rows if r["condition"] == condition]
        successes = sum(r["status"] == "success" for r in selected)
        overall[condition] = {"attempted": len(selected), "successes": successes,
                              "success_per_attempt": successes / len(selected),
                              "wilson95": wilson(successes, len(selected)),
                              "task_mean_success": statistics.mean(t["success_per_attempt"] for t in table if t["condition"] == condition),
                              "total_seconds": sum(r["seconds"] for r in selected),
                              "total_actions": sum(r["steps"] for r in selected),
                              "model_calls": sum(r.get("model_calls", 0) for r in selected),
                              "input_tokens": sum(r.get("input_tokens", 0) for r in selected),
                              "output_tokens": sum(r.get("output_tokens", 0) for r in selected),
                              "outcomes": dict(Counter(r["status"] for r in selected))}
    return {"overall": overall, "per_task": table, "paired": {"memory_wins": wins, "memory_losses": losses,
            "ties": ties, "incomplete_pairs": incomplete, "invalid_pairs": invalid,
            "exact_mcnemar_one_sided": one_sided, "exact_mcnemar_two_sided": two_sided},
            "comparison": {"baseline": baseline, "candidate": candidate},
            "interpretation": "Exploratory comparison, not a promotion gate. Agent/model/output/budget failures count as failures, including in paired statistics. Infrastructure/unknown statuses and incomplete pairs are excluded from pairing and reported; all attempts remain in per-attempt rates."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", default="memory_off")
    parser.add_argument("--candidate", default="memory_on")
    args = parser.parse_args()
    summary = summarize(json.loads((args.run / "results.json").read_text()), baseline=args.baseline, candidate=args.candidate)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
