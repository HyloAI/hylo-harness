"""Collect the dexterous experiments' outputs into examples/dexterous/results.json (one command after the runs; any
experiment whose output is missing stays "not run yet").

    uv run python harness/adapters/hylo-code-agents/examples/dexterous/collect.py [--runs runs]

Reads RUNS/dex-grasps/results.json (grasp_benchmark.py), RUNS/dex-retarget/results.json (retarget_eval.py) and the
replay reports RUNS/dex-replay-<name>/report.json (`hylo replay ... --hand`), next to the gripper baselines
RUNS/dex-replay-<name>-gripper/report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_RUN = "not run yet"


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.is_file() else None


def replay_summary(rep: dict | None) -> dict | str:
    if rep is None:
        return NOT_RUN
    st = rep.get("steps", {})
    c = st.get("checks", {})
    m = c.get("events", {}).get("match", {})
    return {"events_matched": f"{m.get('matched')}/{m.get('of')}" if m else None,
            "object_error_final_m": {n: o.get("err_m") for n, o in c.get("final_state", {}).get("objects", {}).items()},
            "pose_error_sim_vs_tracked": c.get("pose_error_sim_vs_tracked"),
            "final_state_success": c.get("final_state", {}).get("success"),
            "success_reproduced": c.get("success_reproduced"),
            "retarget": st.get("hands", {}).get("retarget"), "grasps": st.get("grasps", {}).get("refined"),
            "video": st.get("video", {}).get("path")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    runs = Path(a.runs)
    res = json.loads((HERE / "results.json").read_text())
    g = _load(runs / "dex-grasps" / "results.json")
    if g:
        for key, r in g["results"].items():
            if key == "videos" or "episodes" not in r:
                continue
            hand, obj = key.split("/")
            n_ok = sum(e["success"] for e in r["episodes"])
            res["grasp_benchmark"]["results"].setdefault(hand, {})[obj] = f"{n_ok}/{r['n']}"
        res["grasp_benchmark"]["videos"] = g["results"].get("videos", NOT_RUN)
    rt = _load(runs / "dex-retarget" / "results.json")
    if rt:
        res["retargeting"]["results"] = {h: {k: v for k, v in s.items() if k != "per_frame_tip_error_mm"}
                                         for h, s in rt["hands"].items()}
        res["retargeting"]["video"] = rt.get("video", NOT_RUN)
    for name, entry in res["replays"].items():
        for variant, folder in (("after", f"dex-replay-{name}"), ("before", f"dex-replay-{name}-gripper")):
            rep = _load(runs / folder / "report.json")
            if rep is not None:
                entry[variant] = replay_summary(rep)
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
