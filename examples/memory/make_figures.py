"""Draw the README figure from the examples' result.json files (run the examples in full mode first).

    uv run python examples/memory/make_figures.py [--root runs/examples/memory] [--out examples/memory/assets]

Needs matplotlib (not a dependency of the examples themselves).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import DEFAULT_ROOT

HERE = Path(__file__).resolve().parent
WITH, WITHOUT = "#2a78d6", "#9a9893"          # memory on / off (categorical slot 1, a recessive neutral)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(HERE / "assets"))
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda slug: json.loads((root / slug / "result.json").read_text())  # noqa: E731
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=110)

    def bars(ax, labels, off, on, names, title):
        x = range(len(labels))
        for dx, vals, colour, name in ((-0.2, off, WITHOUT, names[0]), (0.2, on, WITH, names[1])):
            ys = [v["success"] for v in vals]
            err = [[y - v["ci95"][0] for y, v in zip(ys, vals)], [v["ci95"][1] - y for y, v in zip(ys, vals)]]
            ax.bar([i + dx for i in x], ys, width=0.38, color=colour, label=name, yerr=err,
                   error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("success (95% Wilson CI)")
        ax.set_title(title, color=INK, loc="left", fontsize=9.5)
        ax.legend(frameon=False, loc="upper left", fontsize=8)

    r1 = load("spatial_memory_occlusion")["results"]
    keys = [k for k in ("visible_1", "visible_5", "visible_20", "visible_all") if k in r1]
    bars(axes[0, 0], [k.split("_")[1].replace("all", "always") for k in keys], [r1[k]["memory_off"] for k in keys],
         [r1[k]["memory_on"] for k in keys], ("memory off", "spatial memory"),
         "01 table reach: steps the pads stay visible")
    r3 = load("action_memory_scene_recall")["arms"]
    bars(axes[0, 1], ["revisited scenes", "new scenes"], [r3[s]["control"] for s in ("revisited", "new")],
         [r3[s]["memory"] for s in ("revisited", "new")], ("no memory", "action memory"),
         "03 planar reach: action memory (n.s. on both)")
    r4 = load("skill_library")["fresh_scenes"]
    bars(axes[1, 0], ["fresh scenes"], [r4["parent"]], [r4["parent_plus_skills"]], ("parent", "parent + verified skill"),
         "04 skills agent applies the gate-verified skill")
    r5 = load("memory_across_sessions")["sessions"]
    ax = axes[1, 1]
    s = [r["session"] for r in r5]
    ax.plot(s, [r["control"]["success"] for r in r5], color=WITHOUT, lw=2, marker="o", ms=5, label="no memory")
    ax.plot(s, [r["memory"]["success"] for r in r5], color=WITH, lw=2, marker="o", ms=5, label="bank from earlier sessions")
    ax.set_xticks(s)
    ax.set_xlabel("session (same workcell scenes)")
    ax.set_ylabel("success")
    ax.set_ylim(0.6, 0.9)
    ax.set_title("05 memory carried across sessions", color=INK, loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out / "memory_results.png"
    fig.savefig(path, facecolor="white")
    print(f"{path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
