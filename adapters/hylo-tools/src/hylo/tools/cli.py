"""`hylo tool`: run a perception or motion tool adapter on the CPU in its own environment.

    hylo tool list                     the tools, their models and licences
    hylo tool NAME [ARGS ...]          run one tool (hylo tool NAME --help for its arguments)
"""

from __future__ import annotations

import importlib
import sys

from . import TOOLS


def _summary(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001
        return f"(failed to load: {exc})"
    doc = (mod.__doc__ or "").strip().splitlines()
    line = doc[0] if doc else ""
    extra = "  [" + ", ".join(x for x in (getattr(mod, "MODEL", ""), getattr(mod, "LICENCE", "")) if x) + "]"
    return line + (extra if extra != "  []" else "")


def tool(argv: list[str]) -> int:
    """perception and motion tool adapters (segment, depth, gravity, mesh, convex, hands, human, text_motion, render)"""
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print(__doc__.strip() + "\n\ntools:")
        for name, mod in TOOLS.items():
            print(f"  {name:<12} {_summary(mod)}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in TOOLS:
        print(f"hylo tool: unknown tool {name!r}; tools: {', '.join(TOOLS)}", file=sys.stderr)
        return 2
    return int(importlib.import_module(TOOLS[name]).cli(rest) or 0)


if __name__ == "__main__":
    sys.exit(tool(sys.argv[1:]))
