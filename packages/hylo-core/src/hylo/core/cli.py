"""`hylo`: the command dispatcher. Each package registers its own subcommands.

A subcommand is an entry point in the group "hylo.commands": a name and a callable taking the remaining
arguments and returning an exit code. The first line of the callable's docstring is its help line.

    [project.entry-points."hylo.commands"]
    bench = "hylo.bench.cli:main"          # hylo bench ...  (hylo-bench)

So `hylo` only has the commands of what is installed: on a robot computer with hylo-core and hylo-serve, `hylo serve`; on
a workstation with the harness, `hylo run`, `hylo evolve`, `hylo learn`, ...; with hylo-bench also `hylo bench`.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint, entry_points
from typing import Callable

GROUP = "hylo.commands"


def commands() -> dict[str, EntryPoint]:
    found: dict[str, list[EntryPoint]] = {}
    for ep in entry_points(group=GROUP):
        found.setdefault(ep.name, []).append(ep)
    clashes = {n: sorted({e.value for e in eps}) for n, eps in found.items() if len({e.value for e in eps}) > 1}
    if clashes:
        raise RuntimeError(f"hylo subcommands registered by more than one package: {clashes}")
    return {n: eps[0] for n, eps in sorted(found.items())}


def _summary(fn: Callable) -> str:
    return ((fn.__doc__ or "").strip().splitlines() or [""])[0]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmds = commands()
    if not args or args[0] in ("-h", "--help", "help"):
        print("usage: hylo COMMAND [ARGS ...]\n\ncommands (from the installed Hylo packages):")
        for name, ep in cmds.items():
            try:
                line = _summary(ep.load())
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not hide the others
                line = f"(failed to load: {exc})"
            print(f"  {name:<14} {line}")
        return 0
    name, rest = args[0], args[1:]
    if name not in cmds:
        print(f"hylo: unknown command {name!r}; installed: {', '.join(cmds) or '-'}", file=sys.stderr)
        return 2
    return int(cmds[name].load()(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
