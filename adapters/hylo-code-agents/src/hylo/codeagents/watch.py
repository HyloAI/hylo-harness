"""`hylo watch`: follow a tool-agent episode in the terminal, replay a finished one, or steer a running one.

    hylo watch RUN_DIR [--episode ID]        print the newest episode (or ID) and keep following it while it runs
    hylo watch RUN_DIR --replay [--speed 4]  replay a finished episode on its recorded timing
    hylo watch RUN_DIR --raw                 the episode's agent events as JSON lines
    hylo watch --say "MESSAGE" --inbox PATH  send a message to a running tools.llm agent (its `inbox`)
    hylo watch RUN_DIR --inbox PATH --interactive   follow, and send every line typed to the agent (Ctrl-D stops typing)

One line per thing that happened: the model's words, each tool call with a condensed result, each action's receipt,
operator messages, and the verdict, with a footer of calls, decisions, tokens and cost. Read-only, except --say.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

COLOURS = {"dim": "2", "bold": "1", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
           "cyan": "36"}


class Printer:
    def __init__(self, out: TextIO = sys.stdout, colour: bool | None = None, width: int | None = None) -> None:
        self.out = out
        self.colour = out.isatty() if colour is None else colour
        self.width = width or max(60, min(160, os.get_terminal_size().columns if out.isatty() else 120))

    def c(self, text: str, *styles: str) -> str:
        if not self.colour or not styles:
            return text
        return "\x1b[" + ";".join(COLOURS[s] for s in styles) + "m" + text + "\x1b[0m"

    def line(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def clip(self, text: str, room: int | None = None) -> str:
        text = " ".join(str(text).split())
        room = room or self.width
        return text if len(text) <= room else text[: room - 1] + "…"


def _args(a: Any) -> str:
    if not isinstance(a, dict):
        return str(a)
    return ", ".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in a.items())


def condense(name: str, result: str) -> tuple[str, bool]:
    """A one-line summary of a tool result and whether it succeeded."""
    try:
        r = json.loads(result)
    except (TypeError, ValueError):
        return result[:200], True
    if not r.get("ok", True):
        e = r.get("error") or {}
        return f"{e.get('code')}: {e.get('message')}", False
    if "status" in r and "action_id" in r:
        parts = [f"{r['status']}", f"{r.get('steps', 0)} steps"]
        for s in r.get("stages") or []:
            if "tcp_error_mm" in s:
                parts.append(f"{s['stage']} err {s['tcp_error_mm']} mm")
        if r.get("holding") is not None:
            parts.append(f"holding={r['holding']}")
        if r.get("why"):
            parts.append(r["why"])
        return ", ".join(parts), r["status"] != "refused"
    if name == "locate":
        return "; ".join(f"{c['target']} {c['label']} at {c['center']} size {c['size']}"
                         for c in r.get("candidates", [])), True
    if name == "grasp_options":
        return "; ".join(f"{o['option']} {o['style']} at {o['grasp']['position']} reach={o['reachable']}"
                         for o in r.get("options", [])), True
    keep = {k: v for k, v in r.items() if k not in ("ok", "images")}
    return json.dumps(keep, separators=(",", ":")), True


def render(ev: dict[str, Any], pr: Printer, totals: dict[str, Any]) -> None:
    t, p = ev["type"], ev.get("payload") or {}
    if t == "episode.start":
        pr.line(pr.c(f"■ {p.get('task_id')}  seed {p.get('seed')}  split {p.get('split')}  agent {p.get('agent')}",
                     "bold"))
        comps = p.get("agent_components") or {}
        if comps.get("model"):
            pr.line(pr.c(f"  model {comps['model']}", "dim"))
    elif t == "agent.tool.start":
        pr.line(pr.c(f"  task: {p.get('instruction')}", "bold"))
        pr.line(pr.c(f"  {len(p.get('tools') or [])} tools (surface {p.get('surface')}), prompt "
                     f"{(p.get('prompt') or {}).get('bundle')}", "dim"))
    elif t == "agent.tool.turn":
        totals.update(calls=p.get("call"), usage=p.get("usage") or {}, decisions=p.get("decisions", 0))
        if p.get("model_error"):
            pr.line(pr.c(f"  ! model error: {pr.clip(p['model_error'], pr.width - 18)}", "red"))
            return
        if (p.get("text") or "").strip():
            pr.line(pr.c("  ◆ ", "magenta") + pr.clip(p["text"], pr.width - 4))
        lat = p.get("latency_s")
        pr.line(pr.c(f"    call {p.get('call')} · {p.get('tokens_in')} in / {p.get('tokens_out')} out"
                     + (f" · {lat:.1f} s" if isinstance(lat, (int, float)) else ""), "dim"))
    elif t == "agent.tool.result":
        summary, ok = condense(p.get("name", ""), p.get("result", ""))
        head = f"  {'▶' if p.get('kind') == 'act' else '·'} {p.get('name')}({_args(p.get('arguments'))})"
        pr.line(pr.c(pr.clip(head, pr.width), "cyan" if p.get("kind") == "act" else "blue"))
        pr.line("      " + pr.c(pr.clip(summary, pr.width - 6), "green" if ok else "red"))
    elif t == "agent.tool.operator":
        pr.line(pr.c(f"  ✉ operator: {p.get('message')}", "yellow", "bold"))
    elif t == "agent.tool.finish":
        pr.line(pr.c(f"  ■ finish({p.get('outcome')}): {p.get('reason')}", "bold"))
    elif t == "agent.tool.summary":
        u = p.get("usage") or {}
        parts = [f"{p.get('calls')} calls"]
        if p.get("decisions") is not None:
            parts.append(f"{p.get('decisions')} decisions ({p.get('refused')} refused)")
        parts.append(f"{u.get('tokens_in')} in / {u.get('tokens_out')} out tokens, cost {u.get('cost_usd')}")
        if p.get("stopped"):
            parts.append(f"stopped: {p.get('stopped')}")
        pr.line(pr.c("  summary: " + ", ".join(parts) + (", HUMAN-ASSISTED" if p.get("human_assisted") else ""), "dim"))
        if p.get("answer"):
            pr.line(pr.c(f"  answer: {pr.clip(p['answer'], pr.width - 10)}", "bold"))
    elif t == "episode.end":
        ok = p.get("success")
        pr.line(pr.c(f"■ verdict: {'SUCCESS' if ok else 'failure' if ok is False else 'unjudged'} "
                     f"({p.get('termination_reason') or p.get('reason')}) after {p.get('steps')} steps",
                     "green" if ok else "red", "bold"))


SHOWN = ("episode.start", "episode.end", "agent.tool.start", "agent.tool.turn", "agent.tool.result",
         "agent.tool.operator", "agent.tool.finish", "agent.tool.summary")


def episode_file(run: Path, episode: str | None) -> Path:
    eps = sorted((run / "episodes").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if episode:
        p = run / "episodes" / f"{episode}.jsonl"
        if not p.is_file():
            raise SystemExit(f"no episode {episode} in {run}")
        return p
    if not eps:
        raise SystemExit(f"no episodes in {run}/episodes yet")
    return eps[-1]


def events(path: Path, follow: bool, poll_s: float = 0.5, idle_s: float = 3600.0) -> Iterator[dict[str, Any]]:
    """Events of an episode file; with `follow`, keep reading new lines until episode.end (or idle_s of silence).
    The agent's last events (the final receipt, its summary) may land after episode.end: they are read too, and
    episode.end itself comes last."""
    pos, last = 0, time.monotonic()
    buf = b""
    end: dict[str, Any] | None = None
    while True:
        with open(path, "rb") as f:
            f.seek(pos)
            chunk = f.read()
        pos += len(chunk)
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            if not raw.strip():
                continue
            last = time.monotonic()
            ev = json.loads(raw)
            if ev["type"] == "episode.end":
                end = ev
            elif ev["type"] in SHOWN:
                yield ev
        if end is not None:
            if follow:                                    # give the agent thread a moment to write its last events
                time.sleep(min(2.0, 4 * poll_s))
                with open(path, "rb") as f:
                    f.seek(pos)
                    for raw in (buf + f.read()).split(b"\n"):
                        if raw.strip():
                            ev = json.loads(raw)
                            if ev["type"] in SHOWN and ev["type"] != "episode.end":
                                yield ev
            yield end
            return
        if not follow or time.monotonic() - last > idle_s:
            return
        time.sleep(poll_s)


def watch(run: Path, episode: str | None = None, *, follow: bool = True, replay: bool = False, speed: float = 4.0,
          raw: bool = False, out: TextIO = sys.stdout, colour: bool | None = None) -> dict[str, Any]:
    pr = Printer(out, colour)
    path = episode_file(run, episode)
    totals: dict[str, Any] = {}
    prev_t = None
    for ev in events(path, follow and not replay):
        if replay and prev_t is not None and ev.get("wall_time"):
            time.sleep(max(0.0, min(5.0, (ev["wall_time"] - prev_t) / max(speed, 1e-3))))
        prev_t = ev.get("wall_time") or prev_t
        if raw:
            pr.line(json.dumps({k: v for k, v in ev.items() if k not in ("h", "prev")}, default=str))
        else:
            render(ev, pr, totals)
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    """follow, replay or steer a tool-agent episode in the terminal"""
    ap = argparse.ArgumentParser(prog="hylo watch", description="follow, replay or steer a tool-agent episode")
    ap.add_argument("run", nargs="?", help="run directory")
    ap.add_argument("--episode")
    ap.add_argument("--replay", action="store_true", help="replay on recorded timing (no following)")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--once", action="store_true", help="print what is there and stop (no following)")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--say", help="send this message to a running agent's inbox")
    ap.add_argument("--inbox", help="the agent's inbox file (its `inbox` parameter)")
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--interactive", action="store_true", help="with --inbox: lines typed while following are sent")
    a = ap.parse_args(argv)
    if a.say:
        if not a.inbox:
            ap.error("--say needs --inbox")
        from .tool_agent import post

        post(a.inbox, a.say)
        print(f"sent to {a.inbox}")
        return 0
    if not a.run:
        ap.error("RUN_DIR is required")
    if a.interactive:
        if not a.inbox:
            ap.error("--interactive needs --inbox")
        import threading

        from .tool_agent import post

        def typing() -> None:
            for line in sys.stdin:
                if line.strip():
                    post(a.inbox, line.strip())
                    print(f"  (sent: {line.strip()})", flush=True)

        threading.Thread(target=typing, daemon=True).start()
    try:
        watch(Path(a.run), a.episode, follow=not a.once, replay=a.replay, speed=a.speed, raw=a.raw,
              colour=False if a.no_colour else None)
    except KeyboardInterrupt:
        return 130
    return 0


__all__ = ["condense", "main", "render", "watch"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
