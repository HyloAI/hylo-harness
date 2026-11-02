"""`hylo chats`: tool-agent conversations from run traces as training records (chat format with tool calls).

    hylo chats RUN_DIR [RUN_DIR ...] --out chats.jsonl [--all] [--include-assisted] [--max-result 4000]

One JSON line per episode (or tool session): {"messages": [...], "tools": [...], "meta": {...}} in the common
chat-completions shape (system, user, assistant with tool_calls, tool). By default only episodes the task's check
judged successful are written, and never episodes a person steered (human_assisted) unless --include-assisted:
learning from steered runs would credit the model with the person's help. Images are not included; the
observation turns keep their text with a note of how many images they had. Results longer than --max-result
characters are cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence


def episodes(run: Path) -> Iterator[Path]:
    yield from sorted((run / "episodes").glob("*.jsonl"))


def conversation(path: Path, max_result: int = 4000) -> dict[str, Any] | None:
    start: dict[str, Any] = {}
    msgs: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"episode": path.stem}
    pending: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"agent.tool.' not in line[:40] and '"episode.' not in line[:30]:
                continue
            ev = json.loads(line)
            t, p = ev["type"], ev.get("payload") or {}
            if t == "agent.tool.start":
                start = p
                msgs.append({"role": "system", "content": p.get("system", "")})
                if p.get("task"):
                    msgs.append({"role": "user", "content": p["task"]})
            elif t == "agent.tool.operator" and p.get("kind") == "request":
                msgs.append({"role": "user", "content": p.get("message", "")})
            elif t == "agent.tool.operator":
                meta["human_assisted"] = True
                msgs.append({"role": "user", "content": f"OPERATOR MESSAGE: {p.get('message', '')}"})
            elif t == "agent.tool.turn" and not p.get("model_error"):
                msgs.extend(pending)
                pending = []
                calls = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str)
                    else json.dumps(c["arguments"])}} for c in p.get("tool_calls") or []]
                m: dict[str, Any] = {"role": "assistant", "content": p.get("text") or ""}
                if calls:
                    m["tool_calls"] = calls
                msgs.append(m)
            elif t == "agent.tool.result":
                text = str(p.get("result", ""))
                if len(text) > max_result:
                    text = text[:max_result] + "...(cut)"
                n = len(p.get("images") or [])
                pending.append({"role": "tool", "tool_call_id": p.get("id"), "name": p.get("name"),
                                "content": text + (f"\n[{n} image(s) not included]" if n else "")})
            elif t == "agent.tool.summary":
                meta.update(human_assisted=bool(p.get("human_assisted") or meta.get("human_assisted")),
                            usage=p.get("usage"), stopped=p.get("stopped"))
            elif t == "episode.start":
                meta.update(task=p.get("task_id"), seed=p.get("seed"), split=p.get("split"))
            elif t == "episode.end":
                meta.update(success=p.get("success"), status=p.get("status"))
    msgs.extend(pending)
    if not start:
        return None
    meta.update(model=start.get("model"), surface=start.get("surface"), prompt=start.get("prompt"))
    return {"messages": msgs, "tools": start.get("tools") or [], "meta": meta}


def main(argv: Sequence[str] | None = None) -> int:
    """export tool-agent conversations from traces as training records"""
    ap = argparse.ArgumentParser(prog="hylo chats", description="tool-agent conversations as training records")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="also unsuccessful episodes (and sessions without a verdict)")
    ap.add_argument("--include-assisted", action="store_true", help="also episodes a person steered")
    ap.add_argument("--max-result", type=int, default=4000)
    a = ap.parse_args(argv)
    kept = skipped = 0
    with open(a.out, "w") as out:
        for run in a.runs:
            for ep in episodes(Path(run)):
                c = conversation(ep, a.max_result)
                if c is None:
                    continue
                m = c["meta"]
                if (not a.all and m.get("success") is not True) or (m.get("human_assisted") and not a.include_assisted):
                    skipped += 1
                    continue
                out.write(json.dumps(c) + "\n")
                kept += 1
    print(f"wrote {kept} conversations to {a.out} ({skipped} left out)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["conversation", "main"]
