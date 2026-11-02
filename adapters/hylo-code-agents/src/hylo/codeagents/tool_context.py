"""Context for a tool-calling model (RFC-0007 part 3): the conversation is kept whole; what is sent is a projection.

The canonical history is append-only. Before every model call `ToolContext.project()` builds the request:

```text
  head       the system message and the task message, always verbatim
  tail       the last `tail_decisions` actions (an action = an act call and everything after it up to the next
             act) verbatim, images included for the last `image_decisions`
  older      tool results longer than `inline_chars` become references: {"stored": "ref:<id>", "preview": ...};
             their images become a note naming the reference; identical results are stored once
  pressure   when the estimated request exceeds `compact_tokens`, everything between the head and the tail is
             replaced by a summary the model wrote in a separate call (no tools), plus the list of references
             it covered; the exact turns stay readable through `read_artifact`
```

Nothing is lost: every stored item is in `self.artifacts` (and in the trace), and the model can open it again
with `read_artifact` (a character range of a text, or an image). Tool calls and their results always travel
together, so a projection is always a valid conversation for the provider.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from hylo.core.tooling import Help, ToolDef, ToolError, ToolOutput, obj

SUMMARY_KEYS = ("progress", "confirmed", "ruled_out", "attempts", "constraints", "avoid", "next_checks")
SUMMARY_PROMPT = (
    "You are condensing your own earlier work on a robot task so you can continue with less context. Write ONE JSON "
    "object with exactly these keys, each a list of short strings:\n"
    "  progress     what has been achieved so far, in task terms\n"
    "  confirmed    facts established by measurement or by an action's receipt\n"
    "  ruled_out    beliefs that turned out wrong\n"
    "  attempts     each failed attempt as 'what failed / evidence / what to change'\n"
    "  constraints  limits that still apply (reach, collisions, objects that must not move)\n"
    "  avoid        calls or plans not to repeat\n"
    "  next_checks  what to measure again before acting (positions go stale after motion)\n"
    "Do not copy coordinates or other numbers: they are stale; re-measure instead. At most 250 words in total. "
    "Reply with the JSON object only.")


@dataclass
class Entry:
    message: dict[str, Any]
    decision: int                      # the action count when the message was added
    ref: str | None = None             # set when the content was stored
    tags: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(messages: list[Mapping[str, Any]], image_tokens: int = 800) -> int:
    n = 0
    for m in messages:
        n += len(str(m.get("content") or "")) // 3 + 4
        n += sum(len(json.dumps(c.get("arguments", ""))) // 3 for c in m.get("tool_calls") or [])
        n += image_tokens * len(m.get("images") or [])
    return n


class ToolContext:
    def __init__(self, system: str, task: str, task_images: list[Any] | None = None, *, tail_decisions: int = 3,
                 image_decisions: int = 1, inline_chars: int = 1500, preview_chars: int = 300,
                 compact_tokens: int = 48_000, image_tokens: int = 800,
                 summarize: Callable[[list[dict[str, Any]]], str] | None = None) -> None:
        self.head = [{"role": "system", "content": system},
                     {"role": "user", "content": task, **({"images": list(task_images)} if task_images else {})}]
        self.entries: list[Entry] = []
        self.decision = 0
        self.tail_decisions, self.image_decisions = tail_decisions, image_decisions
        self.inline_chars, self.preview_chars = inline_chars, preview_chars
        self.compact_tokens, self.image_tokens = compact_tokens, image_tokens
        self.summarize = summarize
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] | None = None      # {"upto": entry index, "text": ..., "refs": [...]}
        self.stats = {"stored": 0, "summaries": 0, "reopened": 0}

    # ------------------------------------------------------------------ building
    def add(self, message: Mapping[str, Any], **tags: Any) -> None:
        self.entries.append(Entry(dict(message), self.decision, tags=dict(tags)))

    def acted(self) -> None:
        """An act call ran: the next messages belong to the next decision."""
        self.decision += 1

    def store(self, kind: str, content: Any, note: str = "") -> str:
        if kind == "image":
            arr = np.ascontiguousarray(np.asarray(content, np.uint8))
            digest = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:12]
        else:
            digest = hashlib.sha256(str(content).encode()).hexdigest()[:12]
        ref = f"ref:{digest}"
        if ref not in self.artifacts:
            self.artifacts[ref] = {"kind": kind, "content": content, "note": note}
            self.stats["stored"] += 1
        return ref

    # ------------------------------------------------------------------ projection
    def _aged(self, i: int, e: Entry) -> dict[str, Any]:
        m = dict(e.message)
        age = self.decision - e.decision
        if m["role"] == "tool" and len(str(m.get("content") or "")) > self.inline_chars and age >= self.tail_decisions:
            ref = e.ref or self.store("text", m["content"], f"result of {m.get('name')}")
            e.ref = ref
            text = str(m["content"])
            m["content"] = json.dumps({"stored": ref, "chars": len(text), "preview": text[: self.preview_chars],
                                       "note": "older result stored; read_artifact(ref) shows it in full"})
        if m.get("images") and age >= self.image_decisions:
            refs = [self.store("image", im, f"image from {m.get('name') or m['role']}") for im in m["images"]]
            m.pop("images")
            try:
                body = json.loads(m.get("content") or "")
            except (TypeError, ValueError):
                body = None
            if isinstance(body, dict):                      # keep a JSON result JSON
                body["images_stored"] = refs
                m["content"] = json.dumps(body, separators=(",", ":"))
            else:
                m["content"] = f"{m.get('content') or ''}\n[images stored: {', '.join(refs)}; read_artifact shows one]"
        return m

    def project(self) -> list[dict[str, Any]]:
        start = 0
        head = [dict(m) for m in self.head]
        if len(head) > 1 and self.decision > self.image_decisions and head[1].get("images"):   # the opening view
            refs = [self.store("image", im, "opening view") for im in head[1]["images"]]
            head[1] = {**{k: v for k, v in head[1].items() if k != "images"},
                       "content": f"{head[1]['content']}\n[opening images stored: {', '.join(refs)}]"}
        if self.summary is not None:
            start = self.summary["upto"]
            head.append({"role": "user", "content": "SUMMARY OF YOUR EARLIER WORK (exact turns stored: "
                                                    f"{', '.join(self.summary['refs']) or 'none'})\n{self.summary['text']}"})
            head.append({"role": "assistant", "content": "Understood; I will re-measure before acting."})
        body = [self._aged(i, e) for i, e in enumerate(self.entries) if i >= start]
        msgs = head + body
        if (self.summarize is not None and estimate_tokens(msgs, self.image_tokens) > self.compact_tokens):
            if self._compact():
                return self.project()
        return msgs

    def _compact(self) -> bool:
        """Summarize the entries between the head (or the last summary) and the tail; False if nothing to do."""
        start = self.summary["upto"] if self.summary else 0
        cut = len(self.entries)
        for i in range(len(self.entries) - 1, start - 1, -1):     # first entry of the tail
            if self.decision - self.entries[i].decision < self.tail_decisions:
                cut = i
            else:
                break
        while start < cut < len(self.entries) and self.entries[cut].message["role"] == "tool":   # never split calls from results
            cut -= 1
        if cut - start < 2:
            return False
        old = [self._aged(i, e) for i, e in enumerate(self.entries[start:cut], start)]
        transcript = json.dumps([{k: v for k, v in m.items() if k != "images"} for m in old], default=str)
        ref = self.store("text", transcript, f"exact turns {start}..{cut - 1}")
        prior = f"Earlier summary:\n{self.summary['text']}\n\n" if self.summary else ""
        msgs = [{"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"{prior}Task:\n{self.head[1]['content'] if len(self.head) > 1 else ''}"
                                            f"\n\nTurns to condense:\n"
                                            f"{transcript[-60000:]}"}]
        try:
            text = self.summarize(msgs)
        except Exception as exc:  # noqa: BLE001 - without a summary, the turns are only shortened
            text = json.dumps({"progress": [f"(summary failed: {exc!r})"[:200]]})
        refs = (self.summary["refs"] if self.summary else []) + [ref]
        self.summary = {"upto": cut, "text": _clean_summary(text), "refs": refs}
        self.stats["summaries"] += 1
        return True

    # ------------------------------------------------------------------ recovery tool
    def read_artifact(self, args: dict[str, Any]) -> Any:
        ref = args["ref"]
        art = self.artifacts.get(ref)
        if art is None:
            raise ToolError("unknown_ref", f"no stored item {ref!r}", field="ref", retry="never")
        self.stats["reopened"] += 1
        if art["kind"] == "image":
            return ToolOutput({"ref": ref, "note": art["note"]}, images=[art["content"]], image_notes=[art["note"]])
        text = str(art["content"])
        off = int(args.get("offset") or 0)
        lim = int(args.get("limit") or 6000)
        return {"ref": ref, "note": art["note"], "chars": len(text), "offset": off, "text": text[off: off + lim],
                "more": off + lim < len(text)}

    def tool(self) -> ToolDef:
        return ToolDef("read_artifact", "read", Help(
            "Open a stored earlier result or image by its reference (ref:...).",
            use_when="an older result was stored to save context and you need what it said exactly. For what is "
                     "true now, call the tool again instead: stored results describe the past.",
            returns="text: a character range (offset, limit, more); image: the image."),
            obj({"ref": {"type": "string", "description": "the reference, e.g. ref:1a2b3c4d5e6f"},
                 "offset": {"type": "integer", "minimum": 0, "description": "first character (default 0)"},
                 "limit": {"type": "integer", "minimum": 100, "maximum": 8000, "description": "characters (default "
                                                                                            "6000)"}}, ["ref"]),
            self.read_artifact)


def _clean_summary(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    try:
        start, end = t.index("{"), t.rindex("}") + 1
        obj_ = json.loads(t[start:end])
        if isinstance(obj_, dict):
            return json.dumps({k: obj_.get(k, []) for k in SUMMARY_KEYS}, indent=1)
    except ValueError:
        pass
    return t[:3000]


__all__ = ["Entry", "SUMMARY_KEYS", "ToolContext", "estimate_tokens"]
