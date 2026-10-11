"""Live context readings for ACP CLIs that do not report them on the wire.

claude-agent-acp sends ``usage_update`` after every model call, so it needs
nothing here. grok sends none at all — verified live on 1.0.5: a whole turn
yields only ``user_message_chunk``, ``agent_thought_chunk``,
``agent_message_chunk`` and ``available_commands_update`` — and its
``_meta`` arrives once, with the ``session/prompt`` reply. Between those two
points a grok turn can run a dozen model calls; on rg1_worker eight of them
carried the context from 260740 to 274277 with nothing recorded, past the
operator's ``-l 250000``.

grok does write the number as it goes, to its own telemetry log:
``<GROK_HOME>/logs/unified.jsonl``, one ``shell.turn.inference_done`` per
model call with ``ctx.prompt_tokens``. That is the provider's own count of
the prompt it read, cached prefix included (POLICIES §13 — a report, not an
estimate).

The log is HOST-WIDE: every grok process on the machine appends to it, so a
reading is only ours when ``sid`` matches the session id our ACP client got
from ``session/new`` (verified live: our 01a039b6-82f7-… appears verbatim as
``sid``). Without that filter one agent would publish another's context.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_INFERENCE_MSG = "shell.turn.inference_done"


def parse_grok_inferences(chunk: str, session_id: str) -> list[int]:
    """Context readings for ``session_id`` in ``chunk``, oldest first.

    Tolerant by construction: the tail of a file being appended can be half
    a line, and the log carries many event types (and other sessions).
    """
    if not session_id:
        return []
    out: list[int] = []
    for raw in chunk.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # partial tail or noise
        if not isinstance(entry, dict):
            continue
        if entry.get("msg") != _INFERENCE_MSG or entry.get("sid") != session_id:
            continue
        ctx = entry.get("ctx")
        if not isinstance(ctx, dict):
            continue
        used = ctx.get("prompt_tokens")
        if isinstance(used, int) and not isinstance(used, bool) and used > 0:
            out.append(used)
    return out


def grok_live_log_path() -> Path | None:
    """``<GROK_HOME>/logs/unified.jsonl``.

    ``GROK_HOME`` is honoured because an agent may run with it pointing away
    from ``$HOME`` — that is how the grok CLI keeps its credentials when the
    test harness sandboxes HOME.
    """
    raw = os.environ.get("GROK_HOME")
    home = Path(raw).expanduser() if raw else Path.home() / ".grok"
    return home / "logs" / "unified.jsonl"


#: ``AcpProfile.live_context_source`` values → (path, parser) builders.
def build_live_source(name: str, session_id: str):
    """Return ``(path, parse)`` for a declared live-context source."""
    if name == "grok_unified_log":
        path = grok_live_log_path()
        if path is None:
            return None
        return path, (lambda chunk: parse_grok_inferences(chunk, session_id))
    return None
