"""Every dialog block carries its message's time — and its own zone.

Owner requirement (2026-08-29): in the web UI history, messages and tool
calls must show a readable timestamp; by default the host's local time,
alternatively UTC, and "it must be clear and visible from the timestamp
itself, so that there are no misunderstandings later".

Hence the hard rule: the zone is ALWAYS printed (``+02:00`` or ``UTC``).
Bare wall-clock time next to a log in another zone is exactly the way to
misread an incident timeline that the requirement forbids.

The time source is each message's ``created_at`` (ISO-8601 UTC, set once
and never rewritten, see ``deepagents.middleware.message_timestamps``).
"""
from __future__ import annotations

import re

from app.services.dialog_formatter import (
    format_timestamp,
    render_dialog,
    split_into_turns,
)

# Real shape from a live agent (2026-08-29).
_HUMAN_TS = "2026-08-29T12:29:20.518409+00:00"
_AI_TS = "2026-08-29T12:29:24.341295+00:00"
_TOOL_TS = "2026-08-29T12:29:22.442548+00:00"


def _history() -> list[dict]:
    return [
        {"type": "HumanMessage", "id": "h1", "content": "hello", "created_at": _HUMAN_TS},
        {
            "type": "AIMessage",
            "id": "a1",
            "content": "working on it",
            "created_at": _TOOL_TS,
            "tool_calls": [
                {"id": "t1", "name": "read_file", "args": {"p": "/x"}, "created_at": _TOOL_TS}
            ],
        },
        {"type": "AIMessage", "id": "a2", "content": "done", "created_at": _AI_TS},
    ]


def test_utc_mode_prints_utc_and_the_right_clock() -> None:
    out = render_dialog(split_into_turns(_history()), "final", "utc")

    assert "--- Assistant 2026-08-29 12:29:24 UTC ---" in out
    assert "--- User 2026-08-29 12:29:20 UTC ---" in out


def test_local_mode_always_spells_out_the_offset() -> None:
    """Local time without a zone is exactly the "misunderstanding"
    that the requirement says to avoid."""
    out = render_dialog(split_into_turns(_history()), "final", "local")

    headers = [ln for ln in out.splitlines() if ln.startswith("--- ")]
    assert headers, out
    for line in headers:
        assert re.search(r"[+-]\d{2}:\d{2} ---$", line), f"no offset: {line}"


def test_tool_call_blocks_carry_their_own_timestamp() -> None:
    """A tool call has its own ``created_at`` — and that is what must be printed."""
    out = render_dialog(split_into_turns(_history()), "full", "utc")

    assert "--- Assistant (tool: read_file) 2026-08-29 12:29:22 UTC ---" in out


def test_every_message_in_full_mode_is_stamped() -> None:
    out = render_dialog(split_into_turns(_history()), "full", "utc")

    headers = [ln for ln in out.splitlines() if ln.startswith("--- ")]
    assert len(headers) == 4, headers  # User + 2 Assistant + 1 tool
    for line in headers:
        assert " UTC ---" in line, f"no stamp: {line}"


def test_message_without_timestamp_renders_plain_header() -> None:
    """Old checkpoints were written before stamps existed.

    A missing time must yield a header WITHOUT a time, not an invented
    or zero date: showing a wrong time is worse than showing none.
    """
    history = [
        {"type": "HumanMessage", "id": "h1", "content": "hello"},
        {"type": "AIMessage", "id": "a1", "content": "reply"},
    ]

    out = render_dialog(split_into_turns(history), "final", "utc")

    assert "--- User ---" in out
    assert "--- Assistant ---" in out
    assert "1970" not in out


def test_partial_timestamps_do_not_shift_onto_other_messages() -> None:
    """One message's stamp must never label a neighbouring message.

    The time lists run parallel to the content; a desync here would mean a
    time attributed to the wrong message — a silent lie in the timeline.
    """
    history = [
        {"type": "HumanMessage", "id": "h1", "content": "question"},
        {"type": "AIMessage", "id": "a1", "content": "first", "created_at": _TOOL_TS},
        {"type": "AIMessage", "id": "a2", "content": "second"},
    ]

    out = render_dialog(split_into_turns(history), "text", "utc")
    lines = out.splitlines()

    first = lines.index("first")
    second = lines.index("second")
    assert lines[first - 1] == "--- Assistant 2026-08-29 12:29:22 UTC ---"
    assert lines[second - 1] == "--- Assistant ---"


def test_bad_timestamp_is_dropped_not_echoed() -> None:
    assert format_timestamp("not a date") == ""
    assert format_timestamp(None) == ""
    assert format_timestamp("") == ""
    assert format_timestamp(12345) == ""


def test_naive_timestamp_is_treated_as_utc() -> None:
    """Stamps are documented as UTC; a naive value comes from an old writer."""
    assert format_timestamp("2026-08-29T12:29:20", "utc") == "2026-08-29 12:29:20 UTC"


def test_unknown_timestamps_mode_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="timestamps"):
        render_dialog(split_into_turns(_history()), "final", "moscow")


def test_default_mode_is_local() -> None:
    """The default is the host's local time, as the owner requested."""
    explicit = render_dialog(split_into_turns(_history()), "final", "local")
    default = render_dialog(split_into_turns(_history()), "final")

    assert default == explicit
