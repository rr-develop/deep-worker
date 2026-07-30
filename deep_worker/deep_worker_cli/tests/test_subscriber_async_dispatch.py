"""Test that subscriber.on_message dispatch uses asyncreq(HANDLE_CALLATTR),
bypassing the sync_request guard that would fire on a sync-flagged netref
when called from the asyncio loop (issue #48).
"""

from __future__ import annotations

import pathlib

import pytest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "deep_worker_cli"
    / "process"
    / "agent_process.py"
)


def _source() -> str:
    return SOURCE.read_text()


def test_no_direct_on_message_calls_in_dispatch_methods():
    source = _source()
    lines = source.splitlines()

    in_dispatch_status = False
    in_dispatch_quarantine = False
    problematic_calls = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "def _dispatch_status_event" in line:
            in_dispatch_status = True
            in_dispatch_quarantine = False
        elif "def _dispatch_quarantine_warning" in line:
            in_dispatch_quarantine = True
            in_dispatch_status = False
        elif stripped.startswith("def ") and "def _on_error(" not in line:
            in_dispatch_status = False
            in_dispatch_quarantine = False

        # Skip comment lines
        if stripped.startswith("#"):
            continue

        if (in_dispatch_status or in_dispatch_quarantine) and (
            "subscriber.on_message(" in line
        ):
            problematic_calls.append((i, line.strip()))

    assert not problematic_calls, (
        f"Direct subscriber.on_message() calls at lines: {problematic_calls}"
    )


def test_asyncreq_callattr_imported_twice():
    source = _source()
    count = source.count("from rpyc.core.netref import asyncreq as _asyncreq")
    assert count >= 2


def test_asyncreq_callattr_used_twice():
    source = _source()
    count = source.count("_asyncreq(")
    assert count >= 2


def test_hardcoded_8_matches_handle_callattr():
    from rpyc_async.core.consts import HANDLE_CALLATTR
    assert HANDLE_CALLATTR == 8


def test_asyncreq_import_resolves():
    from rpyc_async.core.netref import asyncreq as _asyncreq
    assert callable(_asyncreq)
