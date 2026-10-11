"""Static guard: Full Dialog opens as an ISOLATED, script-free snapshot tab.

DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md §3.1 / §3.5 / §4 test 2.

Why a source-level guard: the owner's contract is "the Full Dialog window is
a click-time snapshot; it NEVER updates; it can never hang the main UI". The
Python suite never executes ``main.js`` in a browser, so a future
``setInterval``/WebSocket "live view" or a plain ``window.open(url, '_blank')``
(popup shares the opener's browsing-context group ⇒ same renderer) could land
silently. This test pins the contract at pytest-collection time, in the same
spirit as ``tests/integration/test_no_blocking_rpyc_in_handlers.py``.

Post-conditions of the design (RED before implementation):
* ``openFullDialog`` and ``openAgentStats`` call
  ``window.open(<url>, '_blank', 'noopener,noreferrer')``.
* ``openFullDialog`` contains no timer / socket / fetch primitives.
* ``static/dialog.html`` (dead JS-driven dialog page) is gone and nothing
  under ``static/`` mentions it.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
MAIN_JS = STATIC_DIR / "js" / "main.js"


def _function_body(src: str, name: str) -> str:
    """Return the text of ``function <name>(...) { ... }`` (brace-balanced)."""
    m = re.search(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", src)
    assert m, f"function {name} not found in main.js"
    i = m.end()
    depth = 1
    while depth and i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert depth == 0, f"unbalanced braces in {name}"
    return src[m.start():i]


_NOOPENER_OPEN = re.compile(
    r"window\.open\(\s*url\s*,\s*['\"]_blank['\"]\s*,\s*['\"]noopener,noreferrer['\"]\s*\)"
)


def test_open_full_dialog_uses_noopener() -> None:
    body = _function_body(MAIN_JS.read_text(encoding="utf-8"), "openFullDialog")
    assert _NOOPENER_OPEN.search(body), (
        "openFullDialog must open the snapshot tab with "
        "window.open(url, '_blank', 'noopener,noreferrer') so the tab lives in "
        "its own browsing-context group (separate renderer) and can never stall "
        f"the main UI. Got:\n{body}"
    )


def test_open_agent_stats_uses_noopener() -> None:
    body = _function_body(MAIN_JS.read_text(encoding="utf-8"), "openAgentStats")
    assert _NOOPENER_OPEN.search(body), f"openAgentStats must use noopener,noreferrer. Got:\n{body}"


def test_open_full_dialog_has_no_live_update_primitives() -> None:
    body = _function_body(MAIN_JS.read_text(encoding="utf-8"), "openFullDialog")
    for forbidden in ("setInterval", "setTimeout", "WebSocket", "EventSource", "fetch("):
        assert forbidden not in body, f"openFullDialog must not use {forbidden!r}: the tab is a static snapshot"


def test_dead_dialog_html_is_gone() -> None:
    assert not (STATIC_DIR / "dialog.html").exists(), (
        "static/dialog.html is a dead JS-driven dialog page (fetch + Refresh); "
        "it contradicts the static-snapshot contract and must be deleted"
    )


def test_nothing_under_static_mentions_dialog_html() -> None:
    offenders = []
    for p in STATIC_DIR.rglob("*"):
        if p.is_file() and p.suffix in {".html", ".js", ".css"}:
            if "dialog.html" in p.read_text(encoding="utf-8", errors="replace"):
                offenders.append(str(p.relative_to(STATIC_DIR)))
    assert offenders == [], f"references to the deleted dialog.html remain: {offenders}"
