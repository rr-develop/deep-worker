"""Codex stage-A TimeoutError must use ``build_dispatch_timeout_notice`` —
symmetric to claude. See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.

Prior state: on ``asyncio.TimeoutError`` from ``wait_for_pending_calls`` the
handler emitted ``build_in_progress_notice(job_id=it.id or it.tool, …)``,
promising ``wait_for_job``/``kill_job`` for work that was never parked. This
suite pins the code shape so a regression is caught by a fast unit-level
assertion.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import codex_provider.chat_model as chat_model_mod  # noqa: E402


def _timeout_handler_calls() -> list[str]:
    """Return function names invoked inside every ``except`` block whose type
    ends in ``TimeoutError`` in codex_provider.chat_model. Codex uses
    ``asyncio.TimeoutError`` — the AST sees it as ``Attribute(attr="TimeoutError")``.
    Empty means the block is gone (regression)."""
    src = Path(chat_model_mod.__file__).read_text()
    tree = ast.parse(src)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        etype = node.type
        if etype is None:
            continue
        name = (
            etype.id if isinstance(etype, ast.Name)
            else etype.attr if isinstance(etype, ast.Attribute)
            else None
        )
        if name != "TimeoutError":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    calls.append(fn.id)
                elif isinstance(fn, ast.Attribute):
                    calls.append(fn.attr)
    return calls


def test_stage_a_timeout_handler_uses_new_notice() -> None:
    calls = _timeout_handler_calls()
    assert calls, (
        "no `except (asyncio.)TimeoutError` handler found in codex chat_model"
    )
    assert "build_dispatch_timeout_notice" in calls, (
        "codex stage-A TimeoutError must call build_dispatch_timeout_notice "
        "(not build_in_progress_notice — that would advertise a job_id and "
        "wait_for_job/kill_job for work that was never parked)"
    )
    assert "build_in_progress_notice" not in calls, (
        "build_in_progress_notice must not be used on the stage-A path — it "
        "promises wait_for_job/kill_job that cannot be honoured"
    )
